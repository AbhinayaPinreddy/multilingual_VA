import asyncio
import concurrent.futures
import functools
import math
import audioop
import os
import requests
import tempfile
import threading
import wave

import av
from livekit import rtc

import config
from stt import transcribe
from tts import speak
from translator import to_english, from_english
from retriever import retrieve
from llm import generate

# LiveKit outgoing audio (48 kHz mono).
AGENT_PLAYBACK_SR = 48000
AGENT_PLAYBACK_CH = 1

_agent_audio_source: rtc.AudioSource | None = None

# Barge-in / supersession: increment to cancel stale LLM+TTS work.
_pipeline_gen = 0
# Playback only: increment to stop streaming frames to LiveKit.
_play_gen = 0
_agent_speaking = False

# CPU-heavy: Whisper STT, sentence-transformers + Qdrant in retrieve, ffmpeg/av decode.
_CPU_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="va_cpu",
)
# Network-bound: Groq, Google Translate — separate pool so they do not block CPU workers.
_IO_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=6,
    thread_name_prefix="va_io",
)

# faster_whisper / CTranslate2: one decode at a time on the shared model.
_STT_LOCK = threading.Lock()


async def _run_cpu(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    if kwargs:
        return await loop.run_in_executor(_CPU_POOL, functools.partial(fn, *args, **kwargs))
    return await loop.run_in_executor(_CPU_POOL, functools.partial(fn, *args))


async def _run_io(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    if kwargs:
        return await loop.run_in_executor(_IO_POOL, functools.partial(fn, *args, **kwargs))
    return await loop.run_in_executor(_IO_POOL, functools.partial(fn, *args))


def _transcribe_locked(audio_path: str):
    with _STT_LOCK:
        return transcribe(audio_path)


def get_token():
    res = requests.get(
        config.TOKEN_SERVER_URL,
        params={
            "identity": config.AGENT_IDENTITY,
            "room": config.LIVEKIT_ROOM,
        },
        timeout=30,
    )
    res.raise_for_status()
    return res.json()["token"]


def build_context(results):
    if not results:
        return "No products found."

    return "\n".join([
        f"{p['name']} ₹{p['price']} - {p['description']}"
        for p in results
    ])


def _pcm16le_rms(pcm: bytes) -> float:
    # Use audioop (C-accelerated) to avoid Python-level per-sample work.
    # This significantly reduces CPU and helps prevent LiveKit queue overflow.
    if len(pcm) < 2:
        return 0.0
    try:
        return float(audioop.rms(pcm, 2))
    except Exception:
        return 0.0


def _chunk_duration_ms(pcm: bytes, sample_rate: int, channels: int) -> float:
    if sample_rate <= 0 or channels <= 0:
        return 0.0
    samples = len(pcm) // (2 * channels)
    return 1000.0 * samples / float(sample_rate)


def mp3_to_pcm48_mono(path: str) -> bytes:
    out = bytearray()
    with av.open(path) as container:
        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(
            format="s16", layout="mono", rate=AGENT_PLAYBACK_SR
        )
        for packet in container.demux(stream):
            for frame in packet.decode():
                for rf in resampler.resample(frame):
                    arr = rf.to_ndarray()
                    if arr.ndim == 2:
                        arr = arr[0]
                    out.extend(arr.tobytes())
    return bytes(out)


def _barge_in() -> None:
    """User started speaking over the agent — stop TTS and invalidate in-flight reply."""
    global _pipeline_gen, _play_gen
    _pipeline_gen += 1
    _play_gen += 1
    src = _agent_audio_source
    if src is not None:
        src.clear_queue()
    print(" Barge-in: stopped agent audio; listening for your turn.")


def _begin_new_user_turn() -> int:
    """Finalize a user speech segment: stop any playback and assign a new pipeline generation."""
    global _pipeline_gen, _play_gen
    # Bump play gen on every new utterance so TTS stops even if barge-in RMS gate missed (echo, quiet speech).
    _play_gen += 1
    src = _agent_audio_source
    if src is not None:
        src.clear_queue()
    _pipeline_gen += 1
    return _pipeline_gen


async def play_mp3_to_livekit(source: rtc.AudioSource, mp3_path: str, play_snapshot: int) -> None:
    global _agent_speaking
    pcm = await _run_cpu(mp3_to_pcm48_mono, mp3_path)
    if not pcm:
        print(" No PCM decoded from TTS; skipping playback")
        return
    if play_snapshot != _play_gen:
        print(" Playback cancelled before start (barge-in)")
        return

    source.clear_queue()
    _agent_speaking = True
    try:
        # Use 20ms frames to reduce Python overhead and avoid queue overflow.
        samples_per_ch = 960  # 20 ms @ 48 kHz
        frame_bytes = samples_per_ch * AGENT_PLAYBACK_CH * 2
        offset = 0
        print(f" Playback started ({len(pcm)} bytes PCM)")
        while offset < len(pcm):
            if play_snapshot != _play_gen:
                print(" Playback interrupted mid-stream (barge-in)")
                return
            chunk = pcm[offset : offset + frame_bytes]
            if len(chunk) < frame_bytes:
                chunk = chunk + b"\x00" * (frame_bytes - len(chunk))
            frame = rtc.AudioFrame(
                chunk,
                AGENT_PLAYBACK_SR,
                AGENT_PLAYBACK_CH,
                samples_per_ch,
            )
            try:
                await source.capture_frame(frame)
            except Exception as e:
                # If LiveKit drops/overflows, log it and stop this playback.
                print(" capture_frame failed:", repr(e))
                return
            offset += frame_bytes
            await asyncio.sleep(0.019)
        print(" Playback finished")
    finally:
        _agent_speaking = False


_JUNK_TRANSCRIPTS = frozenset(
    x.lower()
    for x in (
        "you",
        "you.",
        "uh",
        "oh",
        "mm",
        "mmm",
        "thanks for watching.",
        "subscribe",
    )
)


def _pipeline_stale(snap: int) -> bool:
    return snap != _pipeline_gen


async def _process_utterance(audio_path: str, pipeline_snap: int) -> None:
    mp3_path: str | None = None
    try:
        if _pipeline_stale(pipeline_snap):
            return

        text, lang = await _run_cpu(_transcribe_locked, audio_path)

        if _pipeline_stale(pipeline_snap):
            return

        t = (text or "").strip()
        if not t:
            return
        if len(t) <= 4 and t.lower() in _JUNK_TRANSCRIPTS:
            print("⏭ Skipping junk / noise transcript:", repr(t))
            return

        print(" User:", t, "| Lang:", lang)

        if lang == "en":
            query_en = t
            results = await _run_cpu(retrieve, t)
        else:
            query_en, results = await asyncio.gather(
                _run_io(to_english, t, lang),
                _run_cpu(retrieve, t),
            )

        if _pipeline_stale(pipeline_snap):
            return

        context = build_context(results)
        answer_en = await _run_io(generate, context, query_en)

        if _pipeline_stale(pipeline_snap):
            return

        final = await _run_io(from_english, answer_en, lang)

        if _pipeline_stale(pipeline_snap):
            return

        print(" Bot:", final)

        mp3_path = await speak(final, lang)
        print(" TTS file:", mp3_path)

        if _pipeline_stale(pipeline_snap):
            return

        play_snap = _play_gen
        src = _agent_audio_source
        if src is not None:
            await play_mp3_to_livekit(src, mp3_path, play_snap)
            print(" Agent audio sent to the meeting")
        else:
            print(" No AudioSource; only saved file above")
    except Exception as e:
        print(" Utterance pipeline error:", repr(e))
    finally:
        try:
            os.unlink(audio_path)
        except OSError:
            pass
        if mp3_path:
            try:
                os.unlink(mp3_path)
            except OSError:
                pass


def _safe_create_utterance_task(coro):
    async def _runner():
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(" Task error:", repr(e))

    return asyncio.create_task(_runner())


def _write_wav(path: str, pcm: bytes, sample_rate: int, channels: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


async def handle_audio(room, track):
    print(" Track received")

    if not isinstance(track, rtc.RemoteAudioTrack):
        print(" Not an audio track, skipping...")
        return

    print(" Audio track confirmed")

    stream = rtc.AudioStream(track)

    sample_rate = 16000
    channels = 1

    # Slightly lower thresholds so quieter speech still triggers (helps “no output sometimes”).
    speech_start_rms = 78.0
    silence_rms = 55.0
    end_silence_ms = 520.0
    min_utterance_ms = 380.0
    max_utterance_ms = 18000.0
    min_flush_rms = 62.0

    speech_buf = bytearray()
    in_speech = False
    silence_ms = 0.0
    # While the agent is speaking, the mic may pick up TTS (echo). Keep barge-in above echo,
    # but low enough that a real interrupt stops playback quickly.
    barge_rms = 155.0
    barge_min_ms = 180.0
    barge_ms = 0.0

    async for event in stream:
        frame = getattr(event, "frame", event)
        chunk = getattr(frame, "data", None)

        if chunk is None:
            continue

        if isinstance(chunk, memoryview):
            chunk = chunk.tobytes()
        elif not isinstance(chunk, (bytes, bytearray)):
            chunk = bytes(chunk)

        if hasattr(frame, "sample_rate"):
            sample_rate = frame.sample_rate
        if hasattr(frame, "num_channels"):
            channels = frame.num_channels

        dur_ms = _chunk_duration_ms(chunk, sample_rate, channels)
        rms = _pcm16le_rms(chunk)

        if not in_speech:
            if rms >= speech_start_rms:
                # Debug: speech start gate passed
                # (kept minimal; useful when users report “no output sometimes”)
                # print(f\" Speech start rms={rms:.1f}\")
                if _agent_speaking:
                    # Only barge-in if we see sustained loud speech-like energy.
                    if rms >= barge_rms:
                        barge_ms += dur_ms
                    else:
                        barge_ms = 0.0

                    if barge_ms < barge_min_ms:
                        continue

                    _barge_in()

                barge_ms = 0.0
                in_speech = True
                speech_buf.clear()
                speech_buf += chunk
                silence_ms = 0.0
            continue

        speech_buf += chunk

        if rms < silence_rms:
            silence_ms += dur_ms
        else:
            silence_ms = 0.0

        ut_ms = _chunk_duration_ms(bytes(speech_buf), sample_rate, channels)

        should_flush = (
            silence_ms >= end_silence_ms and ut_ms >= min_utterance_ms
        ) or (ut_ms >= max_utterance_ms)

        if not should_flush:
            continue

        pcm = bytes(speech_buf)
        speech_buf.clear()
        in_speech = False
        silence_ms = 0.0

        overall_rms = _pcm16le_rms(pcm)
        if overall_rms < min_flush_rms:
            continue

        pipeline_snap = _begin_new_user_turn()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        _write_wav(wav_path, pcm, sample_rate, channels)
        _safe_create_utterance_task(_process_utterance(wav_path, pipeline_snap))


async def _run_session(room: rtc.Room) -> None:
    global _agent_audio_source

    disconnect_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def on_disconnected(reason):
        print(" LiveKit disconnected:", reason)
        loop.call_soon_threadsafe(disconnect_event.set)

    def on_reconnecting():
        print(" LiveKit reconnecting...")

    def on_reconnected():
        print(" LiveKit reconnected")

    room.on("disconnected", on_disconnected)
    room.on("reconnecting", on_reconnecting)
    room.on("reconnected", on_reconnected)

    token = get_token()
    await room.connect(
        config.LIVEKIT_URL,
        token,
        rtc.RoomOptions(auto_subscribe=True),
    )

    _agent_audio_source = rtc.AudioSource(AGENT_PLAYBACK_SR, AGENT_PLAYBACK_CH)
    agent_track = rtc.LocalAudioTrack.create_audio_track("agent-voice", _agent_audio_source)
    pub_opts = rtc.TrackPublishOptions()
    pub_opts.source = rtc.TrackSource.SOURCE_MICROPHONE
    await room.local_participant.publish_track(agent_track, pub_opts)
    print(" Agent running... Mic published for TTS; waiting for user audio...")

    @room.on("track_subscribed")
    def on_track(track, publication, participant):
        if str(participant.identity) == str(room.local_participant.identity):
            return
        print(" Track subscribed")

        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(handle_audio(room, track))
        else:
            print("⏭ Skipping non-audio track")

    await disconnect_event.wait()


async def main():
    while True:
        room = rtc.Room()
        try:
            await _run_session(room)
        except Exception as e:
            print(" Session error:", repr(e))
        finally:
            try:
                await room.disconnect()
            except Exception:
                pass
        print(f" Reconnecting in {config.LIVEKIT_RECONNECT_DELAY}s...")
        await asyncio.sleep(config.LIVEKIT_RECONNECT_DELAY)


if __name__ == "__main__":
    asyncio.run(main())
