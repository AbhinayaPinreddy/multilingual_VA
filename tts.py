import re
import tempfile

import edge_tts
from edge_tts.exceptions import NoAudioReceived

# Microsoft Edge neural voices (India) — extend so STT lang codes map to a valid voice.
VOICE_MAP = {
    "en": "en-IN-NeerjaNeural",
    "hi": "hi-IN-SwaraNeural",
    "te": "te-IN-ShrutiNeural",
    "ta": "ta-IN-PallaviNeural",
    "kn": "kn-IN-SapnaNeural",
    "ml": "ml-IN-SobhanaNeural",
    "mr": "mr-IN-AarohiNeural",
    "gu": "gu-IN-DhwaniNeural",
    "bn": "bn-IN-TanishaaNeural",
    "pa": "pa-IN-VaaniNeural",
    "ur": "ur-IN-GulNeural",
}

_DEFAULT_VOICE = "en-IN-NeerjaNeural"
# When a language-specific voice fails (Edge / script issues), try Hindi then English India.
_FALLBACK_CHAIN = ("hi-IN-SwaraNeural", "en-IN-NeerjaNeural")


def _has_indic_script(s: str) -> bool:
    return bool(re.search(r"[\u0900-\u0D7F]", s))


async def _save_tts(text: str, voice: str, path: str) -> None:
    com = edge_tts.Communicate(text, voice)
    await com.save(path)


async def generate_audio(text: str, lang: str, path: str) -> None:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty TTS text")

    primary = VOICE_MAP.get(lang, VOICE_MAP.get("hi") if _has_indic_script(text) else _DEFAULT_VOICE)
    voices_to_try = [primary]
    for v in _FALLBACK_CHAIN:
        if v not in voices_to_try:
            voices_to_try.append(v)

    last_err: Exception | None = None
    for voice in voices_to_try:
        try:
            await _save_tts(text, voice, path)
            return
        except NoAudioReceived as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            continue

    if last_err:
        raise last_err
    raise RuntimeError("TTS failed with no exception detail")


async def speak(text: str, lang: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        path = tmp.name

    await generate_audio(text, lang, path)
    return path
