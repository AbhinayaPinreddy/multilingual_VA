# Multilingual RAG Voice Assistant (LiveKit)

Voice shopping assistant that joins a **LiveKit** room, listens to participants, runs **STT → RAG → Groq LLM → TTS**, and plays the reply back into the same call. Supports **English, Hindi, Telugu** (and more via Whisper + translation), with product data from `products.json`.

## Will it work on Linux?

**Yes.** The stack is standard Python (3.10+ recommended) with cross-platform wheels:

| Component | Linux notes |
|-----------|-------------|
| **Python / venv** | Same as on Windows. |
| **faster-whisper** | Uses `ctranslate2` / ONNX; works on Linux CPU or GPU. |
| **sentence-transformers** | Works; first run may download models from Hugging Face. |
| **LiveKit (`livekit`, `livekit-api`)** | Official SDK supports Linux. |
| **PyAV (`av`)** | Usually installs via pip; on minimal distros you may need `ffmpeg` dev packages if build fails. |
| **Edge TTS** | Network-based; no local engine required. |
| **Groq / Google Translate** | Requires internet. |

Use `python3` and `source venv/bin/activate` on Linux. If anything fails to import, install missing packages (see [Dependencies](#dependencies)).

---

## Architecture

```
Participant (browser/app) ── WebRTC ──► LiveKit Cloud/Self-hosted
                                              │
                                              ▼
                                    livekit_agent.py (bot)
                                    • Subscribes to user audio
                                    • Faster Whisper STT + VAD
                                    • Multilingual RAG (embeddings)
                                    • Groq LLM
                                    • Translate + Edge TTS
                                    • Publishes TTS as local audio track

token_server.py (FastAPI) ── issues JWTs for LiveKit (room join).
```

---

## Features

- **Speech-to-text**: faster-whisper (configurable model, VAD).
- **RAG**: `sentence-transformers` + `embeddings.npy` built from `products.json`.
- **LLM**: Groq (`llama-3.1-8b-instant`).
- **Translation**: `deep-translator` (Google) for non-English flows.
- **TTS**: Edge TTS with Indian English / Hindi / Telugu voices.
- **Latency tuning**: env vars for Whisper size, RAG `top_k`, LLM `max_tokens` (see [Configuration](#configuration)).

---

## Project layout

| File | Role |
|------|------|
| `livekit_agent.py` | Main bot: connect to room, audio in/out, full pipeline. |
| `token_server.py` | FastAPI server: `/get-token` for LiveKit JWTs. |
| `stt.py` | Whisper transcription. |
| `retriever.py` | Embedding similarity over products. |
| `llm.py` | Groq chat completion. |
| `translator.py` | To/from English. |
| `tts.py` | Edge TTS, voice map per language. |
| `embedder.py` | One-off script: rebuild `embeddings.npy` after editing `products.json`. |
| `config.py` | URLs, latency-related env defaults. |
| `products.json` | Product catalog. |

---

## Prerequisites

- **Python 3.10+** (3.11 or 3.12 is fine).
- **LiveKit** project: URL, API key, and API secret (for tokens).
- **Groq API key** for the LLM.
- **Internet** for Groq, translation, TTS, and (first run) model downloads.

---

## Dependencies

Install from the project directory:

```bash
pip install -r requirements.txt
```

This includes **livekit**, **livekit-api**, **fastapi**, and **uvicorn** for the agent and token server.

---

## Configuration

Create a `.env` file in the project root (same folder as `livekit_agent.py`):

```env
# Required
GROQ_API_KEY=your_groq_key
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_key
LIVEKIT_API_SECRET=your_livekit_secret

# Optional: latency / quality (defaults are in config.py)
# WHISPER_MODEL=base
# RAG_TOP_K=3
# LLM_MAX_TOKENS=72
```

`config.py` sets `TOKEN_SERVER_URL` to `http://127.0.0.1:8000/get-token` by default. Change it if the token server runs on another host/port.

---

## One-time: build embeddings

After changing `products.json`:

```bash
python embedder.py
```

This overwrites `embeddings.npy` used by `retriever.py`.

---

## Run (two terminals)

**1. Token server**

```bash
uvicorn token_server:app --host 127.0.0.1 --port 8000
```

**2. Voice agent**

```bash
python livekit_agent.py
```

Join the same LiveKit room from a client (e.g. LiveKit Meet or your app) as a **different identity** than the bot (`agent1` in code). Speak; the agent should log transcripts and play TTS into the room.

---

## Linux quick start

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip
# Optional if PyAV or audio tooling needs it:
# sudo apt install -y ffmpeg

python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Then set `.env`, run `embedder.py` if needed, start `uvicorn` and `livekit_agent.py` as above.

---

## Windows quick start

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

---

## Troubleshooting

- **`native audio stream queue overflow`**: Heavy CPU load; latency settings in `config.py` / STT options help. Close other heavy apps.
- **No bot audio in the meeting**: Ensure `AudioSource.capture_frame` is **awaited** (current code does); check client volume and that the agent track is subscribed.
- **Wrong language / voice**: Whisper language detection can confuse similar languages; you can force `language=` in `stt.py` for a fixed locale.
- **`main.py`**: Legacy demo that imports `listen` from `stt`; the maintained entry point for voice in meetings is **`livekit_agent.py`**.

---
## Collaboration

This project was created in collaboration with [Superteams.ai](https://superteams.ai).

The collaboration involved contributing technical content, sharing insights, and building projects/blogs focused on learning, innovation, and community engagement in the tech ecosystem.

