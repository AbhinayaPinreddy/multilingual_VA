from faster_whisper import WhisperModel
from langdetect import detect as langdetect_detect
from langdetect.lang_detect_exception import LangDetectException

import config

model = WhisperModel(config.WHISPER_MODEL, compute_type="int8")

# Whisper uses ISO 639-1; langdetect may return longer codes — normalize to Whisper-style.
_LANG_ALIASES = {
    "zh-cn": "zh",
    "zh-tw": "zh",
}


def _normalize_lang(code: str) -> str:
    c = (code or "en").lower().strip()
    return _LANG_ALIASES.get(c, c.split("-")[0] if "-" in c else c)

def _script_lang_hint(text: str) -> str | None:
    # Prefer script hints when text is mixed or Whisper is uncertain.
    for ch in text:
        o = ord(ch)
        # Telugu
        if 0x0C00 <= o <= 0x0C7F:
            return "te"
        # Devanagari (Hindi/Marathi/etc.) -> hi
        if 0x0900 <= o <= 0x097F:
            return "hi"
        # Gurmukhi (Punjabi)
        if 0x0A00 <= o <= 0x0A7F:
            return "pa"
        # Gujarati
        if 0x0A80 <= o <= 0x0AFF:
            return "gu"
        # Bengali
        if 0x0980 <= o <= 0x09FF:
            return "bn"
        # Tamil
        if 0x0B80 <= o <= 0x0BFF:
            return "ta"
        # Kannada
        if 0x0C80 <= o <= 0x0CFF:
            return "kn"
        # Malayalam
        if 0x0D00 <= o <= 0x0D7F:
            return "ml"
        # Arabic (often Urdu)
        if 0x0600 <= o <= 0x06FF:
            return "ur"
    return None


def _refine_language(text: str, whisper_lang: str, prob: float) -> str:
    w = _normalize_lang(whisper_lang)
    if len(text) < 12:
        return w

    hint = _script_lang_hint(text)
    if hint and prob < 0.90:
        return hint

    # Heuristic: if transcript is mostly ASCII letters/spaces, treat as English unless Whisper is very confident.
    # This fixes cases like "Do you have cotton kurtas?" being tagged as hi.
    ascii_letters = sum(1 for c in text if ("a" <= c.lower() <= "z") or c in " ',-.?!")
    ascii_ratio = ascii_letters / max(1, len(text))
    # Do NOT override if Whisper already strongly suggests Hindi/Urdu/Punjabi (often romanized).
    if w in ("hi", "ur", "pa") and prob >= 0.65:
        return w

    if w != "en" and ascii_ratio >= 0.82 and prob < 0.85:
        # Detect romanized Hindi/Urdu/Punjabi cues (Hinglish) to avoid forcing English.
        t = text.lower()
        roman_hi_cues = (
            "aap", "apke", "aapke", "tum", "kya", "ky", "hai", "hain", "nahi",
            "ka", "ki", "ke", "under", "rupaye", "rupya", "rs", "price", "prais",
            "kurti", "kurta", "saari", "saree", "lehenga", "dupatta",
        )
        if sum(1 for c in roman_hi_cues if c in t) >= 2:
            return "hi"
        return "en"

    if prob >= config.LANG_CONFIDENCE_MIN:
        return w
    try:
        ld = _normalize_lang(langdetect_detect(text))
    except LangDetectException:
        return w
    # Only override Whisper when it is very unsure (avoids flipping English/Hindi).
    if prob < 0.4:
        return ld
    return w


def transcribe(audio_path):
    # vad_filter: trim silence; beam_size=1: lower latency.
    segments, info = model.transcribe(
        audio_path,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=400),
        beam_size=1,
        condition_on_previous_text=False,
    )
    text = " ".join([s.text for s in segments])
    text = text.strip()
    lang = _refine_language(
        text,
        info.language,
        getattr(info, "language_probability", 1.0),
    )
    # Debug signal for tuning language/barge-in; keep short.
    try:
        lp = getattr(info, "language_probability", None)
        if lp is not None:
            print(f" STT lang={info.language} prob={lp:.2f} refined={lang}")
    except Exception:
        pass
    return text, lang
