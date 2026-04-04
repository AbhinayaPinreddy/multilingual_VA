from deep_translator import GoogleTranslator

# Google Translate target codes; map rare codes to close regional variants.
_TARGET_FALLBACK = {
    "pa": "pa",
    "doi": "hi",
    "bh": "hi",
    "mai": "hi",
    "sat": "hi",
}


def _target_lang(lang: str) -> str:
    return _TARGET_FALLBACK.get(lang, lang)


def to_english(text, lang):
    if lang == "en":
        return text
    try:
        # Prefer auto-detect for source: STT language can be wrong or text may be mixed-script.
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        try:
            return GoogleTranslator(source="auto", target="en").translate(text)
        except Exception:
            return text


def from_english(text, lang):
    if lang == "en":
        return text
    tgt = _target_lang(lang)
    try:
        return GoogleTranslator(source="en", target=tgt).translate(text)
    except Exception:
        # If translation fails, fall back to English instead of producing garbage.
        return text
