"""
Small multilingual text signals shared by role detection and analysis.

Deliberately lexical, not learned. Whisper output for Malayalam classroom
audio is noisy enough that anything cleverer would be fitting to noise, and
a wordlist is at least auditable when a number looks wrong.
"""

import re

# Whisper punctuates well enough that "?" carries most of the signal. The
# wordlists are the backstop for when it doesn't.
QUESTION_WORDS = {
    "ml": [
        "എന്ത", "എന്താ", "ഏത", "ആര", "എവിടെ", "എപ്പോ", "എങ്ങനെ",
        "എന്തുകൊണ്ട", "എത്ര", "ഉണ്ടോ", "ആണോ", "അല്ലേ", "അറിയാമോ",
        "മനസ്സിലായോ", "ശരിയാണോ", "പറയാമോ",
    ],
    "hi": ["क्या", "कौन", "कहाँ", "कहां", "कब", "कैसे", "क्यों", "कितना", "कितने", "ना?"],
    "ta": ["என்ன", "யார்", "எங்கே", "எப்போது", "எப்படி", "ஏன்", "எத்தனை", "இல்லையா"],
    "en": ["what", "why", "how", "when", "where", "who", "which", "whose",
           "do you", "did you", "can you", "could you", "is it", "are you",
           "anyone", "right?", "okay?", "understand"],
}

# Phrases a teacher says and a student essentially never does: directives to
# the room, references to the lesson apparatus, comprehension checks.
TEACHER_CUES = {
    "ml": [
        "കുട്ടികളേ", "കുട്ടികള", "ശ്രദ്ധി", "എഴുതൂ", "എഴുതു", "നോക്കൂ", "നോക്ക",
        "പുസ്തക", "പേജ", "മനസ്സിലായോ", "കേൾക്ക", "പറയൂ", "വായിക്ക", "ക്ലാസ",
        "ഹോംവർക", "ഉത്തരം", "ചോദ്യം", "ഉദാഹരണ", "അധ്യായ",
    ],
    "hi": ["बच्चों", "ध्यान", "लिखो", "देखो", "किताब", "पन्ना", "समझे", "सुनो",
           "पढ़ो", "कक्षा", "उत्तर", "प्रश्न", "उदाहरण", "अध्याय"],
    "ta": ["குழந்தைகளே", "கவனி", "எழுது", "பார்", "புத்தகம்", "பக்கம்",
           "புரிந்ததா", "படி", "வகுப்பு", "பதில்", "கேள்வி", "உதாரணம்"],
    "en": ["class", "children", "everyone", "listen", "look at", "write down",
           "open your", "page", "notebook", "textbook", "homework", "chapter",
           "example", "answer the", "quiet", "pay attention", "let's", "repeat after"],
}

# Short acknowledgements. Teachers use them as feedback moves; a speaker made
# almost entirely of these is a student answering, not one teaching.
PRAISE_CUES = {
    "ml": ["നന്നായി", "കൊള്ളാം", "ശരി", "മിടുക്ക", "കറക്റ്റ"],
    "hi": ["शाबाश", "बहुत अच्छा", "सही", "ठीक"],
    "ta": ["நல்லது", "சரி", "மிகவும் நன்று"],
    "en": ["good", "very good", "well done", "excellent", "correct", "exactly",
           "right", "perfect", "nice", "great"],
}

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Unicode block each language should actually come back in. Whisper does not
# fail when a model is too small for a language - it emits confident English
# or romanised text instead, which reads like a transcript and is not one.
SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F),      # Devanagari
    "mr": (0x0900, 0x097F),
    "ne": (0x0900, 0x097F),
    "bn": (0x0980, 0x09FF),      # Bengali
    "pa": (0x0A00, 0x0A7F),      # Gurmukhi
    "gu": (0x0A80, 0x0AFF),      # Gujarati
    "or": (0x0B00, 0x0B7F),      # Odia
    "ta": (0x0B80, 0x0BFF),      # Tamil
    "te": (0x0C00, 0x0C7F),      # Telugu
    "kn": (0x0C80, 0x0CFF),      # Kannada
    "ml": (0x0D00, 0x0D7F),      # Malayalam
    "si": (0x0D80, 0x0DFF),      # Sinhala
    "ur": (0x0600, 0x06FF),      # Arabic script
    "ar": (0x0600, 0x06FF),
}


def script_ratio(text: str, language: str) -> float | None:
    """
    Share of letters in the script this language is written in.

    None when we have no expectation for the language (English and the rest
    of the Latin-script set). Otherwise a number near 1.0 means the model
    wrote the language; near 0.0 means it produced something else entirely -
    which is what a too-small model does with Indian languages.
    """
    span = SCRIPT_RANGES.get(language)
    if not span:
        return None
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None
    lo, hi = span
    return round(sum(1 for c in letters if lo <= ord(c) <= hi) / len(letters), 4)


def _cues(table: dict, language: str) -> list[str]:
    """Always include English - classroom speech is code-mixed."""
    cues = list(table.get(language, []))
    if language != "en":
        cues += table["en"]
    return cues


def is_question(text: str, language: str = "ml") -> bool:
    t = text.strip()
    if not t:
        return False
    if "?" in t:
        return True
    low = t.lower()
    return any(c in low for c in _cues(QUESTION_WORDS, language))


def count_cues(text: str, table: dict, language: str = "ml") -> int:
    low = text.lower()
    return sum(1 for c in _cues(table, language) if c in low)


def teacher_cue_score(text: str, language: str = "ml") -> int:
    return count_cues(text, TEACHER_CUES, language)


def praise_score(text: str, language: str = "ml") -> int:
    return count_cues(text, PRAISE_CUES, language)


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def type_token_ratio(text: str) -> float:
    w = [x.lower() for x in words(text)]
    return round(len(set(w)) / len(w), 4) if w else 0.0
