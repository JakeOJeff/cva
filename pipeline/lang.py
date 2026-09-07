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
