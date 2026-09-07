"""
Role-labelled utterances -> what the teacher actually did.

Two layers, deliberately separate:

  metrics()    deterministic, offline, comparable across sessions. These are
               the numbers you can trend - talk ratio, wait time, question
               rate. No model in the loop, so they never drift.

  llm_review() judgement. Claude reads the teacher's speech in context and
               says what the numbers can't: was the question worth asking,
               did the explanation land, was the feedback specific.

Numbers without judgement flatter a lecture; judgement without numbers
can't be compared week to week. The report carries both.
"""

import json
import os
import statistics

from . import lang, roles

MODEL = "claude-opus-5"
RESPONSE_WINDOW = 10.0        # seconds a student has to answer before it stops counting as a response
LONG_MONOLOGUE = 60.0         # seconds of unbroken teacher speech


# ---------------------------------------------------------------- metrics

def metrics(utterances, meta, language: str = lang.DEFAULT_LANGUAGE) -> dict:
    duration = meta.get("duration") or (utterances[-1]["end"] if utterances else 0.0)
    teacher, student = roles.split_by_role(utterances)

    t_time = sum(u["end"] - u["start"] for u in teacher)
    s_time = sum(u["end"] - u["start"] for u in student)
    # Speech the diarizer could not attribute to anyone. It is real speech, so
    # it belongs in the denominator - but it is not a student, and counting it
    # as one would overstate how much the class talked.
    u_time = sum(u["end"] - u["start"] for u in roles.unattributed(utterances))
    speech = t_time + s_time + u_time or 1.0

    t_questions = [u for u in teacher if lang.is_question(u["text"], language)]
    t_words = sum(len(lang.words(u["text"])) for u in teacher)

    wait_times, answered = _response_behaviour(utterances, t_questions)

    return {
        "duration": round(duration, 2),
        "speech_time": round(speech, 2),
        "silence_time": round(max(duration - speech, 0.0), 2),
        "silence_ratio": round(max(duration - speech, 0.0) / (duration or 1), 4),

        "teacher_talk_time": round(t_time, 2),
        "student_talk_time": round(s_time, 2),
        "unattributed_talk_time": round(u_time, 2),
        "unattributed_ratio": round(u_time / speech, 4),
        # The headline number in classroom research. Sustained above ~0.80
        # means lecture, not discussion.
        "teacher_talk_ratio": round(t_time / speech, 4),
        "student_talk_ratio": round(s_time / speech, 4),

        "teacher_turns": len(teacher),
        "student_turns": len(student),
        "teacher_avg_turn": round(t_time / len(teacher), 2) if teacher else 0.0,
        "teacher_longest_turn": round(max((u["end"] - u["start"] for u in teacher), default=0), 2),
        "long_monologues": sum(1 for u in teacher if u["end"] - u["start"] >= LONG_MONOLOGUE),

        "teacher_questions": len(t_questions),
        "questions_per_10min": round(len(t_questions) / (duration / 600), 2) if duration else 0.0,
        "questions_answered": answered,
        "question_response_rate": round(answered / len(t_questions), 4) if t_questions else 0.0,
        # Rowe's "wait time": under about 3s and students cannot formulate
        # anything beyond recall.
        "median_wait_time": round(statistics.median(wait_times), 2) if wait_times else None,

        "teacher_words": t_words,
        "teacher_words_per_min": round(t_words / (duration / 60), 1) if duration else 0.0,
        "teacher_vocabulary_diversity": lang.type_token_ratio(" ".join(u["text"] for u in teacher)),
        "praise_moves": sum(lang.praise_score(u["text"], language) for u in teacher),

        "irf_triads": _irf_triads(utterances),
        "n_students_heard": len({u["speaker"] for u in student}),
        "student_participation": _participation(student),
    }


def _response_behaviour(utterances, questions):
    """How long after a teacher question does anyone speak, and does a student?"""
    q_ends = {round(q["end"], 3) for q in questions}
    waits, answered = [], 0
    for cur, nxt in zip(utterances, utterances[1:]):
        if round(cur["end"], 3) not in q_ends or cur.get("role") != "teacher":
            continue
        gap = nxt["start"] - cur["end"]
        waits.append(max(gap, 0.0))
        if nxt.get("role") == "student" and gap <= RESPONSE_WINDOW:
            answered += 1
    return waits, answered


def _irf_triads(utterances) -> int:
    """teacher -> student -> teacher. The signature move of recitation teaching."""
    return sum(
        1 for a, b, c in zip(utterances, utterances[1:], utterances[2:])
        if a.get("role") == "teacher" and b.get("role") == "student" and c.get("role") == "teacher"
    )


def _participation(student_utts) -> dict:
    """Per-student share. One student answering everything is not participation."""
    by = {}
    for u in student_utts:
        d = by.setdefault(u.get("label", u["speaker"]), {"turns": 0, "talk_time": 0.0})
        d["turns"] += 1
        d["talk_time"] += u["end"] - u["start"]
    total = sum(d["talk_time"] for d in by.values()) or 1.0
    for d in by.values():
        d["talk_time"] = round(d["talk_time"], 2)
        d["share"] = round(d["talk_time"] / total, 4)
    return dict(sorted(by.items(), key=lambda kv: -kv[1]["talk_time"]))


# ------------------------------------------------------------------- LLM

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "topics_covered": {"type": "array", "items": {"type": "string"}},
        "questioning": {
            "type": "object",
            "properties": {
                "assessment": {"type": "string"},
                "dominant_question_type": {
                    "type": "string",
                    "enum": ["recall", "comprehension", "application", "analysis",
                             "open_ended", "mixed", "few_questions"],
                },
                "examples": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["assessment", "dominant_question_type", "examples"],
            "additionalProperties": False,
        },
        "explanation_quality": {
            "type": "object",
            "properties": {
                "assessment": {"type": "string"},
                "uses_examples": {"type": "boolean"},
                "checks_understanding": {"type": "boolean"},
            },
            "required": ["assessment", "uses_examples", "checks_understanding"],
            "additionalProperties": False,
        },
        "feedback_to_students": {
            "type": "object",
            "properties": {
                "assessment": {"type": "string"},
                "specificity": {"type": "string",
                                "enum": ["specific", "generic", "minimal", "none"]},
            },
            "required": ["assessment", "specificity"],
            "additionalProperties": False,
        },
        "classroom_management": {"type": "string"},
        "language_use": {"type": "string"},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "areas_for_improvement": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "issue": {"type": "string"},
                    "suggestion": {"type": "string"},
                },
                "required": ["issue", "suggestion"],
                "additionalProperties": False,
            },
        },
        "notable_moments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string"},
                    "what_happened": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                },
                "required": ["timestamp", "what_happened", "why_it_matters"],
                "additionalProperties": False,
            },
        },
        "transcript_quality_caveat": {"type": "string"},
    },
    "required": [
        "summary", "topics_covered", "questioning", "explanation_quality",
        "feedback_to_students", "classroom_management", "language_use",
        "strengths", "areas_for_improvement", "notable_moments",
        "transcript_quality_caveat",
    ],
    "additionalProperties": False,
}

SYSTEM = """You are an experienced classroom observer and teacher coach analysing a \
transcript of a real lesson.

The transcript comes from automatic speech recognition and automatic speaker \
diarization. Both make mistakes. Word errors are common, especially for Indian \
languages and code-mixed speech, and a line attributed to the teacher may \
occasionally be a student, or the reverse. Read through the noise: judge the \
teaching, not the transcription. Where a conclusion would depend on wording \
that looks garbled, say so rather than asserting it.

Be concrete and useful to the teacher. Ground every claim in something in the \
transcript - quote it or cite its timestamp. Do not pad with generic pedagogy \
advice that would apply to any lesson. If the evidence for something is thin, \
say that plainly instead of inventing a confident finding. Praise what \
genuinely worked; a report that finds nothing to improve is as useless as one \
that finds nothing good."""


def _fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def build_transcript(utterances, max_chars: int | None = None) -> str:
    """
    Full dialogue, not just the teacher's lines.

    Teacher speech is only assessable in context: whether a question worked
    is a fact about the answer it got.
    """
    lines = [f"[{_fmt(u['start'])}] {u.get('label', u['speaker'])}: {u['text']}"
             for u in utterances]
    text = "\n".join(lines)
    if max_chars and len(text) > max_chars:
        raise ValueError(
            f"transcript is {len(text)} chars, over the {max_chars} limit. "
            "Raise the limit or split the session - truncating would silently "
            "drop part of the lesson from the analysis."
        )
    return text


def llm_review(utterances, computed: dict, meta: dict, language: str = lang.DEFAULT_LANGUAGE,
               model: str = MODEL, effort: str = "high") -> dict:
    """
    Returns the structured review, or an {"error": ...} dict.

    Never raises for a missing key - the deterministic metrics are worth
    having on their own, and a long job should not fail at the last step.
    """
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return {"error": "no ANTHROPIC_API_KEY set; skipped LLM review", "skipped": True}

    import anthropic

    client = anthropic.Anthropic()
    transcript = build_transcript(utterances)

    prompt = f"""Here is a lesson transcript and the measurements taken from it.

<session>
language: {meta.get('language', language)}
duration: {_fmt(meta.get('duration', 0))}
speakers detected: {meta.get('n_speakers', 'unknown')}
</session>

<measurements>
{json.dumps(computed, ensure_ascii=False, indent=2)}
</measurements>

<transcript>
{transcript}
</transcript>

Analyse what the teacher did. Use the measurements as evidence where they are \
relevant - explain what a number means for this lesson rather than restating \
it. Timestamps in notable_moments must be MM:SS values that appear in the \
transcript. Write the report in English, but quote the teacher in the original \
language followed by a short English gloss in parentheses."""

    # Streaming: a full lesson transcript plus a long structured report is
    # exactly the shape that trips non-streaming HTTP timeouts.
    with client.messages.stream(
        model=model,
        max_tokens=32000,
        system=SYSTEM,
        thinking={"type": "adaptive"},
        output_config={"effort": effort,
                       "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        response = stream.get_final_message()

    if response.stop_reason == "refusal":
        return {"error": "model declined to analyse this transcript",
                "detail": getattr(response.stop_details, "explanation", None)}

    text = next((b.text for b in response.content if b.type == "text"), None)
    if not text:
        return {"error": f"no text in response (stop_reason={response.stop_reason})"}

    review = json.loads(text)
    review["_model"] = response.model
    review["_usage"] = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return review


def analyze(utterances, meta, language: str = lang.DEFAULT_LANGUAGE, use_llm: bool = True,
            model: str = MODEL) -> dict:
    computed = metrics(utterances, meta, language)
    out = {"metrics": computed}
    if use_llm:
        out["review"] = llm_review(utterances, computed, meta, language, model)
    return out
