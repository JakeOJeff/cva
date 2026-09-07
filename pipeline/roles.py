"""
Anonymous speakers -> teacher and students.

pyannote hands back SPEAKER_00, SPEAKER_01... with no idea which is which.
Deciding who is teaching is the one genuinely inferential step in this
pipeline, so it is scored out of explicit, inspectable features rather than
hidden in a model. Every score comes with the reasons that produced it.

The signals, in rough order of how much they carry:
  talk_share  - teaching is asymmetric; the teacher holds most of the floor
  coverage    - the teacher is present start to finish, students come and go
  hub         - IRF structure makes the teacher one side of most turn changes
  cue_rate    - "open your book", "മനസ്സിലായോ" - said to a room, not to a teacher
  question    - the teacher asks; students mostly answer
  avg_turn    - explanation is long, answers are short
"""

from . import lang

WEIGHTS = {
    "talk_share": 0.30,
    "coverage": 0.20,
    "hub": 0.15,
    "cue_rate": 0.15,
    "question_rate": 0.10,
    "avg_turn": 0.10,
}


def _normalize(values: dict[str, float]) -> dict[str, float]:
    """Scale to 0..1 against the strongest speaker on that feature."""
    top = max(values.values(), default=0.0)
    if top <= 0:
        return {k: 0.0 for k in values}
    return {k: v / top for k, v in values.items()}


def _hub_scores(utterances) -> dict[str, float]:
    """Share of speaker changes each speaker takes part in."""
    counts: dict[str, int] = {}
    changes = 0
    for a, b in zip(utterances, utterances[1:]):
        if a["speaker"] != b["speaker"]:
            changes += 1
            counts[a["speaker"]] = counts.get(a["speaker"], 0) + 1
            counts[b["speaker"]] = counts.get(b["speaker"], 0) + 1
    if not changes:
        return {u["speaker"]: 0.0 for u in utterances}
    return {sp: n / (2 * changes) for sp, n in counts.items()}


def score_speakers(utterances, stats, language: str = lang.DEFAULT_LANGUAGE, duration=None):
    """Returns {speaker: {"score": float, "features": {...}, "reasons": [...]}}."""
    speakers = list(stats)
    hub = _hub_scores(utterances)

    raw = {f: {} for f in WEIGHTS}
    per_speaker_detail = {}

    for sp in speakers:
        mine = [u for u in utterances if u["speaker"] == sp]
        n_turns = max(len(mine), 1)
        n_words = max(sum(len(lang.words(u["text"])) for u in mine), 1)

        n_questions = sum(1 for u in mine if lang.is_question(u["text"], language))
        n_cues = sum(lang.teacher_cue_score(u["text"], language) for u in mine)

        raw["talk_share"][sp] = stats[sp]["talk_share"]
        raw["coverage"][sp] = stats[sp].get("coverage", 0.0)
        raw["hub"][sp] = hub.get(sp, 0.0)
        raw["cue_rate"][sp] = n_cues / n_words * 100
        raw["question_rate"][sp] = n_questions / n_turns
        raw["avg_turn"][sp] = stats[sp]["avg_turn"]

        per_speaker_detail[sp] = {
            "n_questions": n_questions,
            "n_teacher_cues": n_cues,
            "n_words": n_words,
            "n_turns": n_turns,
        }

    normed = {f: _normalize(vals) for f, vals in raw.items()}

    results = {}
    for sp in speakers:
        features = {f: round(raw[f][sp], 4) for f in WEIGHTS}
        score = sum(WEIGHTS[f] * normed[f][sp] for f in WEIGHTS)
        results[sp] = {
            "score": round(score, 4),
            "features": features,
            "normalized": {f: round(normed[f][sp], 3) for f in WEIGHTS},
            "detail": per_speaker_detail[sp],
            "reasons": _reasons(sp, raw, normed, per_speaker_detail[sp]),
        }
    return dict(sorted(results.items(), key=lambda kv: -kv[1]["score"]))


def _reasons(sp, raw, normed, detail) -> list[str]:
    out = []
    if normed["talk_share"][sp] >= 0.95:
        out.append(f"holds the most floor time ({raw['talk_share'][sp]:.0%} of all speech)")
    if normed["coverage"][sp] >= 0.9:
        out.append(f"present across {raw['coverage'][sp]:.0%} of the session")
    if normed["hub"][sp] >= 0.9:
        out.append(f"party to {raw['hub'][sp]:.0%} of all speaker changes")
    if detail["n_teacher_cues"] > 0 and normed["cue_rate"][sp] >= 0.9:
        out.append(f"uses the most instructional language ({detail['n_teacher_cues']} cues)")
    if detail["n_questions"] > 0 and normed["question_rate"][sp] >= 0.9:
        out.append(f"asks the most questions ({detail['n_questions']})")
    if normed["avg_turn"][sp] >= 0.9:
        out.append(f"longest average turn ({raw['avg_turn'][sp]:.0f}s)")
    return out or ["no distinguishing signal"]


def identify_teacher(utterances, stats, language: str = lang.DEFAULT_LANGUAGE, duration=None):
    """
    Returns the full verdict, not just an id. `margin` is what you check
    before trusting it: two speakers within a few points of each other means
    a co-teaching session, a dominant student, or a diarization that split
    one person in two.
    """
    scored = score_speakers(utterances, stats, language, duration)
    if not scored:
        return {"teacher": None, "confidence": "none", "scores": {}}

    ranked = list(scored.items())
    teacher, top = ranked[0]
    runner_up = ranked[1][1]["score"] if len(ranked) > 1 else 0.0
    margin = round(top["score"] - runner_up, 4)

    if margin >= 0.20:
        confidence = "high"
    elif margin >= 0.08:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "teacher": teacher,
        "confidence": confidence,
        "margin": margin,
        "reasons": top["reasons"],
        "scores": scored,
        "method": "heuristic",
    }


UNATTRIBUTED = "UNKNOWN"       # assign.py's marker for speech it could not place


def role_of(speaker: str, teacher_id: str) -> str:
    if speaker == UNATTRIBUTED:
        return "unknown"
    return "teacher" if speaker == teacher_id else "student"


def label_roles(items, teacher_id: str):
    """
    Stamp role onto segments or utterances, plus a human label.

    Students get stable letters ordered by talk time, so "Student A" means
    the same person on every re-read of the same transcript.

    UNKNOWN is not a person. It is what assign.py writes when a segment
    overlaps no diarization turn at all, and it must never be handed a
    student letter - doing so invents a child who was never in the room and
    quietly inflates every participation number.
    """
    others = sorted(
        {i["speaker"] for i in items
         if i["speaker"] != teacher_id and i["speaker"] != UNATTRIBUTED},
        key=lambda sp: -sum(i["end"] - i["start"] for i in items if i["speaker"] == sp),
    )
    names = {teacher_id: "Teacher"}
    if any(i["speaker"] == UNATTRIBUTED for i in items):
        names[UNATTRIBUTED] = "Unattributed"
    for n, sp in enumerate(others):
        names[sp] = f"Student {chr(ord('A') + n)}" if n < 26 else f"Student {n + 1}"

    return [
        {**i,
         "role": role_of(i["speaker"], teacher_id),
         "label": names.get(i["speaker"], i["speaker"])}
        for i in items
    ], names


def split_by_role(items):
    """teacher, student. Unattributed speech is in neither - see label_roles."""
    teacher = [i for i in items if i.get("role") == "teacher"]
    student = [i for i in items if i.get("role") == "student"]
    return teacher, student


def unattributed(items):
    return [i for i in items if i.get("role") == "unknown"]
