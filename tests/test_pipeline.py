"""
Tests for every stage that is a pure function over segments.

Deliberately no audio and no models here: those stages take minutes and
need a GPU or a token to be interesting. Everything that decides *meaning* -
who the teacher is, what the numbers say - is pure, and this is where the
bugs would actually hurt.

    python -m pytest tests/ -q      (or: python tests/test_pipeline.py)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import analyze, assign, lang, roles


# A tiny synthetic lesson: SPEAKER_00 teaches, 01 and 02 answer.
# Shaped as real classroom talk - long teacher turns, short student ones,
# questions followed by answers.
def build_lesson():
    segments, turns = [], []
    t = 0.0

    def add(speaker, text, dur):
        nonlocal t
        segments.append({"start": round(t, 2), "end": round(t + dur, 2),
                         "text": text, "speaker": None})
        turns.append({"start": round(t, 2), "end": round(t + dur, 2), "speaker": speaker})
        t += dur + 0.4

    add("SPEAKER_00", "Good morning class, open your book to page ten.", 30)
    add("SPEAKER_00", "Today we look at photosynthesis. What do plants need?", 25)
    add("SPEAKER_01", "Sunlight.", 3)
    add("SPEAKER_00", "Very good. And what else do they need?", 10)
    add("SPEAKER_02", "Water.", 2)
    add("SPEAKER_00", "Correct. Now write down the equation, everyone.", 40)
    add("SPEAKER_00", "Let me explain the chloroplast in detail for a while.", 75)
    add("SPEAKER_01", "Teacher, is it the same at night?", 5)
    add("SPEAKER_00", "Good question. No - listen carefully, at night it stops.", 20)
    return segments, turns


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + extra) if extra else ''}")
    return cond


def main():
    ok = True
    segments, turns = build_lesson()
    duration = turns[-1]["end"] + 5

    print("\nassign")
    seg = assign.assign_speakers(segments, turns)
    ok &= check("every segment gets a speaker",
                all(s["speaker"] and s["speaker"] != "UNKNOWN" for s in seg))
    ok &= check("speakers match the turns they came from",
                [s["speaker"] for s in seg] == [t["speaker"] for t in turns])
    ok &= check("clean segments are high confidence",
                all(s["speaker_conf"] > 0.95 for s in seg))

    utts = assign.to_utterances(seg)
    ok &= check("consecutive same-speaker segments merge",
                len(utts) < len(seg), f"{len(seg)} segments -> {len(utts)} utterances")

    stats = assign.speaker_stats(utts, duration)
    # shares are rounded to 4dp for the JSON, so they sum to 1 only to that
    ok &= check("talk shares sum to 1",
                abs(sum(s["talk_share"] for s in stats.values()) - 1.0) < 1e-3)
    ok &= check("teacher holds the most floor time",
                max(stats, key=lambda s: stats[s]["talk_time"]) == "SPEAKER_00")

    print("\nlanguage signals")
    ok &= check("English question mark", lang.is_question("What do plants need?", "en"))
    ok &= check("Malayalam question word", lang.is_question("ഇത് എന്താണ്", "ml"))
    ok &= check("statement is not a question", not lang.is_question("Water.", "en"))
    ok &= check("teacher cue found", lang.teacher_cue_score("open your book", "en") > 0)
    ok &= check("praise found", lang.praise_score("very good", "en") > 0)

    print("\nroles")
    verdict = roles.identify_teacher(utts, stats, "en", duration)
    ok &= check("picks SPEAKER_00 as teacher", verdict["teacher"] == "SPEAKER_00",
                f"score {verdict['scores']['SPEAKER_00']['score']}")
    ok &= check("confidence is not low", verdict["confidence"] in ("high", "medium"),
                f"{verdict['confidence']}, margin {verdict['margin']}")
    ok &= check("gives reasons", len(verdict["reasons"]) >= 2)

    labelled, names = roles.label_roles(utts, verdict["teacher"])
    ok &= check("teacher labelled Teacher", names["SPEAKER_00"] == "Teacher")
    ok &= check("students get letters",
                sorted(v for k, v in names.items() if k != "SPEAKER_00") == ["Student A", "Student B"])
    teacher_utts, student_utts = roles.split_by_role(labelled)
    ok &= check("split covers everything", len(teacher_utts) + len(student_utts) == len(labelled))

    print("\nmetrics")
    m = analyze.metrics(labelled, {"duration": duration}, "en")
    ok &= check("teacher talk ratio is dominant", m["teacher_talk_ratio"] > 0.9,
                f"{m['teacher_talk_ratio']:.2%}")
    ok &= check("talk ratios sum to 1",
                abs(m["teacher_talk_ratio"] + m["student_talk_ratio"] - 1.0) < 1e-6)
    ok &= check("counts the teacher's questions", m["teacher_questions"] >= 2,
                str(m["teacher_questions"]))
    ok &= check("students answered them", m["questions_answered"] >= 2,
                str(m["questions_answered"]))
    ok &= check("wait time measured", m["median_wait_time"] is not None,
                f"{m['median_wait_time']}s")
    ok &= check("finds the long monologue", m["long_monologues"] == 1)
    ok &= check("finds IRF triads", m["irf_triads"] >= 2, str(m["irf_triads"]))
    ok &= check("counts both students", m["n_students_heard"] == 2)
    ok &= check("praise counted", m["praise_moves"] >= 2, str(m["praise_moves"]))
    ok &= check("participation shares sum to 1",
                abs(sum(p["share"] for p in m["student_participation"].values()) - 1.0) < 1e-6)

    print("\nedge cases")
    solo, solo_turns = [segments[0]], [turns[0]]
    s2 = assign.assign_speakers(solo, solo_turns)
    u2 = assign.to_utterances(s2)
    st2 = assign.speaker_stats(u2, 40)
    v2 = roles.identify_teacher(u2, st2, "en", 40)
    ok &= check("single speaker still resolves", v2["teacher"] == "SPEAKER_00")
    l2, _ = roles.label_roles(u2, v2["teacher"])
    m2 = analyze.metrics(l2, {"duration": 40}, "en")
    ok &= check("no students -> no divide by zero", m2["student_talk_ratio"] == 0.0)
    ok &= check("no students heard", m2["n_students_heard"] == 0)

    orphan = [{"start": 500.0, "end": 505.0, "text": "stray", "speaker": None}]
    o = assign.assign_speakers(orphan, turns)
    ok &= check("segment far from any turn is UNKNOWN", o[0]["speaker"] == "UNKNOWN")

    # A Whisper segment that runs across a speaker change: 6s of one voice,
    # 4s of the next. Majority wins, but it must be marked as unreliable.
    two_turns = [{"start": 0.0, "end": 6.0, "speaker": "SPEAKER_00"},
                 {"start": 6.0, "end": 12.0, "speaker": "SPEAKER_01"}]
    contested = [{"start": 0.0, "end": 10.0, "text": "spans two speakers", "speaker": None}]
    c = assign.assign_speakers(contested, two_turns)
    ok &= check("straddling segment takes the majority speaker",
                c[0]["speaker"] == "SPEAKER_00")
    ok &= check("straddling segment is flagged contested", c[0]["contested"] is True,
                f"conf {c[0]['speaker_conf']}")

    print("\ntranscript")
    text = analyze.build_transcript(labelled)
    ok &= check("transcript has one line per utterance",
                len(text.splitlines()) == len(labelled))
    ok &= check("transcript carries labels and timestamps",
                "[00:00] Teacher:" in text)
    try:
        analyze.build_transcript(labelled, max_chars=10)
        ok &= check("over-long transcript raises rather than truncates", False)
    except ValueError:
        ok &= check("over-long transcript raises rather than truncates", True)

    print("\n" + ("ALL PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
