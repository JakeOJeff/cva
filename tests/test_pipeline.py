"""
Tests for every stage that is a pure function over segments.

Deliberately no audio and no models here: those stages take minutes and
need a GPU or a token to be interesting. Everything that decides *meaning* -
who the teacher is, what the numbers say - is pure, and this is where the
bugs would actually hurt.

    python -m pytest tests/ -q      (or: python tests/test_pipeline.py)
"""

import json
import os
import sys
import tempfile

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
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' - ' + extra) if extra else ''}")
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

    print("\nunattributed speech is not a student")
    # A segment overlapping no diarization turn gets UNKNOWN. It must never be
    # handed a student letter - that invents a child who was never there.
    mixed = labelled + [{"start": 400.0, "end": 410.0, "text": "stray audio",
                         "speaker": "UNKNOWN", "speaker_conf": 0.0}]
    lab2, names2 = roles.label_roles(mixed, "SPEAKER_00")
    ok &= check("UNKNOWN is labelled Unattributed", names2["UNKNOWN"] == "Unattributed")
    ok &= check("UNKNOWN gets no student letter",
                not any(v.startswith("Student") and k == "UNKNOWN" for k, v in names2.items()))
    ok &= check("student letters unchanged by its presence",
                sorted(v for k, v in names2.items() if v.startswith("Student"))
                == ["Student A", "Student B"])
    t3, s3 = roles.split_by_role(lab2)
    ok &= check("UNKNOWN is in neither teacher nor student",
                all(i["speaker"] != "UNKNOWN" for i in t3 + s3))
    ok &= check("UNKNOWN is reachable as unattributed",
                len(roles.unattributed(lab2)) == 1)
    m3 = analyze.metrics(lab2, {"duration": duration + 60}, "en")
    ok &= check("students still counted as 2", m3["n_students_heard"] == 2,
                str(m3["n_students_heard"]))
    ok &= check("unattributed time reported", m3["unattributed_talk_time"] == 10.0,
                str(m3["unattributed_talk_time"]))
    ok &= check("ratios still sum to 1 with unattributed in the mix",
                abs(m3["teacher_talk_ratio"] + m3["student_talk_ratio"]
                    + m3["unattributed_ratio"] - 1.0) < 1e-3)
    ok &= check("Unattributed absent from participation",
                "Unattributed" not in m3["student_participation"])

    print("\njob dispatch")
    # web/jobs.py builds one kwargs dict and sends it down one of two paths.
    # Nothing else here executes those calls - they need audio and models - so
    # the signatures are checked directly, because a keyword that exists on
    # one path and not the other is a TypeError on every job of that kind.
    import inspect

    from pipeline import run as pipeline_run
    from pipeline import stream as pipeline_stream

    common = dict(work_dir="w", language="hi", model_size="tiny",
                  num_speakers=None, max_speakers=6, use_llm=False,
                  on_progress=lambda *a: None)
    try:
        inspect.signature(pipeline_stream.run_streaming).bind(
            "a.wav", chunk_seconds=300, **common)
        ok &= check("jobs.py kwargs bind to run_streaming", True)
    except TypeError as e:
        ok &= check("jobs.py kwargs bind to run_streaming", False, str(e))
    try:
        inspect.signature(pipeline_run.run).bind(
            "a.wav", min_speakers=None, **common)
        ok &= check("jobs.py kwargs bind to run", True)
    except TypeError as e:
        ok &= check("jobs.py kwargs bind to run", False, str(e))

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

    # ---------------------------------------------------------- library
    #
    # assets/in -> assets/out. The slug is the only part with teeth: it names
    # a directory and it arrives from a URL, so a bad one reads files outside
    # the library. The rest is directory scanning, checked against a real one
    # built in a temp dir.
    print("\nlibrary")
    import importlib

    from pipeline import library as library_mod
    ok &= check("slug drops the extension",
                library_mod.slug("OD11163_2025-12-23.mp3") == "OD11163_2025-12-23")
    # basename() eats the directories on both separators, which is what
    # keeps a path from becoming a nested slug.
    ok &= check("slug keeps only the filename",
                library_mod.slug("a/b\\c d.mp3") == "c-d")
    ok &= check("slug hyphenates spaces",
                library_mod.slug("Grade 4 maths.mp3") == "Grade-4-maths")
    ok &= check("slug never returns empty", library_mod.slug("...") == "session")
    for bad in ("..", ".", "", "a/b", "..\\x", "C:", "a b"):
        ok &= check(f"slug {bad!r} is rejected", not library_mod.is_safe_slug(bad))
    ok &= check("an ordinary slug is accepted",
                library_mod.is_safe_slug("OD11163_2025-12-23"))

    with tempfile.TemporaryDirectory() as root:
        inbox = os.path.join(root, "in")
        outbox = os.path.join(root, "out")
        os.makedirs(os.path.join(outbox, "lesson-one"))
        os.makedirs(os.path.join(outbox, "half-done"))     # no result.json
        os.makedirs(inbox)
        for name in ("lesson-one.mp3", "lesson-two.mp3", "notes.txt"):
            with open(os.path.join(inbox, name), "wb") as f:
                f.write(b"x" * 1024)

        finished = {"meta": {"source": "lesson-one.mp3", "duration": 3600.0,
                             "n_speakers": 4, "language": "mr", "model": "small"},
                    "metrics": {"teacher_talk_ratio": 0.71, "teacher_questions": 24},
                    "utterances": [], "turns": [1], "segments": [2]}
        with open(os.path.join(outbox, "lesson-one", "result.json"), "w",
                  encoding="utf-8") as f:
            json.dump(finished, f)

        os.environ["CVA_INBOX_DIR"], os.environ["CVA_SESSIONS_DIR"] = inbox, outbox
        try:
            lib = importlib.reload(library_mod)
            rows = lib.sessions()
            ok &= check("only directories with a result.json are sessions",
                        len(rows) == 1, f"got {len(rows)}")
            ok &= check("the session is named by its recording, not its folder",
                        rows[0]["file"] == "lesson-one.mp3")
            ok &= check("duration is rendered for the list",
                        rows[0]["duration_label"] == "1h 00m", rows[0]["duration_label"])
            ok &= check("the language code becomes a name",
                        rows[0]["language"] == "Marathi")

            waiting = lib.inbox()
            ok &= check("non-audio in the inbox is ignored",
                        [e["file"] for e in waiting] == ["lesson-one.mp3", "lesson-two.mp3"],
                        str([e["file"] for e in waiting]))
            ok &= check("a recording already analysed is marked done",
                        [e["done"] for e in waiting] == [True, False])
            ok &= check("slim drops the intermediates the viewer never reads",
                        set(lib.slim(finished)) == {"meta", "metrics", "utterances"})
        finally:
            os.environ.pop("CVA_INBOX_DIR")
            os.environ.pop("CVA_SESSIONS_DIR")
            importlib.reload(library_mod)

    print("\n" + ("ALL PASSED" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
