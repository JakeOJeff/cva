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

from pipeline import analyze, assign, backends, lang, roles


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

    print("\nscribe adapter")
    # A response shaped exactly like Scribe's: word-level, with spacing and
    # audio_event items mixed in, speakers as "speaker_N".
    scribe_words = [
        {"type": "word", "text": "Good",  "start": 0.0, "end": 0.4, "speaker_id": "speaker_0"},
        {"type": "spacing", "text": " ", "start": 0.4, "end": 0.42},
        {"type": "word", "text": "morning", "start": 0.42, "end": 0.9, "speaker_id": "speaker_0"},
        {"type": "audio_event", "text": "(laughter)", "start": 1.0, "end": 1.5},
        {"type": "word", "text": "Sunlight", "start": 2.0, "end": 2.6, "speaker_id": "speaker_1"},
        {"type": "word", "text": "Correct", "start": 5.0, "end": 5.5, "speaker_id": "speaker_0"},
        {"type": "word", "text": "now",     "start": 5.6, "end": 5.9, "speaker_id": "speaker_0"},
        # a word with no timings at all - Scribe marks start/end optional
        {"type": "word", "text": "ghost", "speaker_id": "speaker_0"},
    ]
    segs, turns = backends.words_to_segments(scribe_words)
    ok &= check("words become segments", len(segs) == 3, str(len(segs)))
    ok &= check("consecutive same-speaker words merge",
                segs[0]["text"] == "Good morning", segs[0]["text"])
    ok &= check("speaker change starts a new segment",
                segs[1]["text"] == "Sunlight" and segs[1]["speaker"] != segs[0]["speaker"])
    ok &= check("a long pause splits a segment",
                segs[2]["text"] == "Correct now", segs[2]["text"])
    ok &= check("spacing is not a segment",
                all("  " not in s["text"] for s in segs))
    ok &= check("audio events are excluded",
                not any("laughter" in s["text"] for s in segs))
    ok &= check("words without timings are dropped",
                not any("ghost" in s["text"] for s in segs))
    ok &= check("speaker ids map to our convention",
                {s["speaker"] for s in segs} == {"SPEAKER_00", "SPEAKER_01"},
                str({s["speaker"] for s in segs}))
    ok &= check("the same voice keeps one id",
                segs[0]["speaker"] == segs[2]["speaker"])
    ok &= check("turns merge the split segments back into two",
                len(turns) == 3 and turns[0]["speaker"] == "SPEAKER_00", str(len(turns)))
    ok &= check("segments are word-attributed, so never contested",
                all(s["speaker_conf"] == 1.0 and not s["contested"] for s in segs))
    ok &= check("timings are ordered",
                all(s["start"] <= s["end"] for s in segs)
                and all(a["end"] <= b["start"] for a, b in zip(segs, segs[1:])))

    # An unnumbered / missing speaker id must not crash or collide.
    odd = backends.words_to_segments([
        {"type": "word", "text": "a", "start": 0.0, "end": 0.5, "speaker_id": "spk-XY"},
        {"type": "word", "text": "b", "start": 3.0, "end": 3.5},
    ])[0]
    ok &= check("unnumbered speaker id still maps", odd[0]["speaker"] == "SPEAKER_00")
    ok &= check("missing speaker id becomes UNKNOWN", odd[1]["speaker"] == "UNKNOWN")
    ok &= check("empty word list is handled",
                backends.words_to_segments([]) == ([], []))

    # The adapter's output must satisfy the same contract as the local path.
    u = assign.to_utterances(segs)
    st = assign.speaker_stats(u, 10.0)
    vd = roles.identify_teacher(u, st, "en", 10.0)
    lab, _ = roles.label_roles(u, vd["teacher"])
    mm = analyze.metrics(lab, {"duration": 10.0}, "en")
    ok &= check("scribe output flows through the rest of the pipeline",
                vd["teacher"] in ("SPEAKER_00", "SPEAKER_01")
                and 0.0 <= mm["teacher_talk_ratio"] <= 1.0)

    print("\nbackend selection")
    ok &= check("local is the default", backends.get("local").name == "local")
    ok &= check("scribe is selectable", backends.get("scribe").name == "scribe")
    ok &= check("local needs no network", backends.get("local").needs_network is False)
    ok &= check("scribe is marked as networked", backends.get("scribe").needs_network is True)
    try:
        backends.get("nonsense")
        ok &= check("an unknown backend is rejected", False)
    except ValueError:
        ok &= check("an unknown backend is rejected", True)
    ok &= check("availability reports local as always usable",
                backends.available()["local"] is True)

    print("\njob dispatch")
    # web/jobs.py builds one kwargs dict and sends it down one of two paths.
    # Nothing else here executes those calls - they need audio and models - so
    # the signatures are checked directly. A `backend` argument added to run()
    # and not to run_streaming() shipped a TypeError on every streaming job.
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
            "a.wav", backend="scribe", min_speakers=None, **common)
        ok &= check("jobs.py kwargs bind to run", True)
    except TypeError as e:
        ok &= check("jobs.py kwargs bind to run", False, str(e))
    ok &= check("streaming takes no backend - it is local by construction",
                "backend" not in inspect.signature(
                    pipeline_stream.run_streaming).parameters)

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
