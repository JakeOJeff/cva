"""
CLI for the same pipeline the web app runs.

Useful when you want to watch a single lesson go through, or re-score one
without re-transcribing it.
"""

import argparse
import os
import sys

from dotenv import load_dotenv

from pipeline import analyze, run


def fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def progress(stage: str, fraction: float, note: str) -> None:
    bar = "#" * int(fraction * 24)
    sys.stderr.write(f"\r  {stage:<11} [{bar:<24}] {fraction:5.0%}  {note[:44]:<44}")
    if fraction >= 1.0:
        sys.stderr.write("\n")
    sys.stderr.flush()


def print_report(result: dict) -> None:
    m, v = result["metrics"], result["teacher"]
    print("\n" + "=" * 66)
    print(f"  {result['meta']['source']}   {fmt(m['duration'])}   "
          f"{result['meta']['n_speakers']} speakers")
    print("=" * 66)

    print("\nSPEAKERS")
    for sp, s in result["speakers"].items():
        mark = "*" if sp == v["teacher"] else " "
        print(f" {mark} {s['label']:<11} {sp:<12} {s['talk_share']:>6.1%}  "
              f"{s['n_turns']:>4} turns  avg {s['avg_turn']:>5.1f}s")
    print(f"\n  teacher = {v['teacher']} ({v['confidence']} confidence)")
    for r in v.get("reasons", []):
        print(f"    - {r}")

    print("\nMETRICS")
    wait = "n/a" if m["median_wait_time"] is None else f"{m['median_wait_time']:.1f}s"
    rows = [
        ("teacher talk", f"{m['teacher_talk_ratio']:.0%} of speech"),
        ("student talk", f"{m['student_talk_ratio']:.0%} of speech"),
        ("silence", f"{m['silence_ratio']:.0%} of lesson"),
        ("teacher questions", f"{m['teacher_questions']} ({m['questions_per_10min']}/10min)"),
        ("answered", f"{m['questions_answered']} ({m['question_response_rate']:.0%})"),
        ("median wait time", wait),
        ("longest monologue", f"{m['teacher_longest_turn']:.0f}s "
                              f"({m['long_monologues']} over 60s)"),
        ("IRF triads", str(m["irf_triads"])),
        ("students heard", str(m["n_students_heard"])),
    ]
    for k, val in rows:
        print(f"  {k:<20} {val}")

    review = result.get("review")
    if review and not review.get("error"):
        print("\nREVIEW")
        print(f"  {review['summary']}")
        if review.get("strengths"):
            print("\n  worked well:")
            for s in review["strengths"]:
                print(f"    + {s}")
        if review.get("areas_for_improvement"):
            print("\n  to try next:")
            for a in review["areas_for_improvement"]:
                print(f"    - {a['issue']}")
                print(f"      {a['suggestion']}")
    elif review:
        print(f"\nREVIEW  skipped: {review['error']}")


def main() -> None:
    load_dotenv()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio", help="path to the recording")
    p.add_argument("--lang", default="ml", help="ml, hi, ta, en ...")
    p.add_argument("--model", default="small", help="tiny, base, small, medium")
    p.add_argument("--speakers", type=int, help="exact speaker count, if known")
    p.add_argument("--max-speakers", type=int, help="upper bound on speakers")
    p.add_argument("--out", default="out", help="directory for the results")
    p.add_argument("--no-llm", action="store_true", help="metrics only, no API call")
    p.add_argument("--teacher", help="re-score an existing --out with this speaker as teacher")
    args = p.parse_args()

    if args.teacher:
        result = run.reanalyze(args.out, language=args.lang,
                               use_llm=not args.no_llm, teacher=args.teacher)
    else:
        result = run.run(
            args.audio, work_dir=args.out, language=args.lang,
            model_size=args.model, num_speakers=args.speakers,
            max_speakers=args.max_speakers, use_llm=not args.no_llm,
            on_progress=progress,
        )

    print_report(result)

    with open(os.path.join(args.out, "transcript.txt"), "w", encoding="utf-8") as f:
        f.write(analyze.build_transcript(result["utterances"]))

    print(f"\nsaved -> {os.path.join(args.out, 'result.json')}")
    print(f"         {os.path.join(args.out, 'transcript.txt')}")


if __name__ == "__main__":
    main()
