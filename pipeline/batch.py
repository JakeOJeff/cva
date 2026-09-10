"""
Process everything in assets/in, write the analyses to assets/out.

    python -m pipeline.batch                     # everything not done yet
    python -m pipeline.batch --lang hi           # override the language
    python -m pipeline.batch --only OD11163      # one recording
    python -m pipeline.batch --force             # re-run ones already done
    python -m pipeline.batch --list              # show what would run

Each recording gets its own directory under assets/out, named after the file:

    assets/in/OD11163_2025-12-23.mp3
    assets/out/OD11163_2025-12-23/result.json
                                 /transcript.txt

That directory *is* the session. The home page lists them, so a recording
appears there the moment its run finishes - there is no import step.

A recording that already has a result.json is skipped, which makes this safe
to re-run: add a file to assets/in, run it again, and only the new one costs
you an hour. One failure does not stop the rest; the exit code is non-zero if
anything failed, so a script can tell.
"""

import argparse
import os
import sys
import time
import traceback

from dotenv import load_dotenv

from . import analyze, lang, library, run


def _progress():
    """A one-line bar on stderr, so piping stdout to a log stays readable."""
    def show(stage: str, fraction: float, note: str) -> None:
        bar = "#" * int(fraction * 20)
        sys.stderr.write(f"\r    {stage:<11} [{bar:<20}] {fraction:5.0%}  {note[:38]:<38}")
        if fraction >= 1.0:
            sys.stderr.write("\n")
        sys.stderr.flush()
    return show


def _headline(result: dict) -> str:
    m = result["metrics"]
    return (f"{result['meta']['n_speakers']} speakers | "
            f"teacher {m['teacher_talk_ratio']:.0%} of speech | "
            f"{m['teacher_questions']} questions | "
            f"{m['questions_answered']} answered")


def process(entry: dict, opts: argparse.Namespace) -> dict:
    """Run one recording. Returns the result document."""
    out_dir = library.session_path(entry["file"])
    os.makedirs(out_dir, exist_ok=True)

    result = run.run(
        entry["path"], work_dir=out_dir, language=opts.lang,
        model_size=opts.model, beam_size=opts.beam,
        num_speakers=opts.speakers, max_speakers=opts.max_speakers,
        use_llm=opts.llm, on_progress=_progress(),
    )

    with open(os.path.join(out_dir, "transcript.txt"), "w", encoding="utf-8") as f:
        f.write(analyze.build_transcript(result["utterances"]))

    return result


def main() -> None:
    load_dotenv()

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", default=lang.DEFAULT_LANGUAGE,
                   help="mr, hi, ml, ta, en (these have lexicons); te, kn and "
                        "the rest transcribe but score on English cues only. "
                        "Get this right - the wrong one does not error")
    p.add_argument("--model", default="small",
                   help="tiny, base, small (default), medium. Below `small` no "
                        "Indic language comes back in the right script")
    p.add_argument("--beam", type=int, default=5, help="beam width (default 5)")
    p.add_argument("--speakers", type=int, help="exact speaker count, if known")
    p.add_argument("--max-speakers", type=int, default=6,
                   help="upper bound on speakers (default 6)")
    p.add_argument("--llm", action="store_true",
                   help="also run the paid AI teaching review")
    p.add_argument("--only", action="append", metavar="NAME",
                   help="process only recordings whose filename contains NAME. "
                        "Repeatable")
    p.add_argument("--force", action="store_true",
                   help="re-run recordings that already have a result")
    p.add_argument("--list", action="store_true", dest="list_only",
                   help="show what would run, then stop")
    opts = p.parse_args()

    found = library.inbox()
    if not found:
        print(f"Nothing in {library.INBOX_DIR}")
        print("Drop lesson recordings in there (mp3, wav, m4a, ...) and run this again.")
        sys.exit(1)

    todo = found
    if opts.only:
        todo = [e for e in todo
                if any(n.lower() in e["file"].lower() for n in opts.only)]
    if not opts.force:
        todo = [e for e in todo if not e["done"]]

    print(f"assets/in   {len(found)} recording(s), {sum(e['done'] for e in found)} already done")
    for e in found:
        mark = "run " if e in todo else ("done" if e["done"] else "skip")
        print(f"  {mark}  {e['file']}  ({e['size_mb']} MB)")

    if opts.list_only:
        return
    if not todo:
        print("\nNothing to do. --force re-runs the ones already finished.")
        return

    print(f"\n{opts.model} | {opts.lang} | review {'on' if opts.llm else 'off'}"
          f"  ->  {library.SESSION_DIR}\n")

    failed = []
    for i, entry in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {entry['file']}")
        started = time.time()
        try:
            result = process(entry, opts)
        except KeyboardInterrupt:
            print("\n  stopped.")
            sys.exit(130)
        except Exception as exc:                                  # noqa: BLE001
            # One bad recording should not cost you the rest of the batch, but
            # the reason has to survive - a bare message is rarely enough to
            # tell a missing token from a corrupt file.
            failed.append((entry["file"], exc))
            print(f"    FAILED  {type(exc).__name__}: {exc}")
            traceback.print_exc(file=sys.stderr)
            continue
        mins = (time.time() - started) / 60
        print(f"    {_headline(result)}")
        print(f"    {mins:.0f} min  ->  assets/out/{entry['slug']}/\n")

    ok = len(todo) - len(failed)
    print(f"{ok} of {len(todo)} processed.")
    if failed:
        print("failed:")
        for name, exc in failed:
            print(f"  {name}: {type(exc).__name__}: {exc}")
    print("\nOpen them at http://localhost:8000 "
          "(.venv\\Scripts\\uvicorn web.app:app --port 8000)")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
