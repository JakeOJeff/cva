"""
Finished analyses -> the static demo under site/.

The demo interface is a static page. Nothing is transcribed at demo time:
the pipeline runs on a machine that has the audio, and only the derived JSON
is published. This is the step in between.

    python -m pipeline.publish                # every session in assets/out
    python -m pipeline.publish assets/out/OD11163   # one session
    python -m pipeline.publish a/result.json b/result.json

It writes site/data/<recording>.json plus site/data/manifest.json, which is
the list the picker renders.

Two things are deliberately dropped on the way out:

  turns, segments   Pre-assignment intermediates. `turns` alone is a third of
                    a 64-minute result document, and the viewer never reads
                    either - it renders `utterances`. They stay in the source
                    result.json, where `reanalyze` needs them.

Nothing is renamed or rounded. The published document is the same numbers the
pipeline produced, so a reviewer can diff it against a local run.
"""

import argparse
import json
import os
import sqlite3
import sys

from . import library

SITE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "site")
OUT_DIR = os.path.join(SITE_DIR, "data")

DATA_DIR = os.environ.get("CVA_DATA_DIR",
                          os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

# Everything the viewer reads. Anything not listed here is an intermediate.
KEEP = library.VIEW_KEYS
LANGUAGE_NAMES = library.LANGUAGE_NAMES


def _uploaded_names() -> dict[str, str]:
    """
    work_dir -> the filename the user actually uploaded.

    The web app saves uploads as "<random hex>_<name>" so two lessons called
    audio.mp3 cannot collide, and `meta.source` records that mangled name.
    The picker should show what the recording is really called, so the
    original is read back out of the job store when there is one. A --out
    directory from the CLI has no job store and falls back to meta.source,
    which is already the true name there.
    """
    db = os.path.join(DATA_DIR, "jobs.db")
    if not os.path.exists(db):
        return {}
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        rows = conn.execute("SELECT work_dir, filename FROM jobs "
                            "WHERE status = 'done'").fetchall()
        conn.close()
    except sqlite3.Error:
        return {}
    return {os.path.abspath(w): f for w, f in rows}


_slug = library.slug
_hms = library.hms


def _find(targets: list[str]) -> list[str]:
    """
    Resolve CLI arguments to result.json paths.

    With no arguments this is the session library - assets/out - and not the
    job store. The library is the curated set: things are there because
    somebody decided to keep them, by running the batch or by pressing "Save
    to sessions". Publishing every scratch job instead would put failed
    experiments in the demo.
    """
    if not targets:
        if not os.path.isdir(library.SESSION_DIR):
            return []
        return sorted(
            p for d in sorted(os.listdir(library.SESSION_DIR))
            if os.path.isfile(p := os.path.join(library.SESSION_DIR, d, "result.json"))
        )

    paths = []
    for t in targets:
        if os.path.isdir(t):
            candidate = os.path.join(t, "result.json")
            if not os.path.exists(candidate):
                raise SystemExit(f"no result.json in {t}")
            paths.append(candidate)
        elif os.path.exists(t):
            paths.append(t)
        else:
            raise SystemExit(f"no such file: {t}")
    return paths


def publish(paths: list[str], out_dir: str = OUT_DIR) -> list[dict]:
    """Write one JSON per result plus the manifest. Returns the manifest rows."""
    os.makedirs(out_dir, exist_ok=True)
    names = _uploaded_names()
    sessions, seen = [], set()

    for path in paths:
        with open(path, encoding="utf-8") as f:
            result = json.load(f)

        if "metrics" not in result or "utterances" not in result:
            print(f"  skip  {path}  (not a finished result)")
            continue

        meta = result.get("meta", {})
        filename = names.get(os.path.abspath(os.path.dirname(path))) \
            or meta.get("source") or os.path.basename(os.path.dirname(path))

        slug = _slug(filename)
        # Two runs of the same recording - a re-run at a bigger model, say -
        # would otherwise silently overwrite each other. Keep both, newest
        # suffix last, and let whoever publishes decide which to keep.
        if slug in seen:
            n = 2
            while f"{slug}-{n}" in seen:
                n += 1
            slug = f"{slug}-{n}"
        seen.add(slug)

        slim = {k: result[k] for k in KEEP if k in result}
        json_name = f"{slug}.json"
        with open(os.path.join(out_dir, json_name), "w", encoding="utf-8") as f:
            json.dump(slim, f, ensure_ascii=False, separators=(",", ":"))

        # The same row the server's session list renders, so the two views
        # cannot disagree about what a session is. Only the slug and the
        # filename are overridden: publish dedups slugs across the batch, and
        # prefers the name the file was uploaded under.
        row = library.summarize(result, filename)
        row.update({"slug": slug, "file": filename, "json": json_name})
        row.pop("processed_at", None)
        sessions.append(row)

        size = os.path.getsize(os.path.join(out_dir, json_name))
        print(f"  ok    {filename}  ->  site/data/{json_name}  ({size / 1024:.0f} KB)")

    sessions.sort(key=lambda s: s["file"])
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"sessions": sessions}, f, ensure_ascii=False, indent=2)

    return sessions


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("targets", nargs="*",
                   help="result.json files or the directories holding them. "
                        "With none, every session in assets/out.")
    args = p.parse_args()

    paths = _find(args.targets)
    if not paths:
        print(f"No sessions in {library.SESSION_DIR}\n\n"
              "Put recordings in assets/in and process them first:\n"
              "  python -m pipeline.batch\n\n"
              "Or point this at a single run directory:\n"
              "  python -m pipeline.publish out")
        sys.exit(1)

    print(f"publishing {len(paths)} result(s) to site/data/")
    sessions = publish(paths)
    print(f"\n{len(sessions)} session(s) in the manifest. Preview with:\n"
          f"  python -m http.server -d site 8080")


if __name__ == "__main__":
    main()
