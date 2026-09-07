"""
Finished analyses -> the static demo under site/.

The demo interface is a static page. Nothing is transcribed at demo time:
the pipeline runs on a machine that has the audio, and only the derived JSON
is published. This is the step in between.

    python -m pipeline.publish                # every finished job in data/
    python -m pipeline.publish out            # one --out directory
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

SITE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "site")
OUT_DIR = os.path.join(SITE_DIR, "data")

DATA_DIR = os.environ.get("CVA_DATA_DIR",
                          os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

# Everything the viewer reads. Anything not listed here is an intermediate.
KEEP = ("meta", "teacher", "speakers", "metrics", "utterances", "review")

LANGUAGE_NAMES = {
    "hi": "Hindi", "mr": "Marathi", "ml": "Malayalam", "ta": "Tamil",
    "te": "Telugu", "kn": "Kannada", "bn": "Bengali", "gu": "Gujarati",
    "pa": "Punjabi", "ur": "Urdu", "en": "English",
}


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


def _slug(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    safe = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in stem)
    return safe.strip("-.") or "session"


def _hms(seconds: float) -> str:
    seconds = int(seconds or 0)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def _find(targets: list[str]) -> list[str]:
    """Resolve CLI arguments to result.json paths."""
    if not targets:
        results = os.path.join(DATA_DIR, "results")
        if not os.path.isdir(results):
            return []
        return sorted(
            p for d in os.listdir(results)
            if os.path.isfile(p := os.path.join(results, d, "result.json"))
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

        lang = meta.get("language") or "?"
        sessions.append({
            "file": filename,
            "json": json_name,
            "duration_label": _hms(meta.get("duration", 0)),
            "duration": meta.get("duration"),
            "speakers": meta.get("n_speakers"),
            "language": LANGUAGE_NAMES.get(lang, lang),
            "language_code": lang,
            "model": meta.get("model"),
            "backend": meta.get("backend", "local"),
            "has_review": bool(result.get("review")
                               and not result["review"].get("error")),
        })

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
                        "With none, every finished job under data/results/.")
    args = p.parse_args()

    paths = _find(args.targets)
    if not paths:
        print("Nothing to publish. Process a recording first — either through "
              "the local server, or:\n"
              "  python main.py <audio> --lang mr --model small --out out\n"
              "then point this at it:\n"
              "  python -m pipeline.publish out")
        sys.exit(1)

    print(f"publishing {len(paths)} result(s) to site/data/")
    sessions = publish(paths)
    print(f"\n{len(sessions)} session(s) in the manifest. Preview with:\n"
          f"  python -m http.server -d site 8080")


if __name__ == "__main__":
    main()
