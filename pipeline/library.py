"""
The session library: `assets/in` goes in, `assets/out` comes out.

One folder of recordings to process, one folder of finished analyses. The
library is the second of those, and it is the only thing the home page reads.
A session is a directory holding a `result.json` - nothing else identifies
one, so a finished run can be moved, copied or committed as a plain folder
and it stays a session.

Kept separate from `web/` on purpose: the batch CLI, the server and the
publisher all need "what analyses exist on this machine", and there should be
one answer to that.
"""

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Overridable so a container can point at a mounted volume, and so the tests
# can work in a temporary directory.
INBOX_DIR = os.environ.get("CVA_INBOX_DIR", os.path.join(ROOT, "assets", "in"))
SESSION_DIR = os.environ.get("CVA_SESSIONS_DIR", os.path.join(ROOT, "assets", "out"))

AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".flac",
             ".wma", ".mp4", ".mkv", ".webm", ".mov"}

# Everything a viewer reads. `segments` and `turns` are pre-assignment
# intermediates - a third of a 64-minute document - which nothing renders but
# `run.reanalyze` needs, so they stay in the file on disk and are dropped on
# the way to a browser or to the published demo.
VIEW_KEYS = ("meta", "teacher", "speakers", "metrics", "utterances", "review")

LANGUAGE_NAMES = {
    "hi": "Hindi", "mr": "Marathi", "ml": "Malayalam", "ta": "Tamil",
    "te": "Telugu", "kn": "Kannada", "bn": "Bengali", "gu": "Gujarati",
    "pa": "Punjabi", "ur": "Urdu", "en": "English",
}


def slug(name: str) -> str:
    """
    A recording's filename -> its directory name under assets/out.

    Stable and reversible enough to read: "OD11163_2025-12-23.mp3" becomes
    "OD11163_2025-12-23". Anything that would need escaping in a URL or a
    path becomes a hyphen.
    """
    stem = os.path.splitext(os.path.basename(str(name)))[0]
    safe = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in stem)
    # Runs collapse: a browser download called "lesson (1).mp3" would otherwise
    # become "lesson--1", and the doubled hyphen shows up in the URL.
    safe = re.sub(r"-{2,}", "-", safe)
    return safe.strip("-.") or "session"


def is_safe_slug(value: str) -> bool:
    """
    Reject anything that could escape assets/out.

    The slug arrives from a URL path, so "..", a separator or a drive letter
    would otherwise read a result.json from anywhere on the disk.
    """
    return bool(value) and value == slug(value) and value not in (".", "..")


def session_path(name: str) -> str:
    return os.path.join(SESSION_DIR, slug(name))


def result_path(name: str) -> str:
    return os.path.join(session_path(name), "result.json")


def hms(seconds: float) -> str:
    seconds = int(seconds or 0)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def load(name: str) -> dict | None:
    """The full result document for one session, or None if there isn't one."""
    path = result_path(name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def summarize(result: dict, name: str, mtime: float | None = None) -> dict:
    """The row the session list renders. Everything here comes from meta."""
    meta = result.get("meta", {})
    code = meta.get("language") or "?"
    m = result.get("metrics") or {}
    return {
        "slug": slug(name),
        "file": meta.get("source") or slug(name),
        "duration": meta.get("duration"),
        "duration_label": hms(meta.get("duration", 0)),
        "speakers": meta.get("n_speakers"),
        "language": LANGUAGE_NAMES.get(code, code),
        "language_code": code,
        "model": meta.get("model"),
        "has_review": bool(result.get("review") and not result["review"].get("error")),
        # Two numbers on the card, so the list is worth reading on its own
        # rather than being a directory of filenames.
        "teacher_talk_ratio": m.get("teacher_talk_ratio"),
        "teacher_questions": m.get("teacher_questions"),
        "processed_at": mtime,
    }


def is_session(directory: str) -> bool:
    return os.path.isfile(os.path.join(directory, "result.json"))


def sessions() -> list[dict]:
    """
    Every finished analysis under assets/out, newest first.

    A directory without a readable result.json is skipped rather than raised:
    a run that is still going has a work directory here too, and a half-written
    library should still list the sessions that are fine.
    """
    if not os.path.isdir(SESSION_DIR):
        return []

    rows = []
    for entry in sorted(os.listdir(SESSION_DIR)):
        directory = os.path.join(SESSION_DIR, entry)
        if not is_session(directory):
            continue
        result = load(entry)
        if not result or "metrics" not in result:
            continue
        path = os.path.join(directory, "result.json")
        rows.append(summarize(result, entry, os.path.getmtime(path)))

    rows.sort(key=lambda r: r["processed_at"] or 0, reverse=True)
    return rows


def inbox() -> list[dict]:
    """
    Recordings sitting in assets/in, and whether each has been processed.

    The batch CLI uses this to decide what to work on; the home page uses it
    to say "three recordings are waiting" rather than looking empty for no
    visible reason.
    """
    if not os.path.isdir(INBOX_DIR):
        return []

    found = []
    for entry in sorted(os.listdir(INBOX_DIR)):
        path = os.path.join(INBOX_DIR, entry)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(entry)[1].lower() not in AUDIO_EXT:
            continue
        found.append({
            "file": entry,
            "path": path,
            "slug": slug(entry),
            "size_mb": round(os.path.getsize(path) / 1048576, 1),
            "done": is_session(session_path(entry)),
        })
    return found


def slim(result: dict) -> dict:
    """The result document with the intermediates dropped. Nothing is rounded
    or renamed, so a reviewer can diff it against the file on disk."""
    return {k: result[k] for k in VIEW_KEYS if k in result}
