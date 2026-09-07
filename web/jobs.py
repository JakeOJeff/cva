"""
Job store and worker.

One worker thread, one job at a time, on purpose. Whisper and pyannote each
saturate the CPU; running two lessons concurrently makes both slower than
running them in sequence, and turns a predictable queue into a machine that
appears to have hung.

State lives in SQLite so a restart doesn't lose the queue, and so progress
survives the process that produced it.
"""

import json
import os
import queue
import sqlite3
import threading
import traceback
import uuid
from datetime import datetime, timezone

DATA_DIR = os.environ.get("CVA_DATA_DIR",
                          os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
DB_PATH = os.path.join(DATA_DIR, "jobs.db")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
RESULT_DIR = os.path.join(DATA_DIR, "results")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    audio_path   TEXT NOT NULL,
    work_dir     TEXT NOT NULL,
    options      TEXT NOT NULL,
    status       TEXT NOT NULL,   -- queued | running | done | failed
    stage        TEXT,
    progress     REAL DEFAULT 0,
    note         TEXT,
    error        TEXT,
    created_at   TEXT NOT NULL,
    started_at   TEXT,
    finished_at  TEXT
);
"""

_q: "queue.Queue[str]" = queue.Queue()
_worker: threading.Thread | None = None
_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    # The worker writes progress while requests read it; WAL is what makes
    # that not block.
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init() -> None:
    """
    Idempotent and SAFE to call at any time - it must never touch job rows.

    Reaping interrupted jobs lives in reap_interrupted() precisely because
    this one gets called from request handlers: doing the reap here meant
    every upload marked the currently-running job as failed.
    """
    for d in (DATA_DIR, UPLOAD_DIR, RESULT_DIR):
        os.makedirs(d, exist_ok=True)
    with _connect() as conn:
        conn.executescript(SCHEMA)


def reap_interrupted() -> None:
    """
    Startup only. Anything left mid-flight by a crash is not coming back on
    its own, so mark it rather than leave a permanent "running".
    """
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET status='failed', error='interrupted by restart', "
            "finished_at=? WHERE status IN ('running','queued')", (_now(),))


def create(filename: str, audio_path: str, options: dict) -> str:
    job_id = uuid.uuid4().hex[:12]
    work_dir = os.path.join(RESULT_DIR, job_id)
    os.makedirs(work_dir, exist_ok=True)
    with _connect() as conn:
        conn.execute(
            "INSERT INTO jobs (id, filename, audio_path, work_dir, options, status, "
            "stage, created_at) VALUES (?,?,?,?,?,'queued','queued',?)",
            (job_id, filename, audio_path, work_dir, json.dumps(options), _now()))
    _q.put(job_id)
    ensure_worker()
    return job_id


def get(job_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["options"] = json.loads(d["options"])
    return d


def listing(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, filename, status, stage, progress, note, error, created_at, "
            "finished_at FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def result(job_id: str) -> dict | None:
    job = get(job_id)
    if not job:
        return None
    path = os.path.join(job["work_dir"], "result.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _update(job_id: str, **fields) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    with _connect() as conn:
        conn.execute(f"UPDATE jobs SET {cols} WHERE id=?",
                     (*fields.values(), job_id))


def delete(job_id: str) -> bool:
    job = get(job_id)
    if not job:
        return False
    with _connect() as conn:
        conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
    for path in (job["audio_path"],):
        try:
            os.remove(path)
        except OSError:
            pass
    import shutil
    shutil.rmtree(job["work_dir"], ignore_errors=True)
    return True


# ------------------------------------------------------------------ worker

# The stages, and roughly what share of the wall clock each one takes on CPU.
# Used only to turn per-stage progress into one honest overall bar.
STAGE_WEIGHTS = {
    "normalize": 0.03,
    "transcribe": 0.55,
    "diarize": 0.35,
    "assign": 0.01,
    "roles": 0.01,
    "analyze": 0.05,
}


def _overall(stage: str, fraction: float) -> float:
    done = 0.0
    for name, weight in STAGE_WEIGHTS.items():
        if name == stage:
            return round(done + weight * fraction, 4)
        done += weight
    return round(done, 4)


def _run_one(job_id: str) -> None:
    from pipeline import run as pipeline_run

    job = get(job_id)
    if not job or job["status"] != "queued":
        return

    _update(job_id, status="running", started_at=_now(), stage="normalize", progress=0.0)

    def on_progress(stage, fraction, note):
        _update(job_id, stage=stage, progress=_overall(stage, fraction), note=note)

    try:
        opts = job["options"]
        pipeline_run.run(
            job["audio_path"],
            work_dir=job["work_dir"],
            language=opts.get("language", "ml"),
            model_size=opts.get("model_size", "small"),
            num_speakers=opts.get("num_speakers"),
            min_speakers=opts.get("min_speakers"),
            max_speakers=opts.get("max_speakers"),
            use_llm=opts.get("use_llm", True),
            on_progress=on_progress,
        )
        _update(job_id, status="done", stage="done", progress=1.0,
                note="complete", finished_at=_now())
    except Exception as exc:                                  # noqa: BLE001
        traceback.print_exc()
        _update(job_id, status="failed", error=f"{type(exc).__name__}: {exc}",
                finished_at=_now())


def _loop() -> None:
    while True:
        job_id = _q.get()
        try:
            _run_one(job_id)
        finally:
            _q.task_done()


def ensure_worker() -> None:
    global _worker
    with _lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_loop, name="cva-worker", daemon=True)
            _worker.start()


def requeue_stale() -> None:
    """Re-enqueue jobs that init() marked failed only because of a restart."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id FROM jobs WHERE status='failed' AND error='interrupted by restart'"
        ).fetchall()
    for row in rows:
        _update(row["id"], status="queued", stage="queued", progress=0.0, error=None)
        _q.put(row["id"])
    if rows:
        ensure_worker()
