"""
The web app.

Thin on purpose: it accepts a file, hands it to the job queue, and reports
back. Every decision about the audio lives in `pipeline/` and can be tested
without a server.
"""

import asyncio
import base64
import json
import os
import secrets
import shutil

from dotenv import load_dotenv
from fastapi import (FastAPI, File, Form, HTTPException, Request,
                     UploadFile)
from fastapi.responses import (FileResponse, JSONResponse, Response,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles

from . import jobs

load_dotenv()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
MAX_UPLOAD_MB = int(os.environ.get("CVA_MAX_UPLOAD_MB", "500"))
ALLOWED_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".flac",
               ".wma", ".mp4", ".mkv", ".webm", ".mov"}

app = FastAPI(title="Classroom Voice Analysis")

# Password gate. Off when CVA_PASSWORD is unset, which is right for
# localhost and wrong for anything else: this app stores classroom
# recordings of children, and without a gate a public URL lets anyone
# upload audio and download every transcript.
PASSWORD = os.environ.get("CVA_PASSWORD", "").strip()
USERNAME = os.environ.get("CVA_USERNAME", "teacher").strip()


def _authenticated(request) -> bool:
    if not PASSWORD:
        return True
    header = request.headers.get("authorization", "")
    if not header.startswith("Basic "):
        return False
    try:
        raw = base64.b64decode(header[6:]).decode("utf-8", "replace")
        user, _, pwd = raw.partition(":")
        # compare_digest on both halves: no early exit on the first differing
        # byte, so timing does not leak the password.
        return (secrets.compare_digest(user, USERNAME)
                and secrets.compare_digest(pwd, PASSWORD))
    except Exception:                                         # noqa: BLE001
        return False


# The container healthcheck runs as an anonymous local curl. Behind the auth
# gate it got a 401, so the container was reported unhealthy and orchestrators
# would restart it forever. Health is exempt; it returns nothing sensitive
# unless the caller is actually authenticated.
PUBLIC_PATHS = {"/api/health"}


@app.middleware("http")
async def _auth(request, call_next):
    if request.url.path in PUBLIC_PATHS or _authenticated(request):
        return await call_next(request)
    return Response(status_code=401, content="Authentication required",
                    headers={"WWW-Authenticate": 'Basic realm="Classroom Voice Analysis"'})


@app.on_event("startup")
def _startup() -> None:
    jobs.init()
    jobs.reap_interrupted()          # startup only - never from a request
    jobs.ensure_worker()
    if not PASSWORD:
        print("\n  [!] CVA_PASSWORD is not set - this server has NO authentication.")
        print("      Fine on localhost. Do not expose it to the internet like this:")
        print("      anyone with the URL could upload audio and read every")
        print("      transcript of your classroom.\n")


# ------------------------------------------------------------------ pages

@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/jobs/{job_id}")
def job_page(job_id: str):
    return FileResponse(os.path.join(STATIC_DIR, "job.html"))


# -------------------------------------------------------------------- api

@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    language: str = Form("hi"),
    model_size: str = Form("tiny"),
    num_speakers: int | None = Form(None),
    max_speakers: int | None = Form(6),
    use_llm: bool = Form(False),
    stream: bool = Form(True),
    chunk_seconds: int = Form(300),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"unsupported file type '{ext}'. "
                                 f"Accepted: {', '.join(sorted(ALLOWED_EXT))}")

    jobs.init()
    stem = os.path.basename(file.filename or "audio").replace(os.sep, "_")
    dest = os.path.join(jobs.UPLOAD_DIR, f"{os.urandom(4).hex()}_{stem}")

    # Stream to disk - a lesson recording is hundreds of megabytes and does
    # not belong in memory.
    size = 0
    limit = MAX_UPLOAD_MB * 1024 * 1024
    with open(dest, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                out.close()
                os.remove(dest)
                raise HTTPException(413, f"file exceeds {MAX_UPLOAD_MB} MB")
            out.write(chunk)

    if size == 0:
        os.remove(dest)
        raise HTTPException(400, "empty file")

    job_id = jobs.create(file.filename or stem, dest, {
        "language": language,
        "model_size": model_size,
        "num_speakers": num_speakers,
        "max_speakers": max_speakers,
        "use_llm": use_llm,
        "stream": stream,
        "chunk_seconds": chunk_seconds,
    })
    return {"job_id": job_id, "size_bytes": size}


@app.get("/api/jobs")
def list_jobs(limit: int = 50):
    return {"jobs": jobs.listing(limit)}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    job.pop("audio_path", None)
    job.pop("work_dir", None)
    return job


@app.get("/api/jobs/{job_id}/stream")
async def stream_events(job_id: str):
    """
    Server-Sent Events: each chunk's dialogue as soon as that chunk is done.

    Events are replayed from the start of the file, so a browser that opens
    this halfway through still gets everything - no state is held in the
    connection, and a reconnect is not a lost result.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    path = os.path.join(job["work_dir"], "events.jsonl")

    async def generate():
        sent = 0
        idle = 0.0
        while True:
            lines = []
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    lines = f.read().splitlines()

            for line in lines[sent:]:
                if line.strip():
                    yield f"data: {line}\n\n"
            if len(lines) > sent:
                idle = 0.0
                sent = len(lines)

            if lines and '"type": "done"' in lines[-1]:
                return

            current = jobs.get(job_id)
            if not current or current["status"] in ("done", "failed", "cancelled"):
                # The run ended without a done event - say why rather than
                # leaving the browser on an open socket forever.
                payload = json.dumps({"type": "ended",
                                      "status": current["status"] if current else "gone",
                                      "error": (current or {}).get("error")})
                yield f"data: {payload}\n\n"
                return

            await asyncio.sleep(0.5)
            idle += 0.5
            if idle >= 15.0:               # keep proxies from closing an idle stream
                yield ": keepalive\n\n"
                idle = 0.0

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    res = jobs.result(job_id)
    if res is None:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "no such job")
        raise HTTPException(409, f"job is {job['status']}, no result yet")
    return JSONResponse(res)


@app.post("/api/jobs/{job_id}/teacher")
def override_teacher(job_id: str, speaker: str = Form(...), use_llm: bool = Form(False)):
    """
    Correct a wrong teacher call and re-derive everything downstream.

    No audio is touched, so this is seconds rather than minutes - which is
    the point of keeping the stages separate.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    if job["status"] != "done":
        raise HTTPException(409, f"job is {job['status']}")

    from pipeline import run as pipeline_run
    try:
        return JSONResponse(pipeline_run.reanalyze(
            job["work_dir"],
            language=job["options"].get("language", "ml"),
            use_llm=use_llm,
            teacher=speaker,
        ))
    except Exception as exc:                                  # noqa: BLE001
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc


@app.get("/api/jobs/{job_id}/transcript")
def transcript(job_id: str, role: str | None = None):
    """Plain-text transcript. `role=teacher` gives only the teacher's lines."""
    res = jobs.result(job_id)
    if res is None:
        raise HTTPException(404, "no result")
    utts = res["utterances"]
    if role:
        utts = [u for u in utts if u.get("role") == role]

    def stamp(t):
        return f"{int(t // 60):02d}:{int(t % 60):02d}"

    body = "\n".join(f"[{stamp(u['start'])}] {u['label']}: {u['text']}" for u in utts)
    return JSONResponse({"text": body, "n_utterances": len(utts)})


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    """
    Ask a queued or running job to stop.

    A running job stops at the next block boundary, not instantly - Whisper
    and pyannote are opaque calls that cannot be interrupted partway. Whatever
    it already finished stays on disk.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    if not jobs.cancel(job_id):
        raise HTTPException(409, f"job is already {job['status']}")
    return {"cancelling": job_id, "was": job["status"]}


@app.delete("/api/jobs/{job_id}")
def remove_job(job_id: str):
    """
    Remove a job, its uploaded audio and its results.

    A running job is asked to stop first; its files go when it lets go of
    them, so deleting one mid-run is safe.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    if job["status"] in ("queued", "running"):
        jobs.cancel(job_id)
    if not jobs.delete(job_id):
        raise HTTPException(404, "no such job")
    return {"deleted": job_id, "was": job["status"]}


@app.delete("/api/jobs")
def remove_finished(status: str = "failed"):
    """Bulk cleanup, e.g. every failed job. Never touches a live one."""
    if status not in ("failed", "done", "cancelled"):
        raise HTTPException(400, "status must be failed, done or cancelled")
    removed = [j["id"] for j in jobs.listing(500) if j["status"] == status]
    for job_id in removed:
        jobs.delete(job_id)
    return {"deleted": removed, "count": len(removed)}


@app.get("/api/health")
def health(request: Request):
    """
    Liveness, reachable without a password so the container healthcheck works.

    Which credentials are configured and how much disk is left is operational
    detail, not liveness - an anonymous caller gets neither.
    """
    if not _authenticated(request):
        return {"ok": True}
    return {
        "ok": True,
        "hf_token": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")),
        "anthropic_key": bool(os.environ.get("ANTHROPIC_API_KEY")
                              or os.environ.get("ANTHROPIC_AUTH_TOKEN")),
        "disk_free_gb": round(shutil.disk_usage(jobs.DATA_DIR).free / 1e9, 1),
        "device": __import__("pipeline.transcribe", fromlist=["device"]).device()[0],
        "auth": bool(PASSWORD),
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
