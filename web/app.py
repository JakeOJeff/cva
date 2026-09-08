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

from pipeline import lang, library

from . import jobs

load_dotenv()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
# The static demo. Its stylesheet and report renderer are the ones this server
# serves too, so the live results page and the published page cannot drift
# apart - there is one implementation of "render a result", not two.
SITE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "site")
MAX_UPLOAD_MB = int(os.environ.get("CVA_MAX_UPLOAD_MB", "500"))
# 0 = no limit. Set it on any shared or public instance.
MAX_AUDIO_MINUTES = int(os.environ.get("CVA_MAX_AUDIO_MINUTES", "0"))
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

# Two halves, deliberately separate. The home page is a library of finished
# analyses in assets/out - that is the thing anyone actually wants to look at.
# Processing is a workshop you visit to start a run, at /process.

@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "sessions.html"))


@app.get("/process")
def process_page():
    return FileResponse(os.path.join(STATIC_DIR, "process.html"))


@app.get("/sessions/{slug}")
def session_page(slug: str):
    return FileResponse(os.path.join(STATIC_DIR, "session.html"))


@app.get("/jobs/{job_id}")
def job_page(job_id: str):
    return FileResponse(os.path.join(STATIC_DIR, "job.html"))


# ------------------------------------------------------------- the library

@app.get("/api/sessions")
def api_sessions():
    """
    Everything in assets/out, plus what is still waiting in assets/in.

    The inbox is included so the home page can say "two recordings are
    waiting, run this command" instead of looking empty for no visible
    reason when the folder is full.
    """
    return {"sessions": library.sessions(),
            "inbox": [{k: v for k, v in e.items() if k != "path"}
                      for e in library.inbox()],
            "inbox_dir": library.INBOX_DIR,
            "sessions_dir": library.SESSION_DIR}


@app.get("/api/sessions/{slug}")
def api_session(slug: str, full: bool = False):
    # The slug is a URL path segment used to build a filesystem path, so it
    # is checked rather than trusted: ".." or a separator would otherwise
    # read a result.json from anywhere on the disk.
    if not library.is_safe_slug(slug):
        raise HTTPException(400, "bad session name")
    result = library.load(slug)
    if result is None:
        raise HTTPException(404, "no such session")
    # `?full=1` for the raw document including the intermediates - what you
    # want if you are downloading it to diff or to re-run something on it.
    return JSONResponse(result if full else library.slim(result))


@app.get("/api/sessions/{slug}/transcript")
def api_session_transcript(slug: str, role: str | None = None):
    if not library.is_safe_slug(slug):
        raise HTTPException(400, "bad session name")
    result = library.load(slug)
    if result is None:
        raise HTTPException(404, "no such session")

    utterances = result.get("utterances", [])
    if role in ("teacher", "student"):
        utterances = [u for u in utterances if u.get("role") == role]
    body = __import__("pipeline.analyze", fromlist=["build_transcript"])         .build_transcript(utterances)
    return Response(body, media_type="text/plain; charset=utf-8")


@app.post("/api/sessions/{slug}/teacher")
def api_session_teacher(slug: str, speaker: str = Form(...),
                        use_llm: bool = Form(False)):
    """
    Correct the teacher on a session in the library and re-derive from it.

    No audio is touched - the saved utterances are re-scored - so this is a
    second or two, and it is the reason the stages are kept separate.
    """
    if not library.is_safe_slug(slug):
        raise HTTPException(400, "bad session name")
    directory = library.session_path(slug)
    if not library.is_session(directory):
        raise HTTPException(404, "no such session")

    from pipeline import run as pipeline_run
    result = library.load(slug) or {}
    language = (result.get("meta") or {}).get("language", lang.DEFAULT_LANGUAGE)
    try:
        updated = pipeline_run.reanalyze(directory, language=language,
                                         use_llm=use_llm, teacher=speaker)
    except Exception as exc:                                      # noqa: BLE001
        raise HTTPException(400, str(exc))
    return JSONResponse(updated)


# -------------------------------------------------------------------- api

@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    language: str = Form(lang.DEFAULT_LANGUAGE),
    model_size: str = Form("small"),
    num_speakers: int | None = Form(None),
    max_speakers: int | None = Form(6),
    use_llm: bool = Form(False),
    stream: bool = Form(True),
    chunk_seconds: int = Form(300),
    backend: str | None = Form(None),
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

    # On a small public box, one 64-minute upload is an eight-hour job that
    # blocks the queue for everyone else. Read the length from the container
    # header - the file is on disk and this costs nothing - and refuse early
    # rather than accepting work that will never realistically finish.
    if MAX_AUDIO_MINUTES:
        from pipeline import audio as audio_mod
        seconds = audio_mod.probe_duration(dest)
        estimated = seconds is None
        if estimated:
            # A missing duration used to mean "let it through", which is how
            # an hour-long lesson reached an instance sized for five minutes.
            # VBR mp3s and phone recordings routinely omit it, so guess from
            # the byte count at 128kbps. That is a generous bitrate, so the
            # estimate runs short for quieter files and this still errs
            # toward accepting rather than refusing something that would
            # have been fine.
            seconds = size / 16000.0
        if seconds > MAX_AUDIO_MINUTES * 60:
            os.remove(dest)
            about = "about " if estimated else ""
            raise HTTPException(413,
                f"This recording is {about}{seconds / 60:.0f} minutes; this "
                f"instance accepts up to {MAX_AUDIO_MINUTES}. It runs on a "
                "small shared CPU, where an hour of audio takes many hours to "
                "process. Trim the clip, or run it locally where there is no "
                "limit.")

    job_id = jobs.create(file.filename or stem, dest, {
        "language": language,
        "model_size": model_size,
        "num_speakers": num_speakers,
        "max_speakers": max_speakers,
        "use_llm": use_llm,
        "stream": stream,
        "chunk_seconds": chunk_seconds,
        "backend": backend,
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


@app.post("/api/jobs/{job_id}/save")
def save_to_library(job_id: str):
    """
    Copy a finished job into assets/out, where the home page lists it.

    Uploading through the browser and running the batch CLI produce the same
    result document; only where it lands differs. Without this, a lesson
    processed here would sit in the job store and never appear in the library,
    which reads as a bug rather than as two separate places.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    if job["status"] != "done":
        raise HTTPException(400, f"job is {job['status']}, not done")

    source = os.path.join(job["work_dir"], "result.json")
    if not os.path.exists(source):
        raise HTTPException(404, "that job has no result.json")

    slug = library.slug(job["filename"])
    directory = library.session_path(slug)
    # Two uploads of the same filename would otherwise overwrite each other
    # silently. Keep both and let whoever is looking decide.
    if library.is_session(directory):
        n = 2
        while library.is_session(library.session_path(f"{slug}-{n}")):
            n += 1
        slug = f"{slug}-{n}"
        directory = library.session_path(slug)

    os.makedirs(directory, exist_ok=True)
    shutil.copyfile(source, os.path.join(directory, "result.json"))

    result = library.load(slug) or {}
    transcript = __import__("pipeline.analyze", fromlist=["build_transcript"])         .build_transcript(result.get("utterances", []))
    with open(os.path.join(directory, "transcript.txt"), "w", encoding="utf-8") as f:
        f.write(transcript)

    return {"slug": slug, "url": f"/sessions/{slug}"}


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
            language=job["options"].get("language", lang.DEFAULT_LANGUAGE),
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
        "backends": __import__("pipeline.backends", fromlist=["available"]).available(),
        "backend_default": __import__("pipeline.backends", fromlist=["DEFAULT"]).DEFAULT,
        "auth": bool(PASSWORD),
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/site", StaticFiles(directory=SITE_DIR), name="site")
