"""
The web app.

Thin on purpose: it accepts a file, hands it to the job queue, and reports
back. Every decision about the audio lives in `pipeline/` and can be tested
without a server.
"""

import os
import shutil

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import jobs

load_dotenv()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
MAX_UPLOAD_MB = int(os.environ.get("CVA_MAX_UPLOAD_MB", "500"))
ALLOWED_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".flac",
               ".wma", ".mp4", ".mkv", ".webm", ".mov"}

app = FastAPI(title="Classroom Voice Analysis")


@app.on_event("startup")
def _startup() -> None:
    jobs.init()
    jobs.ensure_worker()


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
    language: str = Form("ml"),
    model_size: str = Form("small"),
    num_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
    use_llm: bool = Form(True),
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


@app.delete("/api/jobs/{job_id}")
def remove_job(job_id: str):
    if not jobs.delete(job_id):
        raise HTTPException(404, "no such job")
    return {"deleted": job_id}


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "hf_token": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")),
        "anthropic_key": bool(os.environ.get("ANTHROPIC_API_KEY")
                              or os.environ.get("ANTHROPIC_AUTH_TOKEN")),
        "disk_free_gb": round(shutil.disk_usage(jobs.DATA_DIR).free / 1e9, 1),
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
