import os
import shutil
import threading
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from processor import process_file

UPLOAD_DIR = "storage/uploads"
OUTPUT_DIR = "storage/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="AI Silence & Filler-Word Editor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before deploying for real
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store. Swap for Redis/DB if you need this to survive restarts
# or run multiple worker processes.
JOBS: dict[str, dict] = {}


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | processing | done | error
    error: str | None = None
    removed_seconds: float | None = None
    kept_seconds: float | None = None
    transcript: str | None = None


def _run_job(job_id: str, input_path: str):
    JOBS[job_id]["status"] = "processing"
    try:
        result = process_file(input_path, OUTPUT_DIR)
        JOBS[job_id].update({
            "status": "done",
            "output_path": result.output_path,
            "removed_seconds": result.removed_seconds,
            "kept_seconds": result.kept_seconds,
            "transcript": result.transcript,
            "removed_fillers": result.removed_fillers,
        })
    except Exception as e:  # noqa: BLE001 — surface any failure to the client
        JOBS[job_id].update({"status": "error", "error": str(e)})


@app.post("/upload", response_model=JobStatus)
async def upload(file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex
    ext = os.path.splitext(file.filename or "")[1] or ".mp4"
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    JOBS[job_id] = {"status": "queued"}
    thread = threading.Thread(target=_run_job, args=(job_id, input_path), daemon=True)
    thread.start()

    return JobStatus(job_id=job_id, status="queued")


@app.get("/status/{job_id}", response_model=JobStatus)
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        error=job.get("error"),
        removed_seconds=job.get("removed_seconds"),
        kept_seconds=job.get("kept_seconds"),
        transcript=job.get("transcript"),
    )


@app.get("/download/{job_id}")
def download(job_id: str):
    job = JOBS.get(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "file not ready")
    return FileResponse(job["output_path"], filename="edited.mp4")


# Serve the frontend. Mounted AFTER the API routes above so /upload, /status,
# /download keep working — StaticFiles only catches whatever isn't matched
# by an API route already.
# Guarded with isdir() because StaticFiles raises at import time (i.e. app
# startup) if the directory doesn't exist yet, which would crash the whole
# container before any routes could even load.
if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    # Lets you run this file directly (e.g. PyCharm's Run button) instead of
    # only via `uvicorn main:app --reload` from the terminal.
    # HF Spaces (and most free hosts) inject the port via $PORT — fall back
    # to 7860, the Hugging Face Spaces default, if it's not set.
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port)