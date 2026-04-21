"""
Lightweight remote pipeline server.

Run on the PC:
    python infer_server.py [--port 8765] [--host 0.0.0.0]

Clients (Pi / Mac) submit extract / infer jobs via HTTP.
No Redis, no RQ — state lives in memory.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import cv2

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from video_data_paths import resolve_video_data_root, safe_existing_slug, safe_video_slug

app = FastAPI(title="Pipeline Server", version="0.1.0")

TERMINAL_STATUSES = {"finished", "failed"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".ts"}

# In-memory job store: job_id → state dict
_jobs: dict[str, dict[str, Any]] = {}

# Set at startup via --videos-dir
_videos_dir: Path | None = None

# Set at startup via --data-root (or auto-detected); used by /ls/extracted and slug resolution
_data_root: Path | None = None


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ExtractJobRequest(BaseModel):
    video_path: str | None = Field(None, description="Absolute path to the video file on this PC")
    video_name: str | None = Field(None, description="Filename inside --videos-dir on this PC")


class InferJobRequest(BaseModel):
    index_path: str | None = Field(None, description="Absolute path to extract/index.json on this PC")
    slug: str | None = Field(None, description="Slug of an already-extracted video on this PC")
    shutdown: bool = Field(False, description="Shut down the PC after infer completes")


class JobStatus(BaseModel):
    job_id: str
    status: str              # queued | running | finished | failed
    stage: str | None = None
    progress: int = 0        # 0-100
    eta_seconds: int | None = None
    error: str | None = None
    result: dict | None = None


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------

def _probe_extract_total(video_path: str) -> int:
    """Return expected number of extracted frames (1 fps default)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    duration = frame_count / fps if fps > 0 else 0
    return max(1, int(duration))  # 1 frame per second


def _read_ffmpeg_progress_frame(progress_path: Path) -> int | None:
    """Parse the latest frame= value from an ffmpeg -progress file."""
    try:
        text = progress_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    last_frame = None
    for line in text.splitlines():
        if line.startswith("frame="):
            try:
                last_frame = int(line.split("=", 1)[1].strip())
            except ValueError:
                pass
    return last_frame


async def _poll_extract_progress(
    job: dict[str, Any],
    progress_path: Path,
    total_frames: int,
) -> None:
    """Coroutine that polls the ffmpeg progress file and updates job progress + ETA."""
    start = asyncio.get_event_loop().time()
    while True:
        await asyncio.sleep(1)
        frame = _read_ffmpeg_progress_frame(progress_path)
        if frame is not None and total_frames > 0:
            ratio = frame / total_frames
            job["progress"] = min(99, int(ratio * 100))
            elapsed = asyncio.get_event_loop().time() - start
            if ratio > 0.01:  # wait for at least 1% before estimating
                job["eta_seconds"] = int(elapsed / ratio * (1 - ratio))


async def _run_extract(job_id: str, video_path: str) -> None:
    job = _jobs[job_id]
    job["status"] = "running"
    job["stage"] = "extract"

    total_frames = _probe_extract_total(video_path)
    progress_file = Path(tempfile.mktemp(prefix="ffmpeg_progress_", suffix=".txt"))

    cmd = [
        sys.executable, "pipeline.py", "extract",
        "--video", video_path,
        "--progress-file", str(progress_file),
    ]
    proc = await asyncio.create_subprocess_exec(*cmd)

    poll_task = asyncio.create_task(
        _poll_extract_progress(job, progress_file, total_frames)
    )
    returncode = await proc.wait()
    poll_task.cancel()

    # Clean up temp progress file
    try:
        progress_file.unlink(missing_ok=True)
    except OSError:
        pass

    if returncode == 0:
        slug = safe_video_slug(video_path)
        index_path = resolve_video_data_root() / "videos" / slug / "extract" / "index.json"
        job["status"] = "finished"
        job["progress"] = 100
        job["result"] = {"index_path": str(index_path)}
    else:
        job["status"] = "failed"
        job["error"] = f"pipeline.py extract exited with code {returncode}"


async def _run_infer(job_id: str, index_path: str, video_path: str, shutdown: bool) -> None:
    job = _jobs[job_id]
    job["status"] = "running"
    job["stage"] = "infer"

    cmd = [
        sys.executable, "pipeline.py", "infer",
        "--extract-index", index_path,
        "--video", video_path,
    ]
    if shutdown:
        cmd.append("--shutdown")

    proc = await asyncio.create_subprocess_exec(*cmd)
    returncode = await proc.wait()

    if returncode == 0:
        job["status"] = "finished"
        job["progress"] = 100
    else:
        job["status"] = "failed"
        job["error"] = f"pipeline.py infer exited with code {returncode}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.post("/extract-jobs", response_model=JobStatus, status_code=202)
async def create_extract_job(payload: ExtractJobRequest) -> JobStatus:
    # Resolve video_path: exactly one of video_path / video_name must be provided
    if bool(payload.video_path) == bool(payload.video_name):
        raise HTTPException(status_code=400, detail="Provide exactly one of video_path or video_name")

    if payload.video_name:
        if _videos_dir is None:
            raise HTTPException(status_code=400, detail="Server has no --videos-dir configured")
        name = payload.video_name
        if "/" in name or ".." in name:
            raise HTTPException(status_code=400, detail="video_name must not contain path separators")
        resolved = (_videos_dir / name).resolve()
        if not resolved.is_relative_to(_videos_dir):
            raise HTTPException(status_code=400, detail="video_name resolves outside videos-dir")
        if resolved.suffix.lower() not in VIDEO_EXTS:
            raise HTTPException(status_code=400, detail=f"Not a video file: {name}")
        if not resolved.is_file():
            raise HTTPException(status_code=400, detail=f"video_name not found: {name}")
        video_path = str(resolved)
    else:
        video_path = payload.video_path
        if not Path(video_path).is_file():
            raise HTTPException(status_code=422, detail=f"video_path not found: {video_path}")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "stage": None,
        "progress": 0,
        "error": None,
        "result": None,
    }

    asyncio.create_task(_run_extract(job_id, video_path))

    return JobStatus(**_jobs[job_id])


@app.post("/infer-jobs", response_model=JobStatus, status_code=202)
async def create_infer_job(payload: InferJobRequest) -> JobStatus:
    # Resolve index_path: exactly one of index_path / slug must be provided
    if bool(payload.index_path) == bool(payload.slug):
        raise HTTPException(status_code=400, detail="Provide exactly one of index_path or slug")

    if payload.slug:
        clean = safe_existing_slug(payload.slug)
        index_path = (_data_root or resolve_video_data_root()) / "videos" / clean / "extract" / "index.json"
        if not index_path.is_file():
            raise HTTPException(status_code=404, detail=f"No extracted video with slug: {clean!r}")
    else:
        index_path = Path(payload.index_path)
        if not index_path.is_file():
            raise HTTPException(status_code=422, detail=f"index_path not found: {payload.index_path}")

    try:
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        video_path = index_data["video"]["source_path"]
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot read source_path from index: {exc}") from exc

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "stage": None,
        "progress": 0,
        "error": None,
        "result": None,
    }

    asyncio.create_task(_run_infer(job_id, str(index_path), video_path, payload.shutdown))

    return JobStatus(**_jobs[job_id])


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**job)


@app.get("/sse/jobs/{job_id}")
async def sse_job(job_id: str) -> StreamingResponse:
    """Server-Sent Events stream — pushes a progress event every second until terminal."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_stream():
        while True:
            job = _jobs.get(job_id)
            if job is None:
                break
            payload = json.dumps(JobStatus(**job).model_dump(), ensure_ascii=False)
            yield f"event: progress\ndata: {payload}\n\n"
            if job["status"] in TERMINAL_STATUSES:
                break
            await asyncio.sleep(1.0)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/jobs", response_model=list[JobStatus])
def list_jobs() -> list[JobStatus]:
    return [JobStatus(**j) for j in _jobs.values()]


@app.get("/ls/videos")
def list_videos() -> dict:
    """List raw video files in the directory configured via --videos-dir."""
    if _videos_dir is None or not _videos_dir.is_dir():
        return {"videos": [], "videos_dir": str(_videos_dir)}
    entries = []
    for f in sorted(_videos_dir.iterdir()):
        if f.suffix.lower() in VIDEO_EXTS and f.is_file():
            entries.append({
                "name": f.name,
                "path": str(f),
                "size_mb": round(f.stat().st_size / 1_048_576, 1),
            })
    return {"videos": entries, "videos_dir": str(_videos_dir)}


def _scan_extracted(data_root: Path) -> list[dict]:
    videos_dir = data_root / "videos"
    if not videos_dir.is_dir():
        return []
    entries = []
    for slug_dir in sorted(videos_dir.iterdir()):
        if not slug_dir.is_dir():
            continue
        index_path = slug_dir / "extract" / "index.json"
        if not index_path.is_file():
            continue
        source_path, frame_count = "-", 0
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            source_path = (payload.get("video") or {}).get("source_path") or "-"
            frame_count = len(payload.get("frames") or [])
        except Exception:
            pass
        has_infer = (slug_dir / "infer").is_dir() and any((slug_dir / "infer").iterdir())
        entries.append({
            "slug": slug_dir.name,
            "frames": frame_count,
            "has_infer": has_infer,
            "source_path": source_path,
            "index_path": str(index_path),
        })
    return entries


@app.get("/ls/extracted")
def list_extracted() -> dict:
    """List already-extracted videos in the server's data root."""
    root = _data_root or resolve_video_data_root()
    return {"data_root": str(root), "extracted": _scan_extracted(root)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global _videos_dir, _data_root

    parser = argparse.ArgumentParser(description="Lightweight remote infer HTTP server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    parser.add_argument("--videos-dir", default=None,
                        help="Directory of raw video files to expose via GET /ls/videos")
    parser.add_argument("--data-root", default=None,
                        help="Path to .video_data root (default: auto-detect from repo)")
    args = parser.parse_args()

    if args.videos_dir:
        _videos_dir = Path(args.videos_dir).expanduser().resolve()

    _data_root = Path(args.data_root).expanduser().resolve() if args.data_root \
                 else resolve_video_data_root()
    print(f"[infer_server] video data root: {_data_root}", file=sys.stderr)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
