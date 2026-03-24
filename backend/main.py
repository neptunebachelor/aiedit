from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from rq.job import Job

from backend.models import JobCreateRequest, JobCreateResponse, JobStatusResponse
from backend.queue import redis_conn, task_queue
from backend.tasks import run_pipeline

app = FastAPI(title="AI Highlight Backend", version="0.1.0")
TERMINAL_STATUSES = {"finished", "failed", "stopped", "canceled", "cancelled"}


def _read_job_state(job_id: str) -> dict:
    state = redis_conn.hgetall(f"job:{job_id}:state")
    decoded = {k.decode("utf-8"): v.decode("utf-8") for k, v in state.items()}
    try:
        progress = int(decoded.get("progress", "0"))
    except ValueError:
        progress = 0
    return {
        "stage": decoded.get("stage"),
        "progress": max(0, min(100, progress)),
    }


def _build_status(job: Job) -> JobStatusResponse:
    state = _read_job_state(job.id)
    status = job.get_status(refresh=True)
    return JobStatusResponse(
        job_id=job.id,
        status=status,
        progress=state["progress"],
        stage=state["stage"],
        result=job.result if status == "finished" else None,
        error=str(job.exc_info) if status == "failed" else None,
    )


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.post("/jobs", response_model=JobCreateResponse)
def create_job(payload: JobCreateRequest) -> JobCreateResponse:
    job = task_queue.enqueue(
        run_pipeline,
        payload.video_path,
        payload.target_seconds,
        payload.mode,
        job_timeout="2h",
        result_ttl=24 * 3600,
    )
    redis_conn.hset(f"job:{job.id}:state", mapping={"stage": "queued", "progress": "0"})
    return JobCreateResponse(job_id=job.id, status=job.get_status())


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    return _build_status(job)


@app.websocket("/ws/jobs/{job_id}")
async def ws_job_progress(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    try:
        while True:
            try:
                job = Job.fetch(job_id, connection=redis_conn)
                payload = _build_status(job).model_dump()
            except Exception:
                payload = {"job_id": job_id, "status": "missing", "progress": 0}

            await websocket.send_text(json.dumps(payload, ensure_ascii=False))
            if payload.get("status") in TERMINAL_STATUSES:
                break
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass


@app.get("/sse/jobs/{job_id}")
async def sse_job_progress(job_id: str) -> StreamingResponse:
    async def event_stream():
        while True:
            try:
                job = Job.fetch(job_id, connection=redis_conn)
                payload = _build_status(job).model_dump()
            except Exception:
                payload = {"job_id": job_id, "status": "missing", "progress": 0}

            yield "event: progress\n"
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if payload.get("status") in TERMINAL_STATUSES:
                break
            await asyncio.sleep(1.0)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
