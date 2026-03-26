from __future__ import annotations

import time

from redis import Redis
from rq import get_current_job

from backend.config import settings

redis_conn = Redis.from_url(settings.redis_url)


def _set_progress(job_id: str, stage: str, progress: int) -> None:
    redis_conn.hset(
        f"job:{job_id}:state",
        mapping={
            "stage": stage,
            "progress": str(progress),
        },
    )


def run_pipeline(video_path: str, target_seconds: int, mode: str) -> dict:
    """
    Demo background task for long-running pipeline work.
    Replace the sleep blocks with real extract/infer/review/render calls.
    """
    job = get_current_job()
    if job is None:
        raise RuntimeError("This task must run under an RQ worker.")

    stages = [
        ("extract", 20),
        ("infer", 55),
        ("review", 80),
        ("render", 100),
    ]

    _set_progress(job.id, "queued", 0)
    for stage_name, stage_progress in stages:
        time.sleep(1.0)
        _set_progress(job.id, stage_name, stage_progress)

    result = {
        "video_path": video_path,
        "target_seconds": target_seconds,
        "mode": mode,
        "output_dir": "./output/demo",
    }
    _set_progress(job.id, "completed", 100)
    return result

