from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class JobCreateRequest(BaseModel):
    video_path: str = Field(..., description="Source video path")
    target_seconds: int = Field(30, ge=5, le=300)
    mode: Literal["road", "track"] = "road"


class JobCreateResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int = Field(0, ge=0, le=100)
    stage: str | None = None
    result: dict | None = None
    error: str | None = None

