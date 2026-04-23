from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "pipeline.py").exists():
            return candidate
    raise RuntimeError("Could not find project root containing pipeline.py")


def resolve_video_data_root(repo_root: Path | None = None, override: str | Path | None = None) -> Path:
    """Resolve artifact root. Pure resolver — does not create the directory.

    Priority: explicit override arg > RIDE_VIDEO_DATA_ROOT env var > <repo>/.video_data/.
    """
    if override:
        return Path(override).expanduser().resolve()
    env_override = os.environ.get("RIDE_VIDEO_DATA_ROOT")
    if env_override:
        return Path(env_override).expanduser().resolve()
    return (repo_root or find_repo_root(Path(__file__))).resolve() / ".video_data"


def safe_video_slug(value: str | Path) -> str:
    stem = Path(str(value)).stem
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return slug or "video"


def safe_existing_slug(value: str | Path) -> str:
    raw = Path(str(value)).name
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    return slug or "video"


def video_artifact_dir(video_path: Path, *, data_root: Path | None = None) -> Path:
    return (data_root or resolve_video_data_root()) / "videos" / safe_video_slug(video_path)


def video_frames_dir(video_path: Path, *, data_root: Path | None = None) -> Path:
    return (data_root or resolve_video_data_root()) / "frames" / safe_video_slug(video_path)


def video_identity_from_payload(payload: dict[str, Any], fallback: Path | None = None) -> Path | str:
    video = payload.get("video", {}) if isinstance(payload, dict) else {}
    if isinstance(video, dict):
        source_path = str(video.get("source_path", "") or "").strip()
        if source_path:
            return Path(source_path)
        filename = str(video.get("filename", "") or "").strip()
        if filename:
            return filename
    if fallback is not None:
        return fallback
    return "video"


def read_json_if_exists(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def artifact_dir_from_payload(
    payload: dict[str, Any],
    *,
    fallback: Path | None = None,
    data_root: Path | None = None,
) -> Path:
    identity = video_identity_from_payload(payload, fallback)
    return (data_root or resolve_video_data_root()) / "videos" / safe_video_slug(identity)


def artifact_dir_from_index(index_path: Path, *, data_root: Path | None = None) -> Path:
    payload = read_json_if_exists(index_path)
    if payload:
        return artifact_dir_from_payload(payload, fallback=index_path.parent.parent, data_root=data_root)
    resolved = index_path.expanduser().resolve()
    root = (data_root or resolve_video_data_root()).resolve()
    try:
        resolved.relative_to(root / "videos")
        return resolved.parent.parent
    except ValueError:
        return root / "videos" / safe_video_slug(resolved.parent.parent.name)


def infer_dir_from_index(index_path: Path, *, data_root: Path | None = None) -> Path:
    return artifact_dir_from_index(index_path, data_root=data_root) / "infer"


def resolve_frame_image_path(
    frame: dict[str, Any],
    *,
    index_path: Path | None = None,
    payload: dict[str, Any] | None = None,
    data_root: Path | None = None,
) -> Path:
    raw = str(frame.get("image_path", "") or "").strip()
    if raw:
        raw_path = Path(raw).expanduser()
        if raw_path.exists():
            return raw_path.resolve()

    filename = Path(raw).name if raw else ""
    if not filename:
        frame_number = int(frame.get("frame_number", 0) or 0)
        filename = f"frame_{frame_number:09d}.jpg"

    root = data_root or resolve_video_data_root()
    candidates: list[Path] = []
    if payload:
        identity = video_identity_from_payload(payload, index_path.parent.parent if index_path else None)
        candidates.append(root / "frames" / safe_video_slug(identity) / filename)
    candidates.append(root / "frames" / filename)
    if index_path is not None:
        candidates.extend(
            [
                index_path.parent / "frames" / filename,
                index_path.parent.parent / "extract" / "frames" / filename,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if raw:
        return Path(raw).expanduser().resolve()
    return candidates[0].resolve()
