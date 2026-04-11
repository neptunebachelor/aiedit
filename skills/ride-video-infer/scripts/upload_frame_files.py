#!/usr/bin/env python
"""Upload extracted frame images to provider File APIs and write a stable manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IMAGE_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_MIME_BY_SUFFIX:
        return IMAGE_MIME_BY_SUFFIX[suffix]
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def default_manifest_path(index_path: Path, provider: str) -> Path:
    return index_path.parent.parent / "infer" / f"file_uploads.{provider}.json"


def load_existing_entries(path: Path | None) -> dict[tuple[int, str], dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = read_json(path)
    entries = payload.get("frames", []) if isinstance(payload, dict) else []
    existing: dict[tuple[int, str], dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            frame_number = int(entry["frame_number"])
        except (KeyError, TypeError, ValueError):
            continue
        sha256 = str(entry.get("sha256", "")).strip()
        if sha256:
            existing[(frame_number, sha256)] = entry
    return existing


def frame_rows_from_index(index_path: Path, *, include: str, max_frames: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = read_json(index_path)
    frames = []
    for frame in payload.get("frames", []):
        if not isinstance(frame, dict) or not frame.get("image_path"):
            continue
        if include == "candidates" and not frame.get("candidate"):
            continue
        frames.append(frame)
    if max_frames > 0:
        frames = frames[:max_frames]
    return payload, frames


def provider_refs(entry: dict[str, Any], provider: str) -> dict[str, Any]:
    if provider == "openai":
        return {"openai_file_id": str(entry.get("openai_file_id", "") or "")}
    if provider == "gemini":
        return {
            "gemini_file_name": str(entry.get("gemini_file_name", "") or ""),
            "gemini_file_uri": str(entry.get("gemini_file_uri", "") or ""),
        }
    raise ValueError(f"Unsupported provider: {provider}")


def has_provider_reference(entry: dict[str, Any], provider: str) -> bool:
    refs = provider_refs(entry, provider)
    return any(str(value).strip() for value in refs.values())


def upload_openai_frame(client: Any, image_path: Path) -> dict[str, Any]:
    with image_path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="vision")
    file_id = str(getattr(uploaded, "id", "") or "")
    if not file_id and isinstance(uploaded, dict):
        file_id = str(uploaded.get("id", "") or "")
    if not file_id:
        raise RuntimeError(f"OpenAI file upload did not return an id for {image_path}")
    return {"openai_file_id": file_id}


def upload_gemini_frame(client: Any, image_path: Path, *, mime_type: str) -> dict[str, Any]:
    uploaded = client.files.upload(
        file=str(image_path),
        config={
            "display_name": image_path.name,
            "mime_type": mime_type,
        },
    )
    file_name = str(getattr(uploaded, "name", "") or "")
    file_uri = str(getattr(uploaded, "uri", "") or "")
    if isinstance(uploaded, dict):
        file_name = file_name or str(uploaded.get("name", "") or "")
        file_uri = file_uri or str(uploaded.get("uri", "") or uploaded.get("file_uri", "") or "")
    if not file_uri:
        raise RuntimeError(f"Gemini file upload did not return a uri for {image_path}")
    return {"gemini_file_name": file_name, "gemini_file_uri": file_uri}


def make_client(provider: str) -> Any:
    if provider == "openai":
        from openai import OpenAI

        return OpenAI()
    if provider == "gemini":
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY", "").strip() or None
        return genai.Client(api_key=api_key)
    raise ValueError(f"Unsupported provider: {provider}")


def build_upload_entry(frame: dict[str, Any]) -> dict[str, Any]:
    image_path = Path(str(frame["image_path"])).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    return {
        "frame_number": int(frame["frame_number"]),
        "candidate": bool(frame.get("candidate", False)),
        "timestamp_seconds": float(frame.get("timestamp_seconds", 0.0) or 0.0),
        "timestamp_srt": str(frame.get("timestamp_srt", "") or ""),
        "local_image_path": str(image_path),
        "mime_type": detect_mime_type(image_path),
        "byte_size": image_path.stat().st_size,
        "sha256": sha256_file(image_path),
    }


def build_manifest(
    index_path: Path,
    *,
    provider: str,
    include: str,
    max_frames: int = 0,
    output_path: Path | None = None,
    reuse: bool = False,
    dry_run: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    if include not in {"candidates", "all"}:
        raise ValueError("--include must be candidates or all")
    if provider not in {"openai", "gemini"}:
        raise ValueError("--provider must be openai or gemini")
    index_path = index_path.expanduser().resolve()
    manifest_path = output_path.expanduser().resolve() if output_path else default_manifest_path(index_path, provider)
    _, frames = frame_rows_from_index(index_path, include=include, max_frames=max_frames)
    existing = load_existing_entries(manifest_path if reuse else None)
    upload_client = client if client is not None else (None if dry_run else make_client(provider))

    manifest_frames: list[dict[str, Any]] = []
    uploaded_count = 0
    reused_count = 0
    planned_count = 0
    failed_count = 0
    for frame in frames:
        entry = build_upload_entry(frame)
        reusable = existing.get((int(entry["frame_number"]), str(entry["sha256"])))
        if reuse and reusable and has_provider_reference(reusable, provider):
            entry.update(provider_refs(reusable, provider))
            entry["upload_status"] = "reused"
            reused_count += 1
            manifest_frames.append(entry)
            continue

        if dry_run:
            entry.update(provider_refs({}, provider))
            entry["upload_status"] = "planned"
            planned_count += 1
            manifest_frames.append(entry)
            continue

        try:
            image_path = Path(str(entry["local_image_path"]))
            if provider == "openai":
                refs = upload_openai_frame(upload_client, image_path)
            else:
                refs = upload_gemini_frame(upload_client, image_path, mime_type=str(entry["mime_type"]))
            entry.update(refs)
            entry["upload_status"] = "uploaded"
            uploaded_count += 1
        except Exception as exc:  # noqa: BLE001
            entry.update(provider_refs({}, provider))
            entry["upload_status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            failed_count += 1
        manifest_frames.append(entry)

    manifest = {
        "stage": "infer.file_uploads",
        "provider": provider,
        "source_extract_index": str(index_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "include": include,
        "frame_count": len(manifest_frames),
        "uploaded_count": uploaded_count,
        "reused_count": reused_count,
        "planned_count": planned_count,
        "failed_count": failed_count,
        "dry_run": dry_run,
        "frames": manifest_frames,
    }
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload extract/index.json frame images to a provider Files API.")
    parser.add_argument("--index", required=True, help="Path to extract/index.json")
    parser.add_argument("--provider", required=True, choices=["openai", "gemini"])
    parser.add_argument("--output", help="Destination upload manifest JSON")
    parser.add_argument("--include", choices=["candidates", "all"], default="candidates")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--reuse", action="store_true", help="Reuse matching uploaded file references from the output manifest")
    parser.add_argument("--dry-run", action="store_true", help="Write a planned manifest without uploading files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index_path = Path(args.index).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else None
    manifest = build_manifest(
        index_path,
        provider=str(args.provider),
        include=str(args.include),
        max_frames=int(args.max_frames),
        output_path=output_path,
        reuse=bool(args.reuse),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 2 if int(manifest.get("failed_count", 0) or 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
