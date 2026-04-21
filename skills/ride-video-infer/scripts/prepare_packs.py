#!/usr/bin/env python
"""Prepare non-nested Gemini CLI image packs from extract/index.json."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pipeline.py").exists():
            return candidate
    raise RuntimeError("Could not find project root containing pipeline.py")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
sys.path.insert(0, str(REPO_ROOT))

from video_data_paths import infer_dir_from_index  # noqa: E402


def load_candidate_frames(index_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return [frame for frame in payload["frames"] if frame.get("candidate") and frame.get("image_path")]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Gemini visual-inference packs without calling Gemini.")
    parser.add_argument("--index", required=True, help="Path to extract/index.json")
    parser.add_argument("--pack-size", type=int, default=20, help="Candidate frames per pack")
    parser.add_argument("--output-dir", help="Infer directory. Defaults to the canonical .video_data video infer directory")
    parser.add_argument("--start-pack", type=int, default=1, help="First 1-based pack number to prepare")
    parser.add_argument("--end-pack", type=int, default=0, help="Last 1-based pack number to prepare, inclusive")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional candidate frame limit for calibration")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing manifests/images")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index_path = Path(args.index).expanduser().resolve()
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    candidate_frames = [frame for frame in index_payload["frames"] if frame.get("candidate") and frame.get("image_path")]
    if args.max_frames > 0:
        candidate_frames = candidate_frames[: int(args.max_frames)]
    pack_size = max(1, int(args.pack_size))
    total_packs = math.ceil(len(candidate_frames) / pack_size) if candidate_frames else 0
    start_pack = max(1, int(args.start_pack))
    end_pack = int(args.end_pack) if int(args.end_pack) > 0 else total_packs
    if args.output_dir:
        infer_dir = Path(args.output_dir).expanduser().resolve()
    else:
        infer_dir = infer_dir_from_index(index_path)
    packs_dir = infer_dir / "packs"
    prepared: list[dict[str, Any]] = []

    for pack_number in range(start_pack, min(end_pack, total_packs) + 1):
        start = (pack_number - 1) * pack_size
        pack_frames = candidate_frames[start : start + pack_size]
        pack_dir = packs_dir / f"pack_{pack_number:04d}"
        images_dir = pack_dir / "images"
        manifest_path = pack_dir / "manifest.json"
        if manifest_path.exists() and not args.overwrite:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            prepared.append(
                {
                    "pack": f"pack_{pack_number:04d}",
                    "status": "exists",
                    "manifest_path": str(manifest_path),
                    "response_path": str(pack_dir / "response.json"),
                    "frame_count": len(manifest.get("frames", [])),
                }
            )
            continue

        images_dir.mkdir(parents=True, exist_ok=True)
        manifest_frames: list[dict[str, Any]] = []
        for frame in pack_frames:
            # Directly construct source_path assuming image_path is relative to pack_0013
            source_path = index_path.parent.parent / frame["image_path"]
            suffix = source_path.suffix or ".jpg"
            frame_number = int(frame["frame_number"])
            copied_path = images_dir / f"frame_{frame_number:09d}{suffix.lower()}"
            shutil.copy2(source_path, copied_path)
            manifest_frames.append(
                {
                    "frame_number": frame_number,
                    "timestamp_seconds": float(frame.get("timestamp_seconds", 0.0)),
                    "timestamp_srt": str(frame.get("timestamp_srt", "")),
                    "original_image_path": str(source_path),
                    "copied_image_path": str(copied_path),
                    "relative_image_path": f"images/{copied_path.name}",
                    "image_width": int(frame.get("image_width") or 0),
                    "image_height": int(frame.get("image_height") or 0),
                }
            )

        manifest_payload = {
            "stage": "infer.pack",
            "source_extract_index": str(index_path),
            "pack_number": pack_number,
            "pack_size": pack_size,
            "frame_count": len(manifest_frames),
            "frames": manifest_frames,
            "response_path": str(pack_dir / "response.json"),
        }
        write_json(manifest_path, manifest_payload)
        prepared.append(
            {
                "pack": f"pack_{pack_number:04d}",
                "status": "prepared",
                "manifest_path": str(manifest_path),
                "response_path": str(pack_dir / "response.json"),
                "frame_count": len(manifest_frames),
                "first_frame": manifest_frames[0]["frame_number"] if manifest_frames else None,
                "last_frame": manifest_frames[-1]["frame_number"] if manifest_frames else None,
            }
        )

    summary = {
        "infer_dir": str(infer_dir),
        "packs_dir": str(packs_dir),
        "candidate_frames": len(candidate_frames),
        "pack_size": pack_size,
        "total_packs": total_packs,
        "prepared": prepared,
    }
    write_json(infer_dir / "pack_plan.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
