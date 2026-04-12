#!/usr/bin/env python
"""Create a small calibration extract index from an existing index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pipeline.py").exists():
            return candidate
    raise RuntimeError("Could not find project root containing pipeline.py")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
sys.path.insert(0, str(REPO_ROOT))

from video_data_paths import artifact_dir_from_index  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a candidate-only calibration index.")
    parser.add_argument("--index", required=True, help="Source extract/index.json")
    parser.add_argument("--limit", type=int, default=100, help="Maximum frames to keep")
    parser.add_argument("--output", help="Output path. Defaults under the canonical .video_data tree.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index_path = Path(args.index).expanduser().resolve()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    frames = [frame for frame in payload.get("frames", []) if isinstance(frame, dict) and frame.get("image_path")]
    selected = frames[: max(0, int(args.limit))]
    for frame in selected:
        frame["candidate"] = True
    payload["frames"] = selected
    payload["sampled_frames"] = len(selected)
    payload["candidate_frames"] = len(selected)
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else artifact_dir_from_index(index_path) / "debug" / f"temp_index_{len(selected)}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "frames": len(selected)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
