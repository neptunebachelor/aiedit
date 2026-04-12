#!/usr/bin/env python
"""Print candidate frame numbers from an extract index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect candidate frame numbers in an extract index.")
    parser.add_argument("--index", required=True, help="Path to extract/index.json")
    parser.add_argument("--start", type=int, default=0, help="Zero-based candidate offset")
    parser.add_argument("--count", type=int, default=80, help="Number of candidates to print")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index_path = Path(args.index).expanduser().resolve()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    candidates = [frame for frame in payload.get("frames", []) if isinstance(frame, dict) and frame.get("candidate")]
    start = max(0, int(args.start))
    end = min(len(candidates), start + max(0, int(args.count)))
    for offset in range(start, end):
        frame = candidates[offset]
        print(f'Idx {offset}: frame {frame["frame_number"]}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
