#!/usr/bin/env python
"""Recommend how many candidate frame images to send per Gemini/Codex vision call."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image, UnidentifiedImageError


GEMINI_IMAGE_TILE_TOKENS = 258
GEMINI_TILE_SIZE = 768
DEFAULT_CONTEXT_BUDGET = 1_000_000
DEFAULT_SAFE_INPUT_FRACTION = 0.25
DEFAULT_OUTPUT_TOKENS_PER_FRAME = 90
DEFAULT_PROMPT_OVERHEAD_TOKENS = 900
DEFAULT_INLINE_BYTES_LIMIT = 20 * 1024 * 1024
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def gemini_image_tokens(width: int, height: int) -> int:
    if width <= 0 or height <= 0:
        return GEMINI_IMAGE_TILE_TOKENS
    if width <= 384 and height <= 384:
        return GEMINI_IMAGE_TILE_TOKENS
    tiles = max(1, math.ceil(width / GEMINI_TILE_SIZE) * math.ceil(height / GEMINI_TILE_SIZE))
    return tiles * GEMINI_IMAGE_TILE_TOKENS


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def read_image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except (OSError, UnidentifiedImageError):
        return 0, 0


def frames_from_index(index_path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    frames = [frame for frame in payload["frames"] if frame.get("candidate") and frame.get("image_path")]
    return frames, str(index_path)


def frames_from_image_dir(image_dir: Path) -> tuple[list[dict[str, Any]], str]:
    image_paths = [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    frames: list[dict[str, Any]] = []
    for index, path in enumerate(image_paths, start=1):
        width, height = read_image_size(path)
        frames.append(
            {
                "frame_number": index,
                "timestamp_seconds": float(index - 1),
                "candidate": True,
                "image_path": str(path),
                "image_width": width,
                "image_height": height,
            }
        )
    return frames, str(image_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommend packed image count for ride-video-infer.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--index", help="Path to extract/index.json")
    source.add_argument("--image-dir", help="Directory of image files when no extract/index.json is available")
    parser.add_argument("--daily-requests", type=int, default=1500, help="Plan request quota, e.g. 1500 for Google AI Pro")
    parser.add_argument("--context-budget", type=int, default=DEFAULT_CONTEXT_BUDGET, help="Model context budget in tokens")
    parser.add_argument(
        "--safe-input-fraction",
        type=float,
        default=DEFAULT_SAFE_INPUT_FRACTION,
        help="Fraction of context allowed for prompt+images+expected output",
    )
    parser.add_argument(
        "--default-pack",
        type=int,
        default=20,
        help="Preferred comparative pack size before applying safety bounds.",
    )
    parser.add_argument(
        "--max-pack",
        type=int,
        default=32,
        help="Upper bound for reliable comparative JSON output. Raise only after calibration.",
    )
    parser.add_argument(
        "--target-requests",
        type=int,
        default=0,
        help="Optional desired request count for this video; recommendation will try to fit within it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.index:
        frames, source_path = frames_from_index(Path(args.index).expanduser().resolve())
    else:
        frames, source_path = frames_from_image_dir(Path(args.image_dir).expanduser().resolve())
    if not frames:
        print(json.dumps({"candidate_frames": 0, "recommendation": "no candidate frames"}, indent=2))
        return 0

    token_estimates: list[int] = []
    byte_sizes: list[int] = []
    for frame in frames:
        width = int(frame.get("image_width") or 0)
        height = int(frame.get("image_height") or 0)
        token_estimates.append(gemini_image_tokens(width, height))
        image_path = Path(str(frame.get("image_path", "")))
        try:
            byte_sizes.append(image_path.stat().st_size)
        except OSError:
            pass

    p95_tokens = max(1, int(math.ceil(percentile([float(value) for value in token_estimates], 0.95))))
    p95_bytes = max(1, int(math.ceil(percentile([float(value) for value in byte_sizes], 0.95)))) if byte_sizes else 250_000
    usable_tokens = int(args.context_budget * args.safe_input_fraction)
    token_bound = max(
        1,
        (usable_tokens - DEFAULT_PROMPT_OVERHEAD_TOKENS)
        // (p95_tokens + DEFAULT_OUTPUT_TOKENS_PER_FRAME),
    )
    bytes_bound = max(1, int((DEFAULT_INLINE_BYTES_LIMIT * 0.80) // p95_bytes))

    reliability_bound = max(1, int(args.max_pack))
    hard_bound = clamp(min(token_bound, bytes_bound, reliability_bound), 1, reliability_bound)
    if args.target_requests > 0:
        needed_for_target = math.ceil(len(frames) / args.target_requests)
        recommended = clamp(needed_for_target, 1, hard_bound)
    else:
        recommended = clamp(int(args.default_pack), 1, hard_bound)

    result: dict[str, Any] = {
        "source_path": source_path,
        "candidate_frames": len(frames),
        "image_tokens": {
            "median": int(median(token_estimates)),
            "p95": p95_tokens,
            "max": max(token_estimates),
        },
        "image_bytes": {
            "median": int(median(byte_sizes)) if byte_sizes else 0,
            "p95": p95_bytes if byte_sizes else 0,
            "max": max(byte_sizes) if byte_sizes else 0,
        },
        "bounds": {
            "token_bound": token_bound,
            "inline_bytes_bound": bytes_bound,
            "json_reliability_bound": reliability_bound,
            "safe_upper_bound": hard_bound,
        },
        "recommended_pack_size": recommended,
        "estimated_requests_for_video": math.ceil(len(frames) / recommended),
        "estimated_frames_per_day": {
            "strict_1x": int(args.daily_requests),
            f"packed_{recommended}x": int(args.daily_requests) * recommended,
        },
        "modes": {
            "strict": 1,
            "comparative_default": recommended,
            "aggressive_after_calibration": hard_bound,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
