#!/usr/bin/env python
"""Run Gemini CLI comparative packed image inference with checkpointed JSONL output."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "ride_video_infer"


def safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return cleaned or "workspace"


def default_workspace_root() -> Path:
    project_slug = safe_slug(Path.cwd().name)
    return Path.home() / ".gemini" / "tmp" / project_slug / "ride-video-infer"


def ensure_run_dir(preferred_root: Path, run_name: str) -> Path:
    preferred = preferred_root / run_name
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred
    except OSError:
        fallback = Path.cwd() / ".ride-video-infer-tmp" / run_name
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def parse_frame_number(path: Path, fallback: int) -> int:
    matches = re.findall(r"\d+", path.stem)
    if matches:
        return int(matches[-1])
    return fallback


def frames_from_image_dir(image_dir: Path, *, max_frames: int) -> list[dict[str, Any]]:
    image_paths = [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if max_frames > 0:
        image_paths = image_paths[:max_frames]
    frames: list[dict[str, Any]] = []
    for index, image_path in enumerate(image_paths, start=1):
        frame_number = parse_frame_number(image_path, index)
        frames.append(
            {
                "frame_number": frame_number,
                "timestamp_seconds": float(frame_number),
                "image_path": str(image_path.resolve()),
            }
        )
    return frames


def frames_from_index(index_path: Path, *, max_frames: int) -> list[dict[str, Any]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    frames = [frame for frame in payload["frames"] if frame.get("candidate") and frame.get("image_path")]
    if max_frames > 0:
        frames = frames[:max_frames]
    return frames


def load_existing_frame_numbers(path: Path, *, restart: bool) -> set[int]:
    if restart or not path.exists():
        return set()
    frame_numbers: set[int] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            frame_numbers.add(int(item["frame_number"]))
        except (KeyError, TypeError, ValueError):
            continue
    return frame_numbers


def write_decision_jsonl(path: Path, decisions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(decision, ensure_ascii=False) + "\n")


def reject_decision(frame: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "frame_number": int(frame["frame_number"]),
        "keep": False,
        "score": 0.0,
        "labels": [],
        "reason": "",
        "discard_reason": reason[:160],
    }


def normalize_decision(item: dict[str, Any]) -> dict[str, Any]:
    labels = item.get("labels") or []
    if isinstance(labels, str):
        labels = [labels]
    score = float(item.get("score", 0.0) or 0.0)
    if score > 1.0 and score <= 100.0:
        score /= 100.0
    return {
        "frame_number": int(item["frame_number"]),
        "keep": bool(item.get("keep", False)),
        "score": max(0.0, min(score, 1.0)),
        "labels": [str(label).strip() for label in labels if str(label).strip()],
        "reason": str(item.get("reason", "") or "").strip()[:160],
        "discard_reason": str(item.get("discard_reason", "") or "").strip()[:160],
    }


def find_decision_array(value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        if any("frame_number" in item for item in value):
            return value
    if isinstance(value, dict):
        for item in value.values():
            found = find_decision_array(item)
            if found is not None:
                return found
    if isinstance(value, str):
        parsed = parse_decision_array(value)
        if parsed is not None:
            return parsed
    return None


def parse_decision_array(text: str) -> list[dict[str, Any]] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if payload is not None:
        found = find_decision_array(payload)
        if found is not None:
            return found

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "[":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        found = find_decision_array(value)
        if found is not None:
            return found
    return None


def validate_pack_decisions(
    decisions: list[dict[str, Any]],
    expected_frame_numbers: list[int],
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    normalized_by_frame: dict[int, dict[str, Any]] = {}
    duplicates: list[int] = []
    expected = set(expected_frame_numbers)
    for item in decisions:
        try:
            normalized = normalize_decision(item)
        except (KeyError, TypeError, ValueError):
            continue
        frame_number = normalized["frame_number"]
        if frame_number not in expected:
            continue
        if frame_number in normalized_by_frame:
            duplicates.append(frame_number)
        normalized_by_frame[frame_number] = normalized
    missing = [frame_number for frame_number in expected_frame_numbers if frame_number not in normalized_by_frame]
    ordered = [normalized_by_frame[frame_number] for frame_number in expected_frame_numbers if frame_number in normalized_by_frame]
    return ordered, missing, duplicates


def copy_pack_images(pack: list[dict[str, Any]], pack_dir: Path) -> list[dict[str, Any]]:
    image_dir = pack_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    for frame in pack:
        src = Path(str(frame["image_path"])).expanduser().resolve()
        suffix = src.suffix.lower() if src.suffix else ".jpg"
        frame_number = int(frame["frame_number"])
        dst = image_dir / f"frame_{frame_number:09d}{suffix}"
        shutil.copy2(src, dst)
        updated = dict(frame)
        updated["workspace_image_path"] = str(dst)
        updated["workspace_image_name"] = dst.name
        copied.append(updated)
    return copied


def build_prompt(pack: list[dict[str, Any]], *, strict_retry: bool = False) -> str:
    max_keep = max(1, min(4, len(pack) // 5 + 1))
    lines = [
        "Use the ride-video-infer skill contract.",
        "Compare these motorcycle ride candidate frames as separate images, not as a contact sheet.",
        "Score frames relative to the group to find short-form highlight moments.",
        f"Keep only the strongest 1-{max_keep} frames unless many frames are genuinely strong.",
        "High scores: apex/lean, rapid transition, scenery reveal, nearby traffic, overtake, near pass, high speed, strong motion.",
        "Low scores: waiting, parking, boring straight, severe blur, repetitive low-value frames.",
        "",
        "Before deciding, inspect each listed image attachment from the current workspace.",
        "Frames to analyze:",
    ]
    for frame in pack:
        lines.append(
            f"- Frame {int(frame['frame_number'])}, timestamp {float(frame.get('timestamp_seconds', 0.0)):.3f}s, "
            f"image @images/{frame['workspace_image_name']}"
        )
    lines.extend(
        [
            "",
            "Return ONLY a valid JSON array. No markdown, no commentary, no code fences.",
            "Return exactly one object for every listed frame_number, with no missing or duplicate frame_number values.",
            "Object keys: frame_number, keep, score, labels, reason, discard_reason.",
            "score must be a number from 0.0 to 1.0. labels must be an array of short strings.",
        ]
    )
    if strict_retry:
        lines.extend(
            [
                "",
                "This is a retry after invalid output. Be strict: exactly one JSON object per requested frame_number.",
            ]
        )
    return "\n".join(lines)


def call_gemini(prompt: str, *, cwd: Path, model: str, timeout_seconds: int) -> str:
    executable = shutil.which("gemini")
    if executable is None:
        raise RuntimeError("Gemini CLI executable was not found on PATH.")
    cmd = [executable, "--yolo", "--output-format", "json"]
    if model.strip():
        cmd.extend(["--model", model.strip()])
    cmd.extend(["-p", prompt])
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        check=False,
    )
    return "\n".join(part for part in [result.stdout, result.stderr] if part)


def recommend_pack_size(args: argparse.Namespace) -> int:
    if args.pack_size:
        return int(args.pack_size)
    script_path = Path(__file__).with_name("recommend_pack_size.py")
    cmd = [sys.executable, str(script_path), "--daily-requests", str(args.daily_requests)]
    if args.index:
        cmd.extend(["--index", str(Path(args.index).expanduser().resolve())])
    else:
        cmd.extend(["--image-dir", str(Path(args.image_dir).expanduser().resolve())])
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "recommend_pack_size.py failed")
    payload = json.loads(result.stdout)
    return int(payload["recommended_pack_size"])


def default_output_path(args: argparse.Namespace) -> Path:
    if args.output_decisions:
        return Path(args.output_decisions).expanduser().resolve()
    if args.index:
        return Path(args.index).expanduser().resolve().parent.parent / "gemini_cli.frame_decisions.jsonl"
    return Path(args.image_dir).expanduser().resolve() / "gemini_cli.frame_decisions.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gemini CLI packed comparative image inference.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--index", help="Path to extract/index.json")
    source.add_argument("--image-dir", help="Directory of image files when no extract/index.json is available")
    parser.add_argument("--output-decisions", help="Destination JSONL. Defaults near the index/image folder.")
    parser.add_argument("--pack-size", type=int, default=0, help="Images per Gemini request. Defaults to recommendation.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit candidate frames/images for a test run.")
    parser.add_argument("--daily-requests", type=int, default=1500, help="Plan request quota used by pack recommendation.")
    parser.add_argument("--model", default="gemini-2.5-flash-lite", help="Gemini CLI model. Empty string uses CLI default.")
    parser.add_argument("--workspace", help="Temporary ASCII workspace for copied images.")
    parser.add_argument("--timeout-seconds", type=int, default=240, help="Timeout per pack.")
    parser.add_argument("--restart", action="store_true", help="Overwrite existing decisions instead of resuming.")
    parser.add_argument("--dry-run", action="store_true", help="Plan packs without calling Gemini CLI.")
    parser.add_argument("--apply", action="store_true", help="Run apply_decisions.py after inference. Requires --index.")
    parser.add_argument("--config", default="config.toml", help="Pipeline config path for --apply.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.index:
        frames = frames_from_index(Path(args.index).expanduser().resolve(), max_frames=int(args.max_frames))
    else:
        frames = frames_from_image_dir(Path(args.image_dir).expanduser().resolve(), max_frames=int(args.max_frames))
    if not frames:
        print(json.dumps({"status": "no_frames"}, indent=2))
        return 0

    pack_size = recommend_pack_size(args)
    output_path = default_output_path(args)
    if args.restart and output_path.exists():
        output_path.unlink()
    completed = load_existing_frame_numbers(output_path, restart=bool(args.restart))
    remaining = [frame for frame in frames if int(frame["frame_number"]) not in completed]

    workspace_root = (
        Path(args.workspace).expanduser().resolve()
        if args.workspace
        else default_workspace_root()
    )
    run_name = f"run_{int(time.time())}_{safe_stem(output_path.stem)}"
    run_dir = workspace_root / run_name if args.dry_run else ensure_run_dir(workspace_root, run_name)

    total_packs = math_ceil_div(len(remaining), pack_size)
    print(json.dumps(
        {
            "status": "running",
            "frames": len(frames),
            "remaining": len(remaining),
            "pack_size": pack_size,
            "packs": total_packs,
            "model": args.model,
            "output_decisions": str(output_path),
            "workspace": str(run_dir),
        },
        ensure_ascii=False,
        indent=2,
    ))
    if args.dry_run:
        return 0

    processed = 0
    kept = 0
    failed_packs = 0
    for pack_index in range(total_packs):
        start = pack_index * pack_size
        pack = remaining[start : start + pack_size]
        pack_dir = run_dir / f"pack_{pack_index + 1:04d}"
        copied_pack = copy_pack_images(pack, pack_dir)
        expected = [int(frame["frame_number"]) for frame in copied_pack]

        output = call_gemini(
            build_prompt(copied_pack),
            cwd=pack_dir,
            model=str(args.model),
            timeout_seconds=int(args.timeout_seconds),
        )
        (pack_dir / "raw_response.txt").write_text(output, encoding="utf-8")
        decisions = parse_decision_array(output) or []
        normalized, missing, duplicates = validate_pack_decisions(decisions, expected)
        if missing or duplicates:
            output = call_gemini(
                build_prompt(copied_pack, strict_retry=True),
                cwd=pack_dir,
                model=str(args.model),
                timeout_seconds=int(args.timeout_seconds),
            )
            (pack_dir / "raw_response.retry.txt").write_text(output, encoding="utf-8")
            decisions = parse_decision_array(output) or []
            normalized, missing, duplicates = validate_pack_decisions(decisions, expected)

        if missing or duplicates:
            failed_packs += 1
            present = {int(item["frame_number"]) for item in normalized}
            for frame in copied_pack:
                if int(frame["frame_number"]) not in present:
                    normalized.append(reject_decision(frame, "skill_error: missing or duplicate frame in gemini output"))
            normalized.sort(key=lambda item: int(item["frame_number"]))

        write_decision_jsonl(output_path, normalized)
        processed += len(normalized)
        kept += sum(1 for item in normalized if item.get("keep"))
        print(f"pack {pack_index + 1}/{total_packs}: wrote {len(normalized)} decisions, missing={missing}, duplicates={duplicates}")

    result: dict[str, Any] = {
        "status": "done",
        "output_decisions": str(output_path),
        "processed_frames": processed,
        "kept_frames": kept,
        "failed_packs": failed_packs,
        "pack_size": pack_size,
    }
    if args.apply:
        if not args.index:
            raise ValueError("--apply requires --index")
        apply_script = Path(__file__).with_name("apply_decisions.py")
        cmd = [
            sys.executable,
            str(apply_script),
            "--index",
            str(Path(args.index).expanduser().resolve()),
            "--decisions",
            str(output_path),
            "--config",
            str(args.config),
            "--provider",
            "gemini_cli",
        ]
        apply_result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        result["apply_returncode"] = apply_result.returncode
        result["apply_output"] = apply_result.stdout.strip()
        if apply_result.returncode != 0:
            result["apply_error"] = apply_result.stderr.strip()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def math_ceil_div(value: int, divisor: int) -> int:
    if value <= 0:
        return 0
    return (value + divisor - 1) // divisor


if __name__ == "__main__":
    raise SystemExit(main())
