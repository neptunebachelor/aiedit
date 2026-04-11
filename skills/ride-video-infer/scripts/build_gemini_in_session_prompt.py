#!/usr/bin/env python
"""Build a stable minimal prompt for in-session Gemini CLI pack inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{path} has a UTF-8 BOM; rewrite as UTF-8 without BOM")
    return json.loads(raw.decode("utf-8"))


def load_manifest_frames(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    frames = payload.get("frames") if isinstance(payload, dict) else payload
    if not isinstance(frames, list):
        raise ValueError(f"{path} does not contain a frames list")
    return [frame for frame in frames if isinstance(frame, dict)]


def frame_image_path(pack_dir: Path, frame: dict[str, Any]) -> Path:
    for key in ("relative_image_path", "copied_image_path", "temp_image_path", "image_path"):
        value = str(frame.get(key, "") or "").strip()
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = pack_dir / path
        return path
    frame_number = int(frame["frame_number"])
    candidates = sorted((pack_dir / "images").glob(f"*{frame_number:09d}*"))
    if candidates:
        return candidates[0]
    raise ValueError(f"No image path found for frame {frame_number} in {pack_dir}")


def gemini_path(path: Path) -> str:
    return path.resolve().as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a stable Gemini CLI in-session prompt.")
    parser.add_argument("--packs-dir", required=True, help="Directory containing pack_XXXX children")
    parser.add_argument("--start-pack", type=int, required=True, help="First 1-based pack number")
    parser.add_argument("--end-pack", type=int, required=True, help="Last 1-based pack number, inclusive")
    parser.add_argument(
        "--skill-path",
        default=str(Path(__file__).resolve().parents[1] / "SKILL.md"),
        help="Path to ride-video-infer/SKILL.md",
    )
    parser.add_argument("--output", help="Optional file to write the prompt to")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    packs_dir = Path(args.packs_dir).expanduser().resolve()
    skill_path = Path(args.skill_path).expanduser().resolve()
    start_pack = int(args.start_pack)
    end_pack = int(args.end_pack)
    if start_pack < 1:
        raise ValueError("--start-pack must be >= 1")
    if end_pack < start_pack:
        raise ValueError("--end-pack must be >= --start-pack")
    if not skill_path.exists():
        raise FileNotFoundError(skill_path)
    if not packs_dir.exists():
        raise FileNotFoundError(packs_dir)

    pack_lines: list[str] = []
    image_lines: list[str] = []
    for pack_number in range(start_pack, end_pack + 1):
        pack_name = f"pack_{pack_number:04d}"
        pack_dir = packs_dir / pack_name
        manifest_path = pack_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        frames = load_manifest_frames(manifest_path)
        if not frames:
            raise ValueError(f"{manifest_path} has no frames")
        pack_lines.append(
            f"- {pack_name}: manifest @{gemini_path(manifest_path)}, response {gemini_path(pack_dir / 'response.json')}, frames {len(frames)}"
        )
        image_lines.append(f"{pack_name} image inputs:")
        for frame in frames:
            frame_number = int(frame["frame_number"])
            timestamp = float(frame.get("timestamp_seconds", 0.0) or 0.0)
            image_path = frame_image_path(pack_dir, frame)
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            image_lines.append(
                f"- frame_number {frame_number}, timestamp {timestamp:.3f}s, image @{gemini_path(image_path)}"
            )

    prompt_lines = [
        "Read and follow this skill file exactly:",
        f"@{gemini_path(skill_path)}",
        "",
        "Task: run the skill's Gemini In-Session Pack Inference Contract for this deterministic pack range.",
        "",
        f"Skill directory: {gemini_path(skill_path.parent)}",
        f"Packs directory: {gemini_path(packs_dir)}",
        f"Start pack: {start_pack:04d}",
        f"End pack: {end_pack:04d}",
        "",
        "Packs to infer:",
        *pack_lines,
        "",
        "Deterministic image inputs:",
        *image_lines,
        "",
        "Do not run another gemini command.",
        "Do not infer packs outside this range.",
        "Use each manifest as the source of truth, inspect the pack images as separate image inputs, and write one response.json per pack.",
        "After writing response files, run the skill's BOM cleanup script for this exact pack range before reporting completion.",
        "After writing the response files, report only the response paths, frame counts, and any pack you could not complete.",
        "",
    ]
    prompt = "\n".join(prompt_lines)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(prompt, encoding="utf-8")
    print(prompt, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
