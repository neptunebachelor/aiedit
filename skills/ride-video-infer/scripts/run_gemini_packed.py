#!/usr/bin/env python
"""Run Gemini CLI comparative packed image inference with checkpointed JSONL output.
Supercharged version with robust JSON parsing and --yolo enforcement.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pipeline.py").exists():
            return candidate
    raise RuntimeError("Could not find project root containing pipeline.py")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
sys.path.insert(0, str(REPO_ROOT))

from video_data_paths import infer_dir_from_index, resolve_frame_image_path, resolve_video_data_root  # noqa: E402


class GeminiCLIError(RuntimeError):
    def __init__(self, message: str, *, output: str = "", returncode: int | None = None) -> None:
        super().__init__(message)
        self.output = output
        self.returncode = returncode


class GeminiCLITimeoutError(TimeoutError):
    def __init__(self, message: str, *, output: str = "") -> None:
        super().__init__(message)
        self.output = output


def safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "ride_video_infer"


def ensure_run_dir(preferred_root: Path, run_name: str) -> Path:
    preferred = preferred_root / run_name
    preferred.mkdir(parents=True, exist_ok=True)
    return preferred


def frames_from_index(index_path: Path, *, max_frames: int) -> list[dict[str, Any]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    frames = [frame for frame in payload["frames"] if frame.get("candidate") and frame.get("image_path")]
    if max_frames > 0:
        frames = frames[:max_frames]
    for frame in frames:
        frame["image_path"] = str(resolve_frame_image_path(frame, index_path=index_path, payload=payload))
    return frames


def load_existing_frame_numbers(path: Path) -> set[int]:
    if not path.exists():
        return set()
    frame_numbers: set[int] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            frame_numbers.add(int(item["frame_number"]))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return frame_numbers


def write_decision_jsonl(path: Path, decisions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(decision, ensure_ascii=False) + "\n")


def log(message: str) -> None:
    print(message, flush=True)


def normalize_decision(item: dict[str, Any]) -> dict[str, Any]:
    labels = item.get("labels") or []
    if isinstance(labels, str):
        labels = [labels]
    score = float(item.get("score", 0.0) or 0.0)
    if score > 1.0:
        score /= 100.0
    return {
        "frame_number": int(item["frame_number"]),
        "keep": bool(item.get("keep", False)),
        "score": max(0.0, min(score, 1.0)),
        "labels": [str(label).strip() for label in labels if str(label).strip()],
        "reason": str(item.get("reason", "") or "").strip()[:160],
        "discard_reason": str(item.get("discard_reason", "") or "").strip()[:160],
    }


def robust_parse_json_array(text: str) -> list[dict[str, Any]] | None:
    text = text.strip()
    # Try stripping code fences
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(\[.*\])\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
    
    # Try finding the first '[' and last ']'
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        json_candidate = text[start : end + 1]
        try:
            data = json.loads(json_candidate)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    return None


def call_gemini(
    prompt: str,
    *,
    cwd: Path,
    model: str,
    timeout_seconds: int,
) -> str:
    executable = shutil.which("gemini")
    if executable is None:
        raise RuntimeError("Gemini CLI executable was not found on PATH.")
    
    # Enforcement of --yolo and --output-format json
    cmd = [executable, "--yolo", "--output-format", "json"]
    if model.strip():
        cmd.extend(["--model", model.strip()])
    cmd.extend(["-p", prompt])
    
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
        else:
            process.kill()
        raise GeminiCLITimeoutError("Gemini CLI timeout")
        
    if process.returncode != 0:
        raise GeminiCLIError(f"Gemini failed with {process.returncode}", output=stderr or stdout)
    
    return stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--pack-size", type=int, default=26)
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    index_path = Path(args.index).resolve()
    frames = frames_from_index(index_path, max_frames=0)
    output_path = infer_dir_from_index(index_path) / "gemini_cli.frame_decisions.jsonl"
    
    completed = load_existing_frame_numbers(output_path)
    remaining = [f for f in frames if int(f["frame_number"]) not in completed]
    
    if not remaining:
        log("All frames already processed.")
        return 0

    run_dir = output_path.parent / "gemini_cli_runs" / f"run_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log(f"Starting inference: {len(remaining)} frames remaining, {args.pack_size} frames per pack.")
    
    total_packs = (len(remaining) + args.pack_size - 1) // args.pack_size
    
    for i in range(total_packs):
        batch = remaining[i * args.pack_size : (i + 1) * args.pack_size]
        pack_num = i + 1
        pack_dir = run_dir / f"pack_{pack_num:04d}"
        pack_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare prompt
        img_refs = []
        img_dir = pack_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        
        prompt_lines = [
            "Use the ride-video-infer skill contract.",
            "Compare these frames and return a JSON array.",
            "Keep only strong highlights (score >= 0.65).",
            "Frames:"
        ]
        
        for f in batch:
            src = Path(f["image_path"])
            dst = img_dir / src.name
            shutil.copy2(src, dst)
            prompt_lines.append(f"- Frame {f['frame_number']}, timestamp {f['timestamp_seconds']}s, image @images/{src.name}")
        
        prompt_lines.append("\nReturn ONLY a plain JSON array [{\"frame_number\":...}]. No markdown.")
        prompt = "\n".join(prompt_lines)
        
        log(f"[{pack_num}/{total_packs}] Processing {len(batch)} frames...")
        
        try:
            raw_output = call_gemini(prompt, cwd=pack_dir, model=args.model, timeout_seconds=args.timeout_seconds)
            decisions = robust_parse_json_array(raw_output)
            
            if decisions:
                normalized = [normalize_decision(d) for d in decisions]
                # Ensure we have a decision for every frame in batch
                present = {d["frame_number"] for d in normalized}
                for f in batch:
                    if f["frame_number"] not in present:
                        normalized.append({"frame_number": f["frame_number"], "keep": False, "score": 0.0, "labels": [], "reason": "", "discard_reason": "provider_error: missing from output"})
                
                write_decision_jsonl(output_path, normalized)
                log(f"  Done. Kept {sum(1 for d in normalized if d['keep'])} frames.")
            else:
                log(f"  Error: Failed to parse JSON. Skipping pack.")
        except Exception as e:
            log(f"  Error: {e}")
            
    if args.apply:
        apply_script = REPO_ROOT / "skills/ride-video-infer/scripts/apply_decisions.py"
        subprocess.run([sys.executable, str(apply_script), "--index", str(index_path), "--decisions", str(output_path), "--provider", "gemini_cli"], check=True)
        log(f"\nFinal analysis.json materialized.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
