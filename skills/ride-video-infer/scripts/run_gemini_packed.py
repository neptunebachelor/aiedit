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


def robust_parse_decisions(raw_output: str) -> list[dict[str, Any]] | None:
    """Parse decisions from Gemini CLI JSON output."""
    try:
        # 1. First, parse the entire CLI output as JSON
        cli_payload = json.loads(raw_output)
        
        # 2. Extract the model's text response
        # Gemini CLI usually puts the text in a 'text' or 'content' field depending on version
        text_content = ""
        if isinstance(cli_payload, dict):
            text_content = cli_payload.get("response") or cli_payload.get("text") or cli_payload.get("content") or ""
        elif isinstance(cli_payload, list):
            # Sometimes it might return a list of responses
            for part in cli_payload:
                if isinstance(part, dict) and "text" in part:
                    text_content += part["text"]
        
        if not text_content:
            # If we couldn't find a text field, maybe the output WAS just the array?
            if isinstance(cli_payload, list):
                return cli_payload
            return None

        # 3. Clean markdown and extract the JSON array from the text
        text_content = text_content.strip()
        if "```" in text_content:
            match = re.search(r"```(?:json)?\s*(\[.*\])\s*```", text_content, re.DOTALL)
            if match:
                text_content = match.group(1)
        
        start = text_content.find("[")
        end = text_content.rfind("]")
        if start != -1 and end != -1:
            json_str = text_content[start : end + 1]
            data = json.loads(json_str)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    
    # Fallback: Try a brute-force search for an array in the raw string
    try:
        match = re.search(r"(\[\s*\{.*\}\s*\])", raw_output, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except Exception:
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
        # Check if it failed but still produced some JSON (e.g. partial response)
        if stdout.strip().startswith("{") or stdout.strip().startswith("["):
            return stdout
        raise GeminiCLIError(f"Gemini failed with {process.returncode}", output=stderr or stdout)
    
    return stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--pack-size", type=int, default=10)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--timeout-seconds", type=int, default=300)
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
        
        # Prepare images
        img_dir = pack_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        for f in batch:
            src = Path(f["image_path"])
            shutil.copy2(src, img_dir / src.name)
        
        prompt_lines = [
            "Analyze the following motorcycle riding images and return a JSON array containing your decisions.",
            "CRITICAL INSTRUCTIONS FOR AGENT:",
            "1. DO NOT USE ANY TOOLS. You already have the images attached. Output the result immediately.",
            "2. You MUST output exactly one JSON array. Do not include any other text, no markdown blocks, no conversational fillers.",
            "3. You MUST use the EXACT keys specified below. DO NOT invent your own keys like 'image', 'decision', or 'reasoning'.",
            "",
            "Example of the EXACT required output format:",
            '[{"frame_number": 230, "keep": false, "score": 0.1, "labels": [], "reason": "", "discard_reason": "blurry"}]',
            "",
            "Each item in the JSON array must strictly match this exact schema:",
            '{"frame_number": number, "keep": boolean, "score": number, "labels": string[], "reason": string, "discard_reason": string}',
            "Instructions for fields:",
            "- frame_number: MUST be the integer frame number provided in the list.",
            "- score: Float from 0.0 to 1.0.",
            "- keep: Set to true ONLY for visually strong, useful, non-duplicate frames that would make good highlights.",
            "- reason: Brief reason why the frame is kept (or empty if not kept).",
            "- discard_reason: Brief reason if the frame is discarded.",
            "Images to analyze:"
        ]
        for f in batch:
            prompt_lines.append(f"- Frame {f['frame_number']}, timestamp {f['timestamp_seconds']}s, image @images/{Path(f['image_path']).name}")
            
        prompt_lines.append("\nAnalyze the images above and output the JSON array of decisions now. DO NOT USE TOOLS. OUTPUT EXACTLY ONE JSON ARRAY AND NOTHING ELSE.")
        
        prompt = "\n".join(prompt_lines)
        log(f"[{pack_num}/{total_packs}] Processing {len(batch)} frames...")
        
        try:
            raw_output = call_gemini(prompt, cwd=pack_dir, model=args.model, timeout_seconds=args.timeout_seconds)
            (pack_dir / "raw_output.txt").write_text(raw_output, encoding="utf-8")
            
            decisions = robust_parse_decisions(raw_output)
            
            if decisions:
                normalized = [normalize_decision(d) for d in decisions]
                present = {d["frame_number"] for d in normalized}
                # Fix missing frames in response
                for f in batch:
                    if f["frame_number"] not in present:
                        normalized.append({"frame_number": f["frame_number"], "keep": False, "score": 0.0, "labels": [], "reason": "", "discard_reason": "provider_error: missing from output"})
                
                write_decision_jsonl(output_path, normalized)
                log(f"  Done. Kept {sum(1 for d in normalized if d['keep'])} frames.")
            else:
                log(f"  Error: Failed to parse decisions from JSON output. Check {pack_dir / 'raw_output.txt'}")
        except Exception as e:
            log(f"  Error: {e}")
            
    if args.apply:
        apply_script = REPO_ROOT / "skills/ride-video-infer/scripts/apply_decisions.py"
        subprocess.run([sys.executable, str(apply_script), "--index", str(index_path), "--decisions", str(output_path), "--provider", "gemini_cli"], check=True)
        log(f"\nFinal analysis.json materialized.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
