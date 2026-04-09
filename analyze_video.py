from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
import tomllib
from tqdm import tqdm


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {
        "input": "input",
        "output": "output",
        "thumbnail_dirname": "thumbnails",
        "video_extensions": [".mp4", ".mov", ".m4v", ".avi"],
    },
    "sampling": {
        "sample_fps": 1.0,
        "forced_keep_interval_seconds": 3.0,
        "jpeg_quality": 88,
        "max_frames": 0,
    },
    "filters": {
        "min_blur_score": 80.0,
        "min_frame_diff": 6.0,
        "min_hash_distance": 6,
    },
    "ollama": {
        "enabled": True,
        "host": "http://127.0.0.1:11434",
        "model": "llava:7b",
        "temperature": 0.1,
        "timeout_seconds": 120,
    },
    "prompt": {
        "preset": "default",
        "positive_labels": [
            "bend",
            "scenery",
            "traffic",
            "overtake",
            "group_ride",
            "tunnel_transition",
            "water_view",
            "mountain_view",
            "sunset",
        ],
        "negative_labels": [
            "stop",
            "waiting",
            "parking",
            "blur",
            "low_value_straight",
        ],
        "extra_positive_labels": [],
        "extra_negative_labels": [],
        "extra_instructions": "",
        "system_instructions": (
            "You review forward-fixed motorcycle footage exported from Insta360 Studio. "
            "Keep only visually valuable moments for a short-form highlight edit. "
            "Prioritize obvious bends, strong scenery, nearby vehicles, overtakes, group riding, "
            "and tunnel transitions. Reject red lights, parking, boring straight roads, and blur. "
            "Return valid JSON only."
        ),
    },
    "decision": {
        "min_keep_score": 0.65,
        "merge_gap_seconds": 4.0,
        "padding_before_seconds": 1.5,
        "padding_after_seconds": 2.5,
        "min_segment_seconds": 2.0,
        "max_reason_chars": 80,
    },
}

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi"}

PROMPT_PRESET_SYSTEM_APPEND: dict[str, str] = {
    "default": "",
    "douyin_riding": (
        "Optimize for short-form riding highlights that feel strong in the first three seconds. "
        "Prefer moments with visible lean angle changes, overtake tension, close traffic interaction, "
        "tunnel in/out transitions, scenery bursts, speed sensation, and clear motion progression. "
        "For JSON reasons, use concise Chinese phrases that are useful as short-video editing notes."
    ),
}

PROMPT_PRESET_USER_APPEND: dict[str, str] = {
    "default": "",
    "douyin_riding": (
        "Short-video goal: find moments that can anchor a punchy 30-second Douyin/TikTok-style riding clip.\n"
        "Prefer clear hooks such as hard lean-in, rapid rhythm change, close pass, overtake, exit-to-scenery reveal, "
        "or obvious speed sensation.\n"
        "Write reason/discard_reason as concise Chinese snippets, ideally under 18 characters."
    ),
}


@dataclass
class FrameMetrics:
    blur_score: float
    frame_diff: float
    hash_distance: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze front-facing videos with a local Ollama vision model.")
    parser.add_argument("--input", help="Video file or folder. Defaults to config.project.input")
    parser.add_argument("--output", help="Output folder. Defaults to config.project.output")
    parser.add_argument("--config", default="config.toml", help="TOML config path")
    parser.add_argument("--extract-only", action="store_true", help="Skip Ollama and export frame candidates only")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def deep_merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict[str, Any]:
    config = DEFAULT_CONFIG
    if path.exists():
        with path.open("rb") as handle:
            loaded = tomllib.load(handle)
        config = deep_merge_dict(DEFAULT_CONFIG, loaded)
    return config


def resolve_paths(config: dict[str, Any], args: argparse.Namespace) -> tuple[Path, Path]:
    project = config["project"]
    input_path = Path(args.input or project["input"]).expanduser().resolve()
    output_path = Path(args.output or project["output"]).expanduser().resolve()
    return input_path, output_path


def list_videos(input_path: Path, config: dict[str, Any]) -> list[Path]:
    extensions = {ext.lower() for ext in config["project"].get("video_extensions", SUPPORTED_EXTENSIONS)}
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    videos = [path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() in extensions]
    return sorted(videos)


def variance_of_laplacian(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def average_hash(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    threshold = resized.mean()
    return (resized > threshold).astype(np.uint8)


def hamming_distance(first: np.ndarray | None, second: np.ndarray | None) -> int:
    if first is None or second is None:
        return 64
    return int(np.count_nonzero(first != second))


def frame_difference(current: np.ndarray, previous: np.ndarray | None) -> float:
    if previous is None:
        return 255.0
    current_small = cv2.resize(current, (32, 18), interpolation=cv2.INTER_AREA)
    previous_small = cv2.resize(previous, (32, 18), interpolation=cv2.INTER_AREA)
    gray_current = cv2.cvtColor(current_small, cv2.COLOR_BGR2GRAY)
    gray_previous = cv2.cvtColor(previous_small, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_current, gray_previous)
    return float(diff.mean())


def compute_metrics(frame: np.ndarray, previous_frame: np.ndarray | None, previous_hash: np.ndarray | None) -> tuple[FrameMetrics, np.ndarray]:
    current_hash = average_hash(frame)
    metrics = FrameMetrics(
        blur_score=variance_of_laplacian(frame),
        frame_diff=frame_difference(frame, previous_frame),
        hash_distance=hamming_distance(current_hash, previous_hash),
    )
    return metrics, current_hash


def should_send_to_model(
    metrics: FrameMetrics,
    timestamp_seconds: float,
    last_forced_timestamp: float | None,
    config: dict[str, Any],
) -> bool:
    filters = config["filters"]
    sampling = config["sampling"]

    if metrics.blur_score < float(filters["min_blur_score"]):
        return False

    forced_interval = float(sampling["forced_keep_interval_seconds"])
    forced_keep = last_forced_timestamp is None or (timestamp_seconds - last_forced_timestamp) >= forced_interval

    passes_change_gate = metrics.frame_diff >= float(filters["min_frame_diff"])
    passes_duplicate_gate = metrics.hash_distance >= int(filters["min_hash_distance"])
    return passes_duplicate_gate and (passes_change_gate or forced_keep)


def encode_frame(frame: np.ndarray, jpeg_quality: int) -> str:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def build_user_prompt(config: dict[str, Any], timestamp_seconds: float) -> str:
    prompt_config = config["prompt"]
    positive_labels = ", ".join(resolve_prompt_labels(prompt_config, "positive_labels", "extra_positive_labels"))
    negative_labels = ", ".join(resolve_prompt_labels(prompt_config, "negative_labels", "extra_negative_labels"))
    preset = normalize_prompt_preset(prompt_config.get("preset", "default"))
    preset_instructions = PROMPT_PRESET_USER_APPEND.get(preset, "")
    sections = [
        f"Timestamp: {timestamp_seconds:.2f} seconds.",
        "The image comes from a forward-fixed motorcycle riding video.",
        f"Positive labels: {positive_labels}.",
        f"Negative labels: {negative_labels}.",
    ]
    if preset_instructions:
        sections.append(preset_instructions)
    sections.append(
        "Return JSON with this exact schema:\n"
        '{'
        '"keep": true or false, '
        '"score": 0.0 to 1.0, '
        '"labels": ["label"], '
        '"reason": "short explanation", '
        '"discard_reason": "short explanation if keep is false"'
        '}\n'
        "Do not add markdown or commentary."
    )
    return "\n".join(sections)


def normalize_prompt_preset(value: Any) -> str:
    preset = str(value or "default").strip().lower()
    return preset if preset in PROMPT_PRESET_SYSTEM_APPEND else "default"


def resolve_prompt_labels(prompt_config: dict[str, Any], base_key: str, extra_key: str) -> list[str]:
    values: list[str] = []
    for raw_value in list(prompt_config.get(base_key, [])) + list(prompt_config.get(extra_key, [])):
        label = str(raw_value).strip()
        if label and label not in values:
            values.append(label)
    return values


def build_system_prompt(config: dict[str, Any]) -> str:
    prompt_config = config["prompt"]
    sections = [str(prompt_config.get("system_instructions", "")).strip()]
    preset = normalize_prompt_preset(prompt_config.get("preset", "default"))
    preset_append = PROMPT_PRESET_SYSTEM_APPEND.get(preset, "")
    if preset_append:
        sections.append(preset_append)
    extra_instructions = str(prompt_config.get("extra_instructions", "")).strip()
    if extra_instructions:
        sections.append(extra_instructions)
    return "\n\n".join(section for section in sections if section)


def call_ollama(image_base64: str, timestamp_seconds: float, config: dict[str, Any]) -> dict[str, Any]:
    ollama = config["ollama"]
    payload = {
        "model": ollama["model"],
        "stream": False,
        "options": {
            "temperature": float(ollama["temperature"]),
        },
        "messages": [
                {
                    "role": "system",
                    "content": build_system_prompt(config),
                },
            {
                "role": "user",
                "content": build_user_prompt(config, timestamp_seconds),
                "images": [image_base64],
            },
        ],
    }
    response = requests.post(
        f'{str(ollama["host"]).rstrip("/")}/api/chat',
        json=payload,
        timeout=float(ollama["timeout_seconds"]),
    )
    response.raise_for_status()
    message_content = response.json()["message"]["content"]
    return extract_json_block(message_content)


def extract_json_block(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Model response does not contain JSON: {text!r}")
    return json.loads(match.group(0))


def sanitize_decision(decision: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    max_reason_chars = int(config["decision"]["max_reason_chars"])
    labels = decision.get("labels") or []
    if isinstance(labels, str):
        labels = [labels]
    labels = [str(label).strip() for label in labels if str(label).strip()]

    keep = bool(decision.get("keep", False))
    score = float(decision.get("score", 0.0))
    if score > 1.0 and score <= 100.0:
        score /= 100.0

    reason_value = decision.get("reason")
    discard_reason_value = decision.get("discard_reason")
    reason = "" if reason_value is None else str(reason_value).strip()[:max_reason_chars]
    discard_reason = "" if discard_reason_value is None else str(discard_reason_value).strip()[:max_reason_chars]
    return {
        "keep": keep,
        "score": max(0.0, min(score, 1.0)),
        "labels": labels,
        "reason": reason,
        "discard_reason": discard_reason,
    }


def format_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_srt_text(segments: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        labels = ", ".join(segment["labels"]) or "highlight"
        line = f"{labels} | {segment['score']:.2f}"
        reason = segment.get("reason", "").strip()
        if reason:
            line = f"{line}\n{reason}"
        chunks.append(
            f"{index}\n"
            f"{format_timestamp(segment['start_seconds'])} --> {format_timestamp(segment['end_seconds'])}\n"
            f"{line}\n"
        )
    return "\n".join(chunks).strip() + ("\n" if chunks else "")


def save_thumbnail(frame: np.ndarray, output_dir: Path, timestamp_seconds: float) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{timestamp_seconds:09.3f}.jpg".replace(".", "_", 1)
    target = output_dir / filename
    cv2.imwrite(str(target), frame)
    return target


def merge_segments(frame_hits: list[dict[str, Any]], duration_seconds: float, config: dict[str, Any]) -> list[dict[str, Any]]:
    decision = config["decision"]
    min_keep_score = float(decision["min_keep_score"])
    merge_gap = float(decision["merge_gap_seconds"])
    padding_before = float(decision["padding_before_seconds"])
    padding_after = float(decision["padding_after_seconds"])
    min_segment_seconds = float(decision["min_segment_seconds"])

    kept = [hit for hit in frame_hits if hit["keep"] and hit["score"] >= min_keep_score]
    if not kept:
        return []

    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = [kept[0]]
    for hit in kept[1:]:
        if hit["timestamp_seconds"] - current[-1]["timestamp_seconds"] <= merge_gap:
            current.append(hit)
        else:
            segments.append(current)
            current = [hit]
    segments.append(current)

    merged: list[dict[str, Any]] = []
    for chunk in segments:
        labels = Counter(label for hit in chunk for label in hit["labels"])
        top_labels = [label for label, _ in labels.most_common(3)]
        reasons = [hit["reason"] for hit in chunk if hit["reason"]]
        start = max(0.0, chunk[0]["timestamp_seconds"] - padding_before)
        end = min(duration_seconds, chunk[-1]["timestamp_seconds"] + padding_after)
        if end - start < min_segment_seconds:
            end = min(duration_seconds, start + min_segment_seconds)
        merged.append(
            {
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "score": round(max(hit["score"] for hit in chunk), 3),
                "labels": top_labels,
                "reason": reasons[0] if reasons else "",
                "hit_count": len(chunk),
                "source_timestamps": [round(hit["timestamp_seconds"], 3) for hit in chunk],
            }
        )
    return merged


def analyze_video(video_path: Path, output_root: Path, config: dict[str, Any], extract_only: bool) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_seconds = frame_count / fps if fps > 0 else 0.0
    sample_fps = float(config["sampling"]["sample_fps"])
    frame_step = max(1, int(round(fps / sample_fps))) if sample_fps > 0 else 1
    max_frames = int(config["sampling"]["max_frames"])

    video_output = output_root / video_path.stem
    video_output.mkdir(parents=True, exist_ok=True)
    thumbnails_dir = video_output / str(config["project"]["thumbnail_dirname"])

    records: list[dict[str, Any]] = []
    previous_frame: np.ndarray | None = None
    previous_hash: np.ndarray | None = None
    last_forced_timestamp: float | None = None

    sample_indices = list(range(0, max(frame_count, 1), frame_step))
    if max_frames > 0:
        sample_indices = sample_indices[:max_frames]

    progress = tqdm(sample_indices, desc=video_path.name, unit="frame")
    try:
        for frame_number in progress:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = cap.read()
            if not ok:
                continue

            timestamp_seconds = frame_number / fps if fps > 0 else 0.0
            metrics, current_hash = compute_metrics(frame, previous_frame, previous_hash)
            candidate = should_send_to_model(metrics, timestamp_seconds, last_forced_timestamp, config)

            record: dict[str, Any] = {
                "timestamp_seconds": round(timestamp_seconds, 3),
                "timestamp_srt": format_timestamp(timestamp_seconds),
                "frame_number": frame_number,
                "candidate": candidate,
                "blur_score": round(metrics.blur_score, 3),
                "frame_diff": round(metrics.frame_diff, 3),
                "hash_distance": metrics.hash_distance,
                "keep": False,
                "score": 0.0,
                "labels": [],
                "reason": "",
                "discard_reason": "",
            }

            if candidate:
                last_forced_timestamp = timestamp_seconds
                if extract_only or not bool(config["ollama"]["enabled"]):
                    record["keep"] = True
                    record["score"] = 1.0
                    record["labels"] = ["candidate"]
                    record["reason"] = "Candidate frame exported without model inference."
                    record["thumbnail_path"] = str(save_thumbnail(frame, thumbnails_dir, timestamp_seconds))
                else:
                    try:
                        image_base64 = encode_frame(frame, int(config["sampling"]["jpeg_quality"]))
                        decision = sanitize_decision(call_ollama(image_base64, timestamp_seconds, config), config)
                        record.update(decision)
                        if record["keep"]:
                            record["thumbnail_path"] = str(save_thumbnail(frame, thumbnails_dir, timestamp_seconds))
                    except Exception as exc:  # noqa: BLE001
                        record["keep"] = False
                        record["discard_reason"] = f"ollama_error: {exc}"

            records.append(record)
            previous_frame = frame
            previous_hash = current_hash
    finally:
        cap.release()

    segments = merge_segments(records, duration_seconds, config)

    analysis_payload = {
        "video": {
            "source_path": str(video_path),
            "filename": video_path.name,
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": round(duration_seconds, 3),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "extract_only": extract_only,
        "config_snapshot": config,
        "frames": records,
        "segments": segments,
    }

    highlights_payload = {
        "video": analysis_payload["video"],
        "generated_at": analysis_payload["generated_at"],
        "segments": segments,
    }

    (video_output / "analysis.json").write_text(json.dumps(analysis_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (video_output / "highlights.json").write_text(json.dumps(highlights_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (video_output / "highlights.srt").write_text(build_srt_text(segments), encoding="utf-8")
    return analysis_payload


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    config = load_config(Path(args.config).expanduser())
    input_path, output_path = resolve_paths(config, args)
    output_path.mkdir(parents=True, exist_ok=True)
    videos = list_videos(input_path, config)

    if not videos:
        raise FileNotFoundError(f"No videos found in {input_path}")

    for video_path in videos:
        logging.info("Analyzing %s", video_path)
        analysis = analyze_video(video_path, output_path, config, args.extract_only)
        logging.info(
            "Finished %s | segments=%s | duration=%.1fs",
            video_path.name,
            len(analysis["segments"]),
            analysis["video"]["duration_seconds"],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
