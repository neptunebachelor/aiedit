from __future__ import annotations

import argparse
import base64
import copy
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import cv2
import requests
import tomllib
from tqdm import tqdm

from analyze_video import (
    DEFAULT_CONFIG as LEGACY_DEFAULT_CONFIG,
    build_user_prompt,
    compute_metrics,
    deep_merge_dict,
    extract_json_block,
    format_timestamp,
    list_videos,
    merge_segments,
    sanitize_decision,
    should_send_to_model,
)
from create_manual_plan import normalize_segment as normalize_editable_segment
from render_highlights import (
    build_candidates,
    build_export_segments,
    build_srt_text,
    find_ffmpeg,
    load_payload,
    normalize_prebuilt_segments,
    normalize_segment as normalize_review_segment,
    render_video,
    select_candidates,
)


DEFAULT_PIPELINE_CONFIG: dict[str, Any] = {
    "project": copy.deepcopy(LEGACY_DEFAULT_CONFIG["project"]),
    "extract": {
        "frame_interval_seconds": 1.0,
        "sample_fps": 1.0,
        "forced_keep_interval_seconds": 3.0,
        "jpeg_quality": 88,
        "max_frames": 0,
        "resize_for_llm": 1280,
    },
    "filters": copy.deepcopy(LEGACY_DEFAULT_CONFIG["filters"]),
    "provider": {
        "type": "ollama",
        "enabled": True,
        "temperature": 0.1,
        "timeout_seconds": 120,
        "ollama": {
            "api_base": "http://127.0.0.1:11434",
            "model": "qwen3-vl:8b",
        },
        "openai_compatible": {
            "api_base": "",
            "model": "",
            "api_key": "",
            "api_key_env": "OPENAI_API_KEY",
        },
    },
    "prompt": copy.deepcopy(LEGACY_DEFAULT_CONFIG["prompt"]),
    "selection": copy.deepcopy(LEGACY_DEFAULT_CONFIG["decision"]),
    "review": {
        "target_seconds": 30.0,
        "clip_before": 1.0,
        "clip_after": 2.0,
        "cluster_gap": 2.0,
        "min_clip_seconds": 2.0,
        "max_clip_seconds": 6.0,
        "max_clips_per_source_segment": 2,
        "caption_mode": "human",
        "emit_source_srt": True,
        "emit_final_srt": True,
    },
    "preview": {
        "enabled": False,
        "resolution": "720p",
        "crf": 28,
        "preset": "veryfast",
    },
    "render": {
        "resolution": "source",
        "crf": 18,
        "preset": "fast",
    },
}

HUMAN_LABELS = {
    "bend": "bend",
    "scenery": "scenic section",
    "traffic": "traffic",
    "overtake": "overtake",
    "group_ride": "group riding",
    "tunnel_transition": "tunnel transition",
    "water_view": "water view",
    "mountain_view": "mountain view",
    "sunset": "sunset light",
    "corner_entry": "corner entry",
    "apex": "apex",
    "corner_exit": "corner exit",
    "full_throttle": "full throttle",
    "high_speed": "high speed",
    "late_braking": "late braking",
    "handlebar_wobble": "handlebar wobble",
    "near_barrier": "near barrier",
    "pit_lane": "pit lane",
    "cooldown": "cooldown",
    "formation_lap": "formation lap",
    "slow_straight": "slow straight",
    "low_value_track": "low-value track section",
}

RESOLUTION_HEIGHTS = {
    "540p": 540,
    "720p": 720,
    "1080p": 1080,
    "source": 0,
}


@dataclass
class VideoMeta:
    source_path: str
    filename: str
    fps: float
    frame_count: int
    duration_seconds: float
    width: int
    height: int


class VisionProvider(Protocol):
    def infer(self, image_bytes: bytes, *, timestamp_seconds: float, config: dict[str, Any]) -> dict[str, Any]:
        ...


class OllamaVisionProvider:
    def __init__(self, *, api_base: str, model: str, temperature: float, timeout_seconds: float) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    def infer(self, image_bytes: bytes, *, timestamp_seconds: float, config: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": self.temperature},
            "messages": [
                {
                    "role": "system",
                    "content": config["prompt"]["system_instructions"],
                },
                {
                    "role": "user",
                    "content": build_user_prompt(config, timestamp_seconds),
                    "images": [base64.b64encode(image_bytes).decode("ascii")],
                },
            ],
        }
        response = requests.post(
            f"{self.api_base}/api/chat",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        message_content = response.json()["message"]["content"]
        return extract_json_block(message_content)


class OpenAICompatibleVisionProvider:
    def __init__(
        self,
        *,
        api_base: str,
        model: str,
        api_key: str,
        temperature: float,
        timeout_seconds: float,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    def infer(self, image_bytes: bytes, *, timestamp_seconds: float, config: dict[str, Any]) -> dict[str, Any]:
        data_url = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('ascii')}"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": config["prompt"]["system_instructions"],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_user_prompt(config, timestamp_seconds)},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        }
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        message_content = response.json()["choices"][0]["message"]["content"]
        if isinstance(message_content, list):
            message_content = "\n".join(
                item.get("text", "") for item in message_content if isinstance(item, dict) and item.get("type") == "text"
            )
        return extract_json_block(str(message_content))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-platform video highlight pipeline with extract / infer / review / render stages."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Sample candidate frames from video(s).")
    add_common_video_args(extract_parser)
    extract_parser.add_argument("--frame-interval-seconds", type=float, help="Sample one frame every N seconds.")
    extract_parser.add_argument("--sample-fps", type=float, help="Alternative to frame interval. Sample N frames per second.")
    extract_parser.add_argument("--max-frames", type=int, help="Limit the number of sampled frames.")
    extract_parser.add_argument("--jpeg-quality", type=int, help="JPEG quality used when writing extracted frames.")
    extract_parser.add_argument("--resize-for-llm", type=int, help="Maximum output dimension for extracted frames.")
    extract_parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    infer_parser = subparsers.add_parser("infer", help="Run vision-model inference over extracted frames.")
    add_common_video_args(infer_parser)
    infer_parser.add_argument("--extract-index", help="Path to extract/index.json. If missing, the pipeline will generate it.")
    infer_parser.add_argument("--provider-type", choices=["ollama", "openai_compatible"])
    infer_parser.add_argument("--api-base", help="Provider API base URL.")
    infer_parser.add_argument("--api-key", help="API key for third-party providers.")
    infer_parser.add_argument("--api-key-env", help="Environment variable used to read the API key.")
    infer_parser.add_argument("--model", help="Vision model name.")
    infer_parser.add_argument("--temperature", type=float, help="Sampling temperature for the provider.")
    infer_parser.add_argument("--timeout-seconds", type=float, help="Provider timeout in seconds.")
    infer_parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    review_parser = subparsers.add_parser("review", help="Build a reviewable edit plan and optional preview.")
    review_parser.add_argument("--input", required=True, help="Path to analysis.json, segments.raw.json, or editable JSON.")
    review_parser.add_argument("--output-dir", help="Destination directory for review outputs.")
    review_parser.add_argument("--config", default="config.toml", help="Optional TOML config path.")
    review_parser.add_argument("--target-seconds", type=float, help="Target total duration for the compact plan.")
    review_parser.add_argument("--clip-before", type=float, help="Seconds kept before each highlight anchor.")
    review_parser.add_argument("--clip-after", type=float, help="Seconds kept after each highlight anchor.")
    review_parser.add_argument("--cluster-gap", type=float, help="Maximum gap between timestamps before split.")
    review_parser.add_argument("--min-clip-seconds", type=float, help="Minimum duration for selected clips.")
    review_parser.add_argument("--max-clip-seconds", type=float, help="Maximum duration for selected clips.")
    review_parser.add_argument("--max-clips-per-source-segment", type=int, help="Diversity limit per source segment.")
    review_parser.add_argument("--stem", help="Output stem. Defaults to highlights_<target>s.")
    review_parser.add_argument(
        "--caption-mode",
        choices=["score", "reason", "human"],
        help="How the review SRT should be written.",
    )
    review_parser.add_argument("--preview", action="store_true", help="Render a low-resolution preview MP4.")
    review_parser.add_argument(
        "--preview-resolution",
        choices=sorted(RESOLUTION_HEIGHTS.keys()),
        help="Preview resolution. One of 540p, 720p, 1080p, source.",
    )
    review_parser.add_argument("--preview-crf", type=int, help="CRF used for preview rendering.")
    review_parser.add_argument("--preview-preset", help="Preset used for preview rendering.")
    review_parser.add_argument("--ffmpeg", help="Optional path to ffmpeg executable.")
    review_parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    render_parser = subparsers.add_parser("render", help="Render a final video from a reviewed plan or raw analysis.")
    render_parser.add_argument("--input", required=True, help="Path to review JSON, editable JSON, or analysis.json.")
    render_parser.add_argument("--output-dir", help="Destination directory for rendered outputs.")
    render_parser.add_argument("--config", default="config.toml", help="Optional TOML config path.")
    render_parser.add_argument("--target-seconds", type=float, help="Target duration when rendering from raw analysis.")
    render_parser.add_argument("--stem", help="Output stem. Defaults to the input stem or highlights_<target>s.")
    render_parser.add_argument(
        "--caption-mode",
        choices=["score", "reason", "human"],
        help="Caption mode used when the input does not already contain captions.",
    )
    render_parser.add_argument(
        "--resolution",
        choices=sorted(RESOLUTION_HEIGHTS.keys()),
        help="Render resolution. One of 540p, 720p, 1080p, source.",
    )
    render_parser.add_argument("--crf", type=int, help="CRF used when encoding intermediate clips.")
    render_parser.add_argument("--preset", help="Preset used when encoding intermediate clips.")
    render_parser.add_argument("--ffmpeg", help="Optional path to ffmpeg executable.")
    render_parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    edit_parser = subparsers.add_parser("edit", help="Patch an editable plan without opening the JSON manually.")
    edit_subparsers = edit_parser.add_subparsers(dest="edit_command", required=True)

    update_segment_parser = edit_subparsers.add_parser("update-segment", help="Update source cut points for one segment.")
    update_segment_parser.add_argument("--plan", required=True, help="Path to editable plan JSON.")
    update_segment_parser.add_argument("--rank", required=True, type=int, help="Segment rank to update.")
    update_segment_parser.add_argument("--source-start-seconds", type=float, help="New source start time.")
    update_segment_parser.add_argument("--source-end-seconds", type=float, help="New source end time.")
    update_segment_parser.add_argument("--output", help="Optional destination path. Defaults to the original file.")

    update_caption_parser = edit_subparsers.add_parser("update-caption", help="Update caption text for one segment.")
    update_caption_parser.add_argument("--plan", required=True, help="Path to editable plan JSON.")
    update_caption_parser.add_argument("--rank", required=True, type=int, help="Segment rank to update.")
    update_caption_parser.add_argument("--caption", required=True, help="New caption text.")
    update_caption_parser.add_argument("--caption-detail", help="Optional second subtitle line.")
    update_caption_parser.add_argument("--output", help="Optional destination path. Defaults to the original file.")

    return parser.parse_args()


def add_common_video_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video", required=True, help="Video file or folder.")
    parser.add_argument("--output-root", help="Root directory for pipeline outputs.")
    parser.add_argument("--config", default="config.toml", help="Optional TOML config path.")


def load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_pipeline_config(path: Path) -> dict[str, Any]:
    raw = load_toml(path)
    legacy = deep_merge_dict(copy.deepcopy(LEGACY_DEFAULT_CONFIG), raw)

    config = copy.deepcopy(DEFAULT_PIPELINE_CONFIG)
    config["project"] = deep_merge_dict(config["project"], legacy.get("project", {}))
    config["filters"] = deep_merge_dict(config["filters"], legacy.get("filters", {}))
    config["prompt"] = deep_merge_dict(config["prompt"], legacy.get("prompt", {}))
    config["selection"] = deep_merge_dict(config["selection"], legacy.get("decision", {}))
    config["selection"] = deep_merge_dict(config["selection"], raw.get("selection", {}))

    sampling = legacy.get("sampling", {})
    extract = deep_merge_dict(
        config["extract"],
        {
            "forced_keep_interval_seconds": sampling.get(
                "forced_keep_interval_seconds",
                config["extract"]["forced_keep_interval_seconds"],
            ),
            "jpeg_quality": sampling.get("jpeg_quality", config["extract"]["jpeg_quality"]),
            "max_frames": sampling.get("max_frames", config["extract"]["max_frames"]),
        },
    )
    sample_fps = float(sampling.get("sample_fps", 0.0) or 0.0)
    if sample_fps > 0:
        extract["sample_fps"] = sample_fps
        extract["frame_interval_seconds"] = round(1.0 / sample_fps, 3)
    extract = deep_merge_dict(extract, raw.get("extract", {}))
    config["extract"] = finalize_extract_settings(extract)

    provider = copy.deepcopy(config["provider"])
    legacy_ollama = legacy.get("ollama", {})
    provider["enabled"] = bool(legacy_ollama.get("enabled", provider["enabled"]))
    provider["temperature"] = float(legacy_ollama.get("temperature", provider["temperature"]))
    provider["timeout_seconds"] = float(legacy_ollama.get("timeout_seconds", provider["timeout_seconds"]))
    provider["ollama"] = deep_merge_dict(
        provider["ollama"],
        {
            "api_base": legacy_ollama.get("host", provider["ollama"]["api_base"]),
            "model": legacy_ollama.get("model", provider["ollama"]["model"]),
        },
    )
    provider = deep_merge_dict(provider, raw.get("provider", {}))
    config["provider"] = provider

    config["review"] = deep_merge_dict(config["review"], raw.get("review", {}))
    config["preview"] = deep_merge_dict(config["preview"], raw.get("preview", {}))
    config["render"] = deep_merge_dict(config["render"], raw.get("render", {}))
    return config


def finalize_extract_settings(settings: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(settings)
    frame_interval_seconds = float(resolved.get("frame_interval_seconds", 0.0) or 0.0)
    sample_fps = float(resolved.get("sample_fps", 0.0) or 0.0)
    if frame_interval_seconds > 0:
        resolved["frame_interval_seconds"] = frame_interval_seconds
        resolved["sample_fps"] = round(1.0 / frame_interval_seconds, 3)
    elif sample_fps > 0:
        resolved["sample_fps"] = sample_fps
        resolved["frame_interval_seconds"] = round(1.0 / sample_fps, 3)
    else:
        resolved["frame_interval_seconds"] = 1.0
        resolved["sample_fps"] = 1.0
    resolved["jpeg_quality"] = int(resolved.get("jpeg_quality", 88))
    resolved["max_frames"] = int(resolved.get("max_frames", 0))
    resolved["resize_for_llm"] = int(resolved.get("resize_for_llm", 0))
    resolved["forced_keep_interval_seconds"] = float(resolved.get("forced_keep_interval_seconds", 3.0))
    return resolved


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(levelname)s: %(message)s")


def resolve_output_root(config: dict[str, Any], override: str | None) -> Path:
    return Path(override or config["project"]["output"]).expanduser().resolve()


def probe_video(video_path: Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    duration_seconds = frame_count / fps if fps > 0 else 0.0
    return VideoMeta(
        source_path=str(video_path.resolve()),
        filename=video_path.name,
        fps=float(fps),
        frame_count=frame_count,
        duration_seconds=round(duration_seconds, 3),
        width=width,
        height=height,
    )


def resize_frame(frame: Any, max_dimension: int) -> Any:
    if max_dimension <= 0:
        return frame
    height, width = frame.shape[:2]
    longest = max(width, height)
    if longest <= max_dimension:
        return frame
    scale = max_dimension / float(longest)
    target_width = max(2, int(round(width * scale)))
    target_height = max(2, int(round(height * scale)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def write_frame_image(frame: Any, destination: Path, jpeg_quality: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError(f"Failed to encode frame for {destination}")
    destination.write_bytes(encoded.tobytes())


def frame_filename(timestamp_seconds: float) -> str:
    return f"{timestamp_seconds:09.3f}.jpg".replace(".", "_", 1)


def extract_candidates_for_video(
    video_path: Path,
    *,
    output_root: Path,
    config: dict[str, Any],
    frame_interval_seconds: float,
    max_frames: int,
    jpeg_quality: int,
    resize_for_llm: int,
) -> Path:
    video_meta = probe_video(video_path)
    video_output_dir = output_root / video_path.stem
    extract_dir = video_output_dir / "extract"
    frames_dir = extract_dir / "frames"
    extract_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_step = max(1, int(round(video_meta.fps * frame_interval_seconds)))
    sample_indices = list(range(0, max(video_meta.frame_count, 1), frame_step))
    if max_frames > 0:
        sample_indices = sample_indices[:max_frames]

    gate_config = {
        "filters": config["filters"],
        "sampling": {
            "forced_keep_interval_seconds": float(config["extract"]["forced_keep_interval_seconds"]),
        },
    }

    records: list[dict[str, Any]] = []
    previous_frame: Any | None = None
    previous_hash: Any | None = None
    last_forced_timestamp: float | None = None

    progress = tqdm(sample_indices, desc=f"extract:{video_path.name}", unit="frame")
    try:
        for frame_number in progress:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = cap.read()
            if not ok:
                continue

            timestamp_seconds = frame_number / video_meta.fps if video_meta.fps > 0 else 0.0
            metrics, current_hash = compute_metrics(frame, previous_frame, previous_hash)
            candidate = should_send_to_model(metrics, timestamp_seconds, last_forced_timestamp, gate_config)

            record: dict[str, Any] = {
                "timestamp_seconds": round(timestamp_seconds, 3),
                "timestamp_srt": format_timestamp(timestamp_seconds),
                "frame_number": frame_number,
                "candidate": candidate,
                "blur_score": round(metrics.blur_score, 3),
                "frame_diff": round(metrics.frame_diff, 3),
                "hash_distance": int(metrics.hash_distance),
                "image_path": "",
                "image_width": 0,
                "image_height": 0,
            }

            if candidate:
                last_forced_timestamp = timestamp_seconds
                prepared_frame = resize_frame(frame, resize_for_llm)
                image_path = frames_dir / frame_filename(timestamp_seconds)
                write_frame_image(prepared_frame, image_path, jpeg_quality)
                record["image_path"] = str(image_path.resolve())
                record["image_width"] = int(prepared_frame.shape[1])
                record["image_height"] = int(prepared_frame.shape[0])

            records.append(record)
            previous_frame = frame
            previous_hash = current_hash
    finally:
        cap.release()

    payload = {
        "stage": "extract",
        "video": video_meta.__dict__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_snapshot": {
            "extract": {
                "frame_interval_seconds": round(frame_interval_seconds, 3),
                "sample_fps": round(1.0 / frame_interval_seconds, 3),
                "forced_keep_interval_seconds": float(config["extract"]["forced_keep_interval_seconds"]),
                "jpeg_quality": jpeg_quality,
                "max_frames": max_frames,
                "resize_for_llm": resize_for_llm,
            },
            "filters": config["filters"],
        },
        "frames": records,
        "sampled_frames": len(records),
        "candidate_frames": sum(1 for record in records if record["candidate"]),
    }

    index_path = extract_dir / "index.json"
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Wrote %s", index_path)
    return index_path


def resolve_provider_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    provider = copy.deepcopy(config["provider"])
    if getattr(args, "provider_type", None):
        provider["type"] = args.provider_type
    if getattr(args, "temperature", None) is not None:
        provider["temperature"] = float(args.temperature)
    if getattr(args, "timeout_seconds", None) is not None:
        provider["timeout_seconds"] = float(args.timeout_seconds)

    provider_type = provider["type"]
    if provider_type == "ollama":
        if getattr(args, "api_base", None):
            provider["ollama"]["api_base"] = args.api_base
        if getattr(args, "model", None):
            provider["ollama"]["model"] = args.model
    elif provider_type == "openai_compatible":
        if getattr(args, "api_base", None):
            provider["openai_compatible"]["api_base"] = args.api_base
        if getattr(args, "model", None):
            provider["openai_compatible"]["model"] = args.model
        if getattr(args, "api_key", None):
            provider["openai_compatible"]["api_key"] = args.api_key
        if getattr(args, "api_key_env", None):
            provider["openai_compatible"]["api_key_env"] = args.api_key_env
    return provider


def build_provider(provider_config: dict[str, Any]) -> VisionProvider:
    if not bool(provider_config.get("enabled", True)):
        raise ValueError("Provider is disabled in configuration.")

    provider_type = str(provider_config.get("type", "ollama")).strip()
    temperature = float(provider_config.get("temperature", 0.1))
    timeout_seconds = float(provider_config.get("timeout_seconds", 120))

    if provider_type == "ollama":
        ollama_config = provider_config["ollama"]
        return OllamaVisionProvider(
            api_base=str(ollama_config["api_base"]).strip(),
            model=str(ollama_config["model"]).strip(),
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )

    if provider_type == "openai_compatible":
        compatible_config = provider_config["openai_compatible"]
        api_key = str(compatible_config.get("api_key", "")).strip()
        api_key_env = str(compatible_config.get("api_key_env", "")).strip()
        if not api_key and api_key_env:
            api_key = str(os.environ.get(api_key_env, "")).strip()
        if not api_key:
            raise ValueError("No API key configured for openai_compatible provider.")
        return OpenAICompatibleVisionProvider(
            api_base=str(compatible_config["api_base"]).strip(),
            model=str(compatible_config["model"]).strip(),
            api_key=api_key,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )

    raise ValueError(f"Unsupported provider type: {provider_type}")


def ensure_extract_index(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    if getattr(args, "extract_index", None):
        return Path(args.extract_index).expanduser().resolve()

    video_path = Path(args.video).expanduser().resolve()
    output_root = resolve_output_root(config, getattr(args, "output_root", None))
    candidate_index = output_root / video_path.stem / "extract" / "index.json"
    if candidate_index.exists():
        return candidate_index

    logging.info("Extract index missing for %s, running extract stage first.", video_path)
    extract_settings = config["extract"]
    return extract_candidates_for_video(
        video_path,
        output_root=output_root,
        config=config,
        frame_interval_seconds=float(extract_settings["frame_interval_seconds"]),
        max_frames=int(extract_settings["max_frames"]),
        jpeg_quality=int(extract_settings["jpeg_quality"]),
        resize_for_llm=int(extract_settings["resize_for_llm"]),
    )


def infer_from_extract_index(
    index_path: Path,
    *,
    provider: VisionProvider,
    provider_snapshot: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    video_info = dict(payload["video"])
    video_output_dir = index_path.parent.parent

    records: list[dict[str, Any]] = []
    candidate_frames = [frame for frame in payload["frames"] if frame.get("candidate") and frame.get("image_path")]
    progress = tqdm(candidate_frames, desc=f"infer:{video_info['filename']}", unit="frame")

    decisions_by_frame_number: dict[int, dict[str, Any]] = {}
    for frame in progress:
        image_path = Path(frame["image_path"])
        image_bytes = image_path.read_bytes()
        try:
            decision = sanitize_decision(
                provider.infer(image_bytes, timestamp_seconds=float(frame["timestamp_seconds"]), config=config),
                {"decision": config["selection"]},
            )
        except Exception as exc:  # noqa: BLE001
            decision = {
                "keep": False,
                "score": 0.0,
                "labels": [],
                "reason": "",
                "discard_reason": f"provider_error: {exc}",
            }
        decisions_by_frame_number[int(frame["frame_number"])] = decision

    for frame in payload["frames"]:
        decision = decisions_by_frame_number.get(int(frame["frame_number"]), None)
        record = {
            "timestamp_seconds": float(frame["timestamp_seconds"]),
            "timestamp_srt": frame["timestamp_srt"],
            "frame_number": int(frame["frame_number"]),
            "candidate": bool(frame["candidate"]),
            "blur_score": float(frame["blur_score"]),
            "frame_diff": float(frame["frame_diff"]),
            "hash_distance": int(frame["hash_distance"]),
            "image_path": frame.get("image_path", ""),
            "keep": False,
            "score": 0.0,
            "labels": [],
            "reason": "",
            "discard_reason": "",
        }
        if decision:
            record.update(decision)
            if record["keep"] and record["image_path"]:
                record["thumbnail_path"] = record["image_path"]
        records.append(record)

    segments = merge_segments(
        records,
        float(video_info["duration_seconds"]),
        {"decision": config["selection"]},
    )

    analysis_payload = {
        "stage": "infer",
        "video": video_info,
        "source_extract_index": str(index_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider_snapshot": provider_snapshot,
        "config_snapshot": {
            "prompt": config["prompt"],
            "selection": config["selection"],
        },
        "frames": records,
        "segments": segments,
    }
    raw_segments_payload = {
        "stage": "segments.raw",
        "video": video_info,
        "source_analysis": str((video_output_dir / "analysis.json").resolve()),
        "generated_at": analysis_payload["generated_at"],
        "segments": segments,
    }

    analysis_path = video_output_dir / "analysis.json"
    segments_path = video_output_dir / "segments.raw.json"
    segments_srt_path = video_output_dir / "segments.raw.srt"
    highlights_json_path = video_output_dir / "highlights.json"
    highlights_srt_path = video_output_dir / "highlights.srt"
    decisions_jsonl_path = video_output_dir / "frame_decisions.jsonl"

    analysis_path.write_text(json.dumps(analysis_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    segments_path.write_text(json.dumps(raw_segments_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    segments_srt_path.write_text(build_srt_text(segments), encoding="utf-8")
    highlights_json_path.write_text(json.dumps(raw_segments_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    highlights_srt_path.write_text(build_srt_text(segments), encoding="utf-8")
    with decisions_jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    logging.info("Wrote %s", analysis_path)
    logging.info("Wrote %s", segments_path)
    logging.info("Wrote %s", segments_srt_path)
    logging.info("Wrote %s", decisions_jsonl_path)
    return analysis_path


def resolve_review_settings(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    review = copy.deepcopy(config["review"])
    if getattr(args, "target_seconds", None) is not None:
        review["target_seconds"] = float(args.target_seconds)
    if getattr(args, "clip_before", None) is not None:
        review["clip_before"] = float(args.clip_before)
    if getattr(args, "clip_after", None) is not None:
        review["clip_after"] = float(args.clip_after)
    if getattr(args, "cluster_gap", None) is not None:
        review["cluster_gap"] = float(args.cluster_gap)
    if getattr(args, "min_clip_seconds", None) is not None:
        review["min_clip_seconds"] = float(args.min_clip_seconds)
    if getattr(args, "max_clip_seconds", None) is not None:
        review["max_clip_seconds"] = float(args.max_clip_seconds)
    if getattr(args, "max_clips_per_source_segment", None) is not None:
        review["max_clips_per_source_segment"] = int(args.max_clips_per_source_segment)
    if getattr(args, "caption_mode", None):
        review["caption_mode"] = args.caption_mode
    return review


def default_stem(target_seconds: float) -> str:
    if target_seconds == round(target_seconds):
        return f"highlights_{int(round(target_seconds))}s"
    return f"highlights_{str(target_seconds).replace('.', '_')}s"


def humanize_labels(labels: list[str]) -> list[str]:
    return [HUMAN_LABELS.get(label, label.replace("_", " ")) for label in labels]


def sentence_case(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    return cleaned[0].upper() + cleaned[1:]


def auto_caption_for_segment(segment: dict[str, Any], caption_mode: str) -> tuple[str, str]:
    reason = sentence_case(str(segment.get("reason", "")).strip())
    human_labels = humanize_labels([str(label) for label in segment.get("labels", []) if str(label).strip()])
    label_text = ", ".join(human_labels)

    if caption_mode == "score":
        return "", ""
    if caption_mode == "reason":
        if reason:
            return reason, ""
        if label_text:
            return label_text, ""
        return "Highlight segment", ""

    if reason:
        detail = f"Tags: {label_text}" if label_text else ""
        return reason, detail
    if label_text:
        return f"Highlight: {label_text}", ""
    return "Highlight segment", ""


def apply_caption_mode(segments: list[dict[str, Any]], caption_mode: str) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for segment in segments:
        copied = dict(segment)
        existing_caption = str(copied.get("caption", "")).strip()
        existing_detail = str(copied.get("caption_detail", "")).strip()
        if not existing_caption:
            caption, detail = auto_caption_for_segment(copied, caption_mode)
            if caption:
                copied["caption"] = caption
            if detail:
                copied["caption_detail"] = detail
        elif existing_detail:
            copied["caption_detail"] = existing_detail
        updated.append(copied)
    return updated


def build_review_segments(payload: dict[str, Any], review_settings: dict[str, Any]) -> list[dict[str, Any]]:
    raw_segments = list(payload["segments"])
    if raw_segments and "source_start_seconds" in raw_segments[0] and "source_end_seconds" in raw_segments[0]:
        segments = normalize_prebuilt_segments(raw_segments)
    else:
        normalized = [normalize_review_segment(segment, index) for index, segment in enumerate(raw_segments)]
        candidates = build_candidates(
            normalized,
            clip_before=float(review_settings["clip_before"]),
            clip_after=float(review_settings["clip_after"]),
            cluster_gap=float(review_settings["cluster_gap"]),
            min_clip_seconds=float(review_settings["min_clip_seconds"]),
            max_clip_seconds=float(review_settings["max_clip_seconds"]),
        )
        selected = select_candidates(
            candidates,
            target_seconds=float(review_settings["target_seconds"]),
            min_clip_seconds=float(review_settings["min_clip_seconds"]),
            max_clips_per_source_segment=int(review_settings["max_clips_per_source_segment"]),
        )
        segments = build_export_segments(selected)
    return apply_caption_mode(segments, str(review_settings["caption_mode"]))


def parse_resolution(resolution: str | None, source_height: int) -> int | None:
    if not resolution:
        return None
    key = str(resolution).lower()
    target_height = RESOLUTION_HEIGHTS.get(key)
    if target_height is None or target_height <= 0:
        return None
    return min(int(source_height), int(target_height)) if source_height > 0 else int(target_height)


def write_review_outputs(
    *,
    payload: dict[str, Any],
    input_path: Path,
    output_dir: Path,
    stem: str,
    review_settings: dict[str, Any],
    segments: list[dict[str, Any]],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    actual_seconds = round(sum(float(segment["duration_seconds"]) for segment in segments), 3)

    review_payload = {
        "stage": "review",
        "video": payload["video"],
        "source_input": str(input_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_seconds": round(float(review_settings["target_seconds"]), 3),
        "actual_seconds": actual_seconds,
        "caption_mode": str(review_settings["caption_mode"]),
        "segments": segments,
    }
    editable_payload = {
        "video": payload["video"],
        "source_input": str(input_path),
        "generated_at": review_payload["generated_at"],
        "target_seconds": actual_seconds,
        "mode": "editable_review_plan",
        "segments": [normalize_editable_segment(segment, index) for index, segment in enumerate(segments, start=1)],
    }

    review_json_path = output_dir / f"{stem}.review.json"
    editable_json_path = output_dir / f"{stem}.editable.json"
    final_srt_path = output_dir / f"{stem}.final.srt"
    source_srt_path = output_dir / f"{stem}.source.srt"

    review_json_path.write_text(json.dumps(review_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    editable_json_path.write_text(json.dumps(editable_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    final_srt_path.write_text(build_srt_text(segments), encoding="utf-8")
    source_srt_path.write_text(
        build_srt_text(segments, start_key="source_start_seconds", end_key="source_end_seconds"),
        encoding="utf-8",
    )

    return {
        "review_json": review_json_path,
        "editable_json": editable_json_path,
        "final_srt": final_srt_path,
        "source_srt": source_srt_path,
    }


def render_preview_if_requested(
    *,
    requested: bool,
    preview_settings: dict[str, Any],
    ffmpeg_override: str | None,
    payload: dict[str, Any],
    output_dir: Path,
    stem: str,
    segments: list[dict[str, Any]],
) -> Path | None:
    if not requested:
        return None
    source_path_text = str(payload["video"].get("source_path", "")).strip()
    if not source_path_text:
        raise ValueError("Preview rendering requires a source video path.")
    source_path = Path(source_path_text).expanduser().resolve()
    video_meta = probe_video(source_path)
    preview_height = parse_resolution(str(preview_settings["resolution"]), video_meta.height)
    ffmpeg_path = find_ffmpeg(ffmpeg_override)
    preview_path = output_dir / f"{stem}.preview.mp4"
    render_video(
        ffmpeg_path=ffmpeg_path,
        source_video=source_path,
        output_video=preview_path,
        clips=segments,
        preset=str(preview_settings["preset"]),
        crf=int(preview_settings["crf"]),
        scale_height=preview_height,
    )
    return preview_path


def infer_render_stem(input_path: Path, target_seconds: float | None) -> str:
    stem = input_path.stem
    if stem.endswith(".review"):
        return stem[: -len(".review")]
    if stem.endswith(".editable"):
        return stem[: -len(".editable")] + "_final"
    if target_seconds is not None:
        return default_stem(target_seconds)
    return stem


def resolve_render_settings(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    render_settings = copy.deepcopy(config["render"])
    if getattr(args, "resolution", None):
        render_settings["resolution"] = args.resolution
    if getattr(args, "crf", None) is not None:
        render_settings["crf"] = int(args.crf)
    if getattr(args, "preset", None):
        render_settings["preset"] = args.preset
    return render_settings


def write_render_outputs(
    *,
    payload: dict[str, Any],
    input_path: Path,
    output_dir: Path,
    stem: str,
    segments: list[dict[str, Any]],
    render_settings: dict[str, Any],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    actual_seconds = round(sum(float(segment["duration_seconds"]) for segment in segments), 3)
    output_payload = {
        "stage": "render_plan",
        "video": payload["video"],
        "source_input": str(input_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "actual_seconds": actual_seconds,
        "render": render_settings,
        "segments": segments,
    }

    json_path = output_dir / f"{stem}.json"
    final_srt_path = output_dir / f"{stem}.final.srt"
    source_srt_path = output_dir / f"{stem}.source.srt"

    json_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    final_srt_path.write_text(build_srt_text(segments), encoding="utf-8")
    source_srt_path.write_text(
        build_srt_text(segments, start_key="source_start_seconds", end_key="source_end_seconds"),
        encoding="utf-8",
    )
    return json_path, final_srt_path, source_srt_path


def load_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "segments" not in payload:
        raise ValueError(f"{path} does not contain a 'segments' field.")
    return payload


def rebase_editable_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cursor = 0.0
    updated: list[dict[str, Any]] = []
    for index, segment in enumerate(segments, start=1):
        copied = dict(segment)
        source_start = round(float(copied["source_start_seconds"]), 3)
        source_end = round(float(copied["source_end_seconds"]), 3)
        if source_end <= source_start:
            raise ValueError(f"Invalid source range for segment {index}: {source_start} -> {source_end}")
        duration = round(source_end - source_start, 3)
        copied["rank"] = index
        copied["duration_seconds"] = duration
        copied["start_seconds"] = round(cursor, 3)
        copied["end_seconds"] = round(cursor + duration, 3)
        cursor = copied["end_seconds"]
        updated.append(copied)
    return updated


def save_plan(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Wrote %s", path)
    return path


def command_extract(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)
    config = load_pipeline_config(Path(args.config).expanduser())
    input_path = Path(args.video).expanduser().resolve()
    output_root = resolve_output_root(config, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    extract_settings = copy.deepcopy(config["extract"])
    if args.frame_interval_seconds is not None:
        extract_settings["frame_interval_seconds"] = float(args.frame_interval_seconds)
        extract_settings["sample_fps"] = 0.0
    if args.sample_fps is not None:
        extract_settings["sample_fps"] = float(args.sample_fps)
        extract_settings["frame_interval_seconds"] = 0.0
    if args.max_frames is not None:
        extract_settings["max_frames"] = int(args.max_frames)
    if args.jpeg_quality is not None:
        extract_settings["jpeg_quality"] = int(args.jpeg_quality)
    if args.resize_for_llm is not None:
        extract_settings["resize_for_llm"] = int(args.resize_for_llm)
    extract_settings = finalize_extract_settings(extract_settings)
    config["extract"] = extract_settings

    videos = list_videos(input_path, config)
    if not videos:
        raise FileNotFoundError(f"No videos found in {input_path}")

    for video_path in videos:
        extract_candidates_for_video(
            video_path,
            output_root=output_root,
            config=config,
            frame_interval_seconds=float(extract_settings["frame_interval_seconds"]),
            max_frames=int(extract_settings["max_frames"]),
            jpeg_quality=int(extract_settings["jpeg_quality"]),
            resize_for_llm=int(extract_settings["resize_for_llm"]),
        )
    return 0


def command_infer(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)
    config = load_pipeline_config(Path(args.config).expanduser())
    provider_config = resolve_provider_config(config, args)
    provider = build_provider(provider_config)
    if args.extract_index:
        index_paths = [Path(args.extract_index).expanduser().resolve()]
    else:
        input_path = Path(args.video).expanduser().resolve()
        if input_path.is_dir():
            videos = list_videos(input_path, config)
            if not videos:
                raise FileNotFoundError(f"No videos found in {input_path}")
            index_paths = []
            for video_path in videos:
                video_args = argparse.Namespace(**vars(args))
                video_args.video = str(video_path)
                index_paths.append(ensure_extract_index(video_args, config))
        else:
            index_paths = [ensure_extract_index(args, config)]

    for index_path in index_paths:
        infer_from_extract_index(
            index_path,
            provider=provider,
            provider_snapshot=provider_config,
            config=config,
        )
    return 0


def command_review(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)
    config = load_pipeline_config(Path(args.config).expanduser())
    input_path = Path(args.input).expanduser().resolve()
    payload = load_payload(input_path, None)
    review_settings = resolve_review_settings(config, args)
    stem = args.stem or default_stem(float(review_settings["target_seconds"]))
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_path.parent

    segments = build_review_segments(payload, review_settings)
    outputs = write_review_outputs(
        payload=payload,
        input_path=input_path,
        output_dir=output_dir,
        stem=stem,
        review_settings=review_settings,
        segments=segments,
    )
    preview_settings = copy.deepcopy(config["preview"])
    if args.preview_resolution:
        preview_settings["resolution"] = args.preview_resolution
    if args.preview_crf is not None:
        preview_settings["crf"] = int(args.preview_crf)
    if args.preview_preset:
        preview_settings["preset"] = args.preview_preset

    preview_path = render_preview_if_requested(
        requested=bool(args.preview or preview_settings.get("enabled", False)),
        preview_settings=preview_settings,
        ffmpeg_override=args.ffmpeg,
        payload=payload,
        output_dir=output_dir,
        stem=stem,
        segments=segments,
    )

    for path in outputs.values():
        logging.info("Wrote %s", path)
    if preview_path:
        logging.info("Wrote %s", preview_path)
    return 0


def command_render(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)
    config = load_pipeline_config(Path(args.config).expanduser())
    input_path = Path(args.input).expanduser().resolve()
    payload = load_payload(input_path, None)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_path.parent

    review_settings = resolve_review_settings(config, args)
    if args.caption_mode:
        review_settings["caption_mode"] = args.caption_mode
    if args.target_seconds is not None:
        review_settings["target_seconds"] = float(args.target_seconds)

    segments = build_review_segments(payload, review_settings)
    render_settings = resolve_render_settings(config, args)
    stem = args.stem or infer_render_stem(input_path, float(review_settings["target_seconds"]))
    json_path, final_srt_path, source_srt_path = write_render_outputs(
        payload=payload,
        input_path=input_path,
        output_dir=output_dir,
        stem=stem,
        segments=segments,
        render_settings=render_settings,
    )

    source_path_text = str(payload["video"].get("source_path", "")).strip()
    if not source_path_text:
        raise ValueError("Rendering requires a source video path in the input JSON.")
    source_path = Path(source_path_text).expanduser().resolve()
    video_meta = probe_video(source_path)
    scale_height = parse_resolution(str(render_settings["resolution"]), video_meta.height)
    ffmpeg_path = find_ffmpeg(args.ffmpeg)
    output_video = output_dir / f"{stem}.mp4"

    render_video(
        ffmpeg_path=ffmpeg_path,
        source_video=source_path,
        output_video=output_video,
        clips=segments,
        preset=str(render_settings["preset"]),
        crf=int(render_settings["crf"]),
        scale_height=scale_height,
    )

    logging.info("Wrote %s", json_path)
    logging.info("Wrote %s", final_srt_path)
    logging.info("Wrote %s", source_srt_path)
    logging.info("Wrote %s", output_video)
    return 0


def command_edit_update_segment(args: argparse.Namespace) -> int:
    configure_logging("INFO")
    plan_path = Path(args.plan).expanduser().resolve()
    payload = load_plan(plan_path)
    output_path = Path(args.output).expanduser().resolve() if args.output else plan_path

    segments = [dict(segment) for segment in payload["segments"]]
    segment = next((segment for segment in segments if int(segment.get("rank", 0)) == int(args.rank)), None)
    if segment is None:
        raise ValueError(f"Segment rank {args.rank} not found in {plan_path}")

    if args.source_start_seconds is not None:
        segment["source_start_seconds"] = round(float(args.source_start_seconds), 3)
    if args.source_end_seconds is not None:
        segment["source_end_seconds"] = round(float(args.source_end_seconds), 3)

    payload["segments"] = rebase_editable_segments(segments)
    save_plan(output_path, payload)
    return 0


def command_edit_update_caption(args: argparse.Namespace) -> int:
    configure_logging("INFO")
    plan_path = Path(args.plan).expanduser().resolve()
    payload = load_plan(plan_path)
    output_path = Path(args.output).expanduser().resolve() if args.output else plan_path

    segments = [dict(segment) for segment in payload["segments"]]
    segment = next((segment for segment in segments if int(segment.get("rank", 0)) == int(args.rank)), None)
    if segment is None:
        raise ValueError(f"Segment rank {args.rank} not found in {plan_path}")

    segment["caption"] = str(args.caption).strip()
    segment["caption_detail"] = str(args.caption_detail or "").strip()
    payload["segments"] = segments
    save_plan(output_path, payload)
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "extract":
        return command_extract(args)
    if args.command == "infer":
        return command_infer(args)
    if args.command == "review":
        return command_review(args)
    if args.command == "render":
        return command_render(args)
    if args.command == "edit":
        if args.edit_command == "update-segment":
            return command_edit_update_segment(args)
        if args.edit_command == "update-caption":
            return command_edit_update_caption(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
