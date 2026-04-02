from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import io
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import cv2
import requests
import tomllib
from openai import OpenAI
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False

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
        "routing": "auto",
        "submission_mode": "auto",
        "type": "ollama",
        "enabled": True,
        "temperature": 0.1,
        "timeout_seconds": 120,
        "ollama": {
            "enabled": True,
            "supports_vision": True,
            "api_base": "http://127.0.0.1:11434",
            "model": "qwen3-vl:8b",
        },
        "gemini": {
            "enabled": True,
            "profile": "gemini_3_flash",
            "supports_vision": True,
            "supports_async_batch": True,
            "prefer_async_batch": False,
            "min_request_interval_seconds": 0.0,
            "api_base": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "model": "gemini-3-flash-preview",
            "api_key": "",
            "api_key_env": "GEMINI_API_KEY",
            "image_transport": "base64",
            "image_url_template": "",
            "json_output": True,
        },
        "qwen": {
            "enabled": True,
            "profile": "qwen3_vl_flash",
            "supports_vision": True,
            "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen3-vl-flash",
            "api_key": "",
            "api_key_env": "DASHSCOPE_API_KEY",
            "image_transport": "base64",
            "image_url_template": "",
            "json_output": False,
            "extra_body": {
                "enable_thinking": False,
            },
        },
        "openai_compatible": {
            "enabled": True,
            "profile": "deepseek_v3_2",
            "supports_vision": False,
            "api_base": "https://api.deepseek.com",
            "model": "deepseek-chat",
            "api_key": "",
            "api_key_env": "DEEPSEEK_API_KEY",
            "image_transport": "base64",
            "image_url_template": "",
            "json_output": True,
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

MIN_SAMPLE_FPS = 0.1
MAX_SAMPLE_FPS = 24.0


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
    def infer(
        self,
        image_bytes: bytes,
        *,
        image_path: Path | None,
        timestamp_seconds: float,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        ...


class AsyncBatchVisionProvider(Protocol):
    def submit_batch(
        self,
        *,
        index_path: Path,
        provider_snapshot: dict[str, Any],
        prompt_snapshot: dict[str, Any],
        selection_snapshot: dict[str, Any],
        force_resubmit: bool,
    ) -> Path:
        ...

    def collect_batch(self, manifest_path: Path) -> Path:
        ...

    def cancel_batch(self, manifest_path: Path) -> Path:
        ...


@dataclass
class ProviderSelection:
    route: str
    provider_type: str
    profile: str
    provider: Any
    snapshot: dict[str, Any]
    execution_mode: str = "sync"


def detect_mime_type(image_bytes: bytes) -> str:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return str(Image.MIME.get(image.format or "", "image/jpeg"))
    except UnidentifiedImageError:
        return "image/jpeg"


def build_data_url(image_bytes: bytes) -> str:
    mime_type = detect_mime_type(image_bytes)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def extract_message_text(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        text_parts: list[str] = []
        for item in message_content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).strip()
            if item_type in {"text", "output_text"}:
                text_value = item.get("text")
                if text_value:
                    text_parts.append(str(text_value))
        return "\n".join(text_parts)
    return str(message_content)


def json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_compatible(item) for item in value]
    if isinstance(value, tuple):
        return [json_compatible(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return json_compatible(value.model_dump(exclude_none=True))
        except TypeError:
            return json_compatible(value.model_dump())
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str):
        return enum_value
    return str(value)


def normalize_batch_state(state: Any) -> str:
    text = str(getattr(state, "value", state) or "").strip()
    return text or "JOB_STATE_UNSPECIFIED"


def is_terminal_batch_state(state: Any) -> bool:
    return normalize_batch_state(state) in {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_PAUSED",
    }


def build_batch_request_key(frame_number: int) -> str:
    return f"frame_{frame_number:09d}"


def parse_batch_request_key(key: str) -> int | None:
    prefix = "frame_"
    if not key.startswith(prefix):
        return None
    try:
        return int(key[len(prefix) :])
    except ValueError:
        return None


def safe_filename_stem(value: str) -> str:
    allowed = {"-", "_", "."}
    return "".join(char if char.isalnum() or char in allowed else "_" for char in value).strip("._") or "batch"


class OllamaVisionProvider:
    def __init__(self, *, api_base: str, model: str, temperature: float, timeout_seconds: float) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    def infer(
        self,
        image_bytes: bytes,
        *,
        image_path: Path | None,
        timestamp_seconds: float,
        config: dict[str, Any],
    ) -> dict[str, Any]:
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
        image_transport: str,
        image_url_template: str,
        json_output: bool,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.image_transport = image_transport.strip().lower() or "base64"
        self.image_url_template = image_url_template.strip()
        self.json_output = json_output
        self.extra_body = copy.deepcopy(extra_body) if extra_body else None
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.timeout_seconds,
        )

    def build_image_part(self, image_bytes: bytes, image_path: Path | None) -> dict[str, Any]:
        if self.image_transport == "url":
            if image_path is None:
                raise ValueError("image_transport=url requires an image path.")
            if not self.image_url_template:
                raise ValueError("image_transport=url requires image_url_template in provider config.")
            image_url = self.image_url_template.format(
                image_path=image_path.as_posix(),
                image_name=image_path.name,
                image_stem=image_path.stem,
            )
            return {"type": "image_url", "image_url": {"url": image_url}}

        if self.image_transport != "base64":
            raise ValueError(f"Unsupported image transport: {self.image_transport}")
        return {"type": "image_url", "image_url": {"url": build_data_url(image_bytes)}}

    def infer(
        self,
        image_bytes: bytes,
        *,
        image_path: Path | None,
        timestamp_seconds: float,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": config["prompt"]["system_instructions"],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_user_prompt(config, timestamp_seconds)},
                        self.build_image_part(image_bytes, image_path),
                    ],
                },
            ],
        }
        if self.temperature is not None and self.model != "deepseek-reasoner":
            payload["temperature"] = self.temperature
        if self.json_output:
            payload["response_format"] = {"type": "json_object"}
        if self.extra_body:
            payload["extra_body"] = copy.deepcopy(self.extra_body)
        response = self.client.chat.completions.create(**payload)
        message_content = extract_message_text(response.choices[0].message.content)
        return extract_json_block(message_content)


class GeminiBatchVisionProvider:
    def __init__(self, *, api_key: str, model: str, temperature: float, timeout_seconds: float) -> None:
        if genai is None:
            raise RuntimeError(
                "Gemini batch inference requires the official google-genai package. Run: pip install google-genai"
            )
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    def _build_request_record(self, frame: dict[str, Any], *, prompt_snapshot: dict[str, Any]) -> dict[str, Any]:
        image_path = Path(str(frame["image_path"])).expanduser().resolve()
        image_bytes = image_path.read_bytes()
        request: dict[str, Any] = {
            "system_instruction": {
                "parts": [
                    {
                        "text": str(prompt_snapshot["system_instructions"]),
                    }
                ]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": build_user_prompt(
                                {"prompt": prompt_snapshot},
                                float(frame["timestamp_seconds"]),
                            )
                        },
                        {
                            "inline_data": {
                                "mime_type": detect_mime_type(image_bytes),
                                "data": base64.b64encode(image_bytes).decode("ascii"),
                            }
                        },
                    ],
                }
            ],
            "generation_config": {
                "temperature": self.temperature,
                "response_mime_type": "application/json",
            },
        }
        return {
            "key": build_batch_request_key(int(frame["frame_number"])),
            "request": request,
        }

    async def _submit_batch_async(
        self,
        request_jsonl_path: Path,
        *,
        display_name: str,
    ) -> tuple[str, dict[str, Any]]:
        async with genai.Client(api_key=self.api_key).aio as client:
            uploaded_file = await client.files.upload(
                file=str(request_jsonl_path),
                config={
                    "display_name": f"{display_name}-requests",
                    "mime_type": "application/json",
                },
            )
            uploaded_file_name = str(getattr(uploaded_file, "name", uploaded_file))
            batch_job = await client.batches.create(
                model=self.model,
                src=uploaded_file_name,
                config={"display_name": display_name},
            )
        return uploaded_file_name, json_compatible(batch_job)

    async def _collect_batch_async(self, batch_name: str, *, result_file_name: str) -> tuple[dict[str, Any], bytes]:
        async with genai.Client(api_key=self.api_key).aio as client:
            batch_job = await client.batches.get(name=batch_name)
            downloaded = await client.files.download(file=result_file_name)
        if isinstance(downloaded, bytes):
            content_bytes = downloaded
        elif isinstance(downloaded, bytearray):
            content_bytes = bytes(downloaded)
        elif hasattr(downloaded, "read"):
            content_bytes = downloaded.read()
        else:
            data = getattr(downloaded, "data", None)
            if isinstance(data, bytes):
                content_bytes = data
            elif isinstance(data, bytearray):
                content_bytes = bytes(data)
            else:
                raise RuntimeError(f"Unsupported Gemini batch download payload: {type(downloaded).__name__}")
        return json_compatible(batch_job), content_bytes

    async def _get_batch_async(self, batch_name: str) -> dict[str, Any]:
        async with genai.Client(api_key=self.api_key).aio as client:
            batch_job = await client.batches.get(name=batch_name)
        return json_compatible(batch_job)

    def _batch_rest_url(self, batch_name: str) -> str:
        return f"https://generativelanguage.googleapis.com/v1beta/{batch_name}"

    def _get_batch_operation(self, batch_name: str) -> dict[str, Any]:
        response = requests.get(
            self._batch_rest_url(batch_name),
            headers={"x-goog-api-key": self.api_key},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def submit_batch(
        self,
        *,
        index_path: Path,
        provider_snapshot: dict[str, Any],
        prompt_snapshot: dict[str, Any],
        selection_snapshot: dict[str, Any],
        force_resubmit: bool,
    ) -> Path:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        video_info = dict(payload["video"])
        candidate_frames = [frame for frame in payload["frames"] if frame.get("candidate") and frame.get("image_path")]
        if not candidate_frames:
            raise ValueError(f"No candidate frames found in {index_path}")

        video_output_dir = index_path.parent.parent
        manifest_path = video_output_dir / "analysis.batch.json"
        if manifest_path.exists() and not force_resubmit:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing_state = normalize_batch_state(existing.get("batch", {}).get("state", ""))
            if existing.get("source_extract_index") == str(index_path) and not is_terminal_batch_state(existing_state):
                logging.info("Existing async batch is still active for %s: %s", video_info["filename"], manifest_path)
                return manifest_path

        batch_dir = video_output_dir / "batch"
        batch_dir.mkdir(parents=True, exist_ok=True)
        request_jsonl_path = batch_dir / f"{safe_filename_stem(video_info['filename'])}.{safe_filename_stem(self.model)}.requests.json"
        with request_jsonl_path.open("w", encoding="utf-8") as handle:
            for frame in candidate_frames:
                handle.write(
                    json.dumps(
                        self._build_request_record(frame, prompt_snapshot=prompt_snapshot),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        display_name = f"{safe_filename_stem(Path(video_info['filename']).stem)}-{int(time.time())}"
        uploaded_file_name, batch_job = asyncio.run(
            self._submit_batch_async(
                request_jsonl_path,
                display_name=display_name,
            )
        )
        batch_payload = {
            "stage": "infer.batch",
            "status": "submitted",
            "video": video_info,
            "source_extract_index": str(index_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "provider_snapshot": provider_snapshot,
            "config_snapshot": {
                "prompt": prompt_snapshot,
                "selection": selection_snapshot,
            },
            "batch": {
                "provider": "gemini",
                "execution_mode": "async_batch",
                "request_count": len(candidate_frames),
                "request_jsonl_path": str(request_jsonl_path),
                "source_file_name": uploaded_file_name,
                "job": batch_job,
                "name": str(batch_job.get("name", "")),
                "state": normalize_batch_state(batch_job.get("state", "")),
                "dest_file_name": str(
                    batch_job.get("dest", {}).get("file_name", "")
                    if isinstance(batch_job.get("dest"), dict)
                    else ""
                ),
            },
        }
        manifest_path.write_text(json.dumps(batch_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("Submitted Gemini batch job %s for %s", batch_payload["batch"]["name"], video_info["filename"])
        logging.info("Wrote %s", manifest_path)
        return manifest_path

    def collect_batch(self, manifest_path: Path) -> Path:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        batch_info = dict(manifest.get("batch", {}))
        batch_name = str(batch_info.get("name", "")).strip()
        if not batch_name:
            raise ValueError(f"{manifest_path} does not contain a batch name.")

        operation = self._get_batch_operation(batch_name)
        latest_job = asyncio.run(self._get_batch_async(batch_name))
        batch_info["job"] = latest_job
        batch_info["state"] = normalize_batch_state(latest_job.get("state", ""))
        batch_info["operation"] = operation
        metadata = operation.get("metadata", {}) if isinstance(operation, dict) else {}
        if isinstance(metadata, dict):
            batch_info["batch_stats"] = metadata.get("batchStats", {})
        dest = latest_job.get("dest", {})
        if isinstance(dest, dict):
            batch_info["dest_file_name"] = str(dest.get("file_name", "")).strip()
        manifest["batch"] = batch_info
        manifest["last_checked_at"] = datetime.now(timezone.utc).isoformat()

        if batch_info["state"] != "JOB_STATE_SUCCEEDED":
            manifest["status"] = "pending" if not is_terminal_batch_state(batch_info["state"]) else "failed"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            if batch_info["state"] == "JOB_STATE_FAILED":
                raise RuntimeError(f"Gemini batch job failed: {json.dumps(latest_job, ensure_ascii=False)}")
            logging.info("Gemini batch job is not ready yet: %s", batch_info["state"])
            logging.info("Updated %s", manifest_path)
            return manifest_path

        result_file_name = str(batch_info.get("dest_file_name", "")).strip()
        if not result_file_name:
            raise RuntimeError(f"Gemini batch {batch_name} succeeded but did not expose a destination file.")
        latest_job, content_bytes = asyncio.run(
            self._collect_batch_async(
                batch_name,
                result_file_name=result_file_name,
            )
        )
        batch_info["job"] = latest_job
        batch_info["state"] = normalize_batch_state(latest_job.get("state", ""))
        manifest["batch"] = batch_info
        manifest["status"] = "succeeded"

        results_path = manifest_path.parent / "batch" / f"{safe_filename_stem(Path(manifest['video']['filename']).stem)}.responses.jsonl"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_bytes(content_bytes)
        manifest["batch"]["results_jsonl_path"] = str(results_path)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("Downloaded Gemini batch results to %s", results_path)
        return results_path

    def cancel_batch(self, manifest_path: Path) -> Path:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        batch_info = dict(manifest.get("batch", {}))
        batch_name = str(batch_info.get("name", "")).strip()
        if not batch_name:
            raise ValueError(f"{manifest_path} does not contain a batch name.")
        response = requests.post(
            f"{self._batch_rest_url(batch_name)}:cancel",
            headers={"x-goog-api-key": self.api_key},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        operation = self._get_batch_operation(batch_name)
        metadata = operation.get("metadata", {}) if isinstance(operation, dict) else {}
        if isinstance(metadata, dict):
            batch_info["state"] = str(metadata.get("state", batch_info.get("state", ""))).strip()
            batch_info["batch_stats"] = metadata.get("batchStats", {})
        batch_info["operation"] = operation
        manifest["batch"] = batch_info
        manifest["status"] = "cancelling"
        manifest["cancel_requested_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("Requested cancellation for %s", batch_name)
        logging.info("Updated %s", manifest_path)
        return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-platform video highlight pipeline with extract / infer / review / render stages."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Sample candidate frames from video(s).")
    add_common_video_args(extract_parser)
    extract_parser.add_argument("--frame-interval-seconds", type=float, help="Sample one frame every N seconds.")
    extract_parser.add_argument(
        "--sample-fps",
        type=float,
        help=f"Alternative to frame interval. Sample N frames per second ({MIN_SAMPLE_FPS}-{MAX_SAMPLE_FPS}, default 1).",
    )
    extract_parser.add_argument("--max-frames", type=int, help="Limit the number of sampled frames.")
    extract_parser.add_argument("--jpeg-quality", type=int, help="JPEG quality used when writing extracted frames.")
    extract_parser.add_argument("--resize-for-llm", type=int, help="Maximum output dimension for extracted frames.")
    extract_parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    infer_parser = subparsers.add_parser("infer", help="Run vision-model inference over extracted frames.")
    add_common_video_args(infer_parser)
    infer_parser.add_argument("--extract-index", help="Path to extract/index.json. If missing, the pipeline will generate it.")
    infer_parser.add_argument(
        "--provider",
        choices=["auto", "local", "gemini", "qwen", "api"],
        help="Provider routing for this run. auto prefers local Ollama, then Gemini, then Qwen, then generic API.",
    )
    infer_parser.add_argument("--provider-type", choices=["ollama", "openai_compatible"])
    infer_parser.add_argument("--api-base", help="Provider API base URL.")
    infer_parser.add_argument("--api-key", help="API key for third-party providers.")
    infer_parser.add_argument("--api-key-env", help="Environment variable used to read the API key.")
    infer_parser.add_argument("--model", help="Vision model name.")
    infer_parser.add_argument("--temperature", type=float, help="Sampling temperature for the provider.")
    infer_parser.add_argument("--timeout-seconds", type=float, help="Provider timeout in seconds.")
    infer_parser.add_argument(
        "--submission-mode",
        choices=["auto", "sync", "async"],
        help="Execution mode for inference. auto uses async batch when the selected API provider supports it.",
    )
    infer_parser.add_argument(
        "--force-resubmit",
        action="store_true",
        help="Create a new async batch even if an active manifest already exists for the same extract output.",
    )
    infer_parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore any saved sync infer checkpoint and start from the first candidate frame again.",
    )
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

    collect_parser = subparsers.add_parser("collect", help="Collect results for a previously submitted async infer job.")
    collect_parser.add_argument("--manifest", required=True, help="Path to analysis.batch.json")
    collect_parser.add_argument("--config", default="config.toml", help="Optional TOML config path.")
    collect_parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    cancel_parser = subparsers.add_parser("cancel", help="Cancel a submitted async infer batch job.")
    cancel_parser.add_argument("--manifest", required=True, help="Path to analysis.batch.json")
    cancel_parser.add_argument("--config", default="config.toml", help="Optional TOML config path.")
    cancel_parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    run_parser = subparsers.add_parser("run", help="Run extract -> infer -> review -> render in one command.")
    add_common_video_args(run_parser)
    run_parser.add_argument("--extract-index", help="Optional extract/index.json. Skips extract when provided.")
    run_parser.add_argument(
        "--provider",
        choices=["auto", "local", "gemini", "qwen", "api"],
        help="Provider routing for this run. auto prefers local Ollama, then Gemini, then Qwen, then generic API.",
    )
    run_parser.add_argument("--provider-type", choices=["ollama", "openai_compatible"])
    run_parser.add_argument("--api-base", help="Provider API base URL.")
    run_parser.add_argument("--api-key", help="API key for third-party providers.")
    run_parser.add_argument("--api-key-env", help="Environment variable used to read the API key.")
    run_parser.add_argument("--model", help="Vision model name.")
    run_parser.add_argument("--temperature", type=float, help="Sampling temperature for the provider.")
    run_parser.add_argument("--timeout-seconds", type=float, help="Provider timeout in seconds.")
    run_parser.add_argument(
        "--submission-mode",
        choices=["auto", "sync", "async"],
        help="Execution mode for inference. auto uses async batch when the selected API provider supports it.",
    )
    run_parser.add_argument(
        "--force-resubmit",
        action="store_true",
        help="Create a new async batch even if an active manifest already exists for the same extract output.",
    )
    run_parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore any saved sync infer checkpoint and start from the first candidate frame again.",
    )
    run_parser.add_argument("--frame-interval-seconds", type=float, help="Sample one frame every N seconds.")
    run_parser.add_argument(
        "--sample-fps",
        type=float,
        help=f"Alternative to frame interval. Sample N frames per second ({MIN_SAMPLE_FPS}-{MAX_SAMPLE_FPS}, default 1).",
    )
    run_parser.add_argument("--max-frames", type=int, help="Limit the number of sampled frames.")
    run_parser.add_argument("--jpeg-quality", type=int, help="JPEG quality used when writing extracted frames.")
    run_parser.add_argument("--resize-for-llm", type=int, help="Maximum output dimension for extracted frames.")
    run_parser.add_argument("--target-seconds", type=float, help="Target total duration for the compact plan.")
    run_parser.add_argument("--clip-before", type=float, help="Seconds kept before each highlight anchor.")
    run_parser.add_argument("--clip-after", type=float, help="Seconds kept after each highlight anchor.")
    run_parser.add_argument("--cluster-gap", type=float, help="Maximum gap between timestamps before split.")
    run_parser.add_argument("--min-clip-seconds", type=float, help="Minimum duration for selected clips.")
    run_parser.add_argument("--max-clip-seconds", type=float, help="Maximum duration for selected clips.")
    run_parser.add_argument("--max-clips-per-source-segment", type=int, help="Diversity limit per source segment.")
    run_parser.add_argument(
        "--caption-mode",
        choices=["score", "reason", "human"],
        help="How subtitles should be written for review and render.",
    )
    run_parser.add_argument("--stem", help="Output stem for review/render outputs.")
    run_parser.add_argument(
        "--resolution",
        choices=sorted(RESOLUTION_HEIGHTS.keys()),
        help="Render resolution. One of 540p, 720p, 1080p, source.",
    )
    run_parser.add_argument("--crf", type=int, help="CRF used when encoding intermediate clips.")
    run_parser.add_argument("--preset", help="Preset used when encoding intermediate clips.")
    run_parser.add_argument("--ffmpeg", help="Optional path to ffmpeg executable.")
    run_parser.add_argument("--skip-review", action="store_true", help="Render directly from analysis.json.")
    run_parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

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


def load_project_env(config_path: Path) -> None:
    seen: set[Path] = set()
    candidates = [Path.cwd() / ".env", config_path.parent / ".env"]
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen or not resolved.exists():
            continue
        load_dotenv(resolved, override=False)
        seen.add(resolved)


def load_pipeline_config(path: Path) -> dict[str, Any]:
    load_project_env(path)
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
            "enabled": legacy_ollama.get("enabled", provider["ollama"]["enabled"]),
            "api_base": legacy_ollama.get("host", provider["ollama"]["api_base"]),
            "model": legacy_ollama.get("model", provider["ollama"]["model"]),
        },
    )
    raw_provider = raw.get("provider", {})
    provider = deep_merge_dict(provider, raw_provider)
    if "routing" not in raw_provider:
        legacy_provider_type = str(raw_provider.get("type", "")).strip()
        if legacy_provider_type == "ollama":
            provider["routing"] = "local"
        elif legacy_provider_type == "openai_compatible":
            provider["routing"] = "api"
    deepseek_base_url = str(os.environ.get("DEEPSEEK_BASE_URL", "")).strip()
    if deepseek_base_url:
        provider["openai_compatible"]["api_base"] = deepseek_base_url
    gemini_base_url = str(os.environ.get("GEMINI_BASE_URL", "")).strip()
    if gemini_base_url:
        provider["gemini"]["api_base"] = gemini_base_url
    dashscope_base_url = str(os.environ.get("DASHSCOPE_BASE_URL", "")).strip()
    if dashscope_base_url:
        provider["qwen"]["api_base"] = dashscope_base_url
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
    resolved_sample_fps = float(resolved["sample_fps"])
    if not (MIN_SAMPLE_FPS <= resolved_sample_fps <= MAX_SAMPLE_FPS):
        raise ValueError(
            f"sample_fps must be between {MIN_SAMPLE_FPS} and {MAX_SAMPLE_FPS}, got {resolved_sample_fps}"
        )
    resolved["jpeg_quality"] = int(resolved.get("jpeg_quality", 88))
    resolved["max_frames"] = int(resolved.get("max_frames", 0))
    resolved["resize_for_llm"] = int(resolved.get("resize_for_llm", 0))
    resolved["forced_keep_interval_seconds"] = float(resolved.get("forced_keep_interval_seconds", 3.0))
    return resolved


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(levelname)s: %(message)s")


def resolve_output_root(config: dict[str, Any], override: str | None, video_path: Path | None = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    if video_path is not None:
        return video_path.parent.resolve()
    return Path(config["project"]["output"]).expanduser().resolve()


def resolve_video_output_dir(config: dict[str, Any], override: str | None, video_path: Path) -> Path:
    output_root = resolve_output_root(config, override, video_path)
    return output_root / video_path.stem


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
    if getattr(args, "provider", None):
        provider["routing"] = args.provider
    if getattr(args, "submission_mode", None):
        provider["submission_mode"] = args.submission_mode
    if getattr(args, "provider_type", None):
        provider["type"] = args.provider_type
    if getattr(args, "temperature", None) is not None:
        provider["temperature"] = float(args.temperature)
    if getattr(args, "timeout_seconds", None) is not None:
        provider["timeout_seconds"] = float(args.timeout_seconds)

    provider_target = infer_provider_target(args, provider)
    if provider_target == "local":
        if getattr(args, "api_base", None):
            provider["ollama"]["api_base"] = args.api_base
        if getattr(args, "model", None):
            provider["ollama"]["model"] = args.model
    elif provider_target == "gemini":
        if getattr(args, "api_base", None):
            provider["gemini"]["api_base"] = args.api_base
        if getattr(args, "model", None):
            provider["gemini"]["model"] = args.model
        if getattr(args, "api_key", None):
            provider["gemini"]["api_key"] = args.api_key
        if getattr(args, "api_key_env", None):
            provider["gemini"]["api_key_env"] = args.api_key_env
    elif provider_target == "qwen":
        if getattr(args, "api_base", None):
            provider["qwen"]["api_base"] = args.api_base
        if getattr(args, "model", None):
            provider["qwen"]["model"] = args.model
        if getattr(args, "api_key", None):
            provider["qwen"]["api_key"] = args.api_key
        if getattr(args, "api_key_env", None):
            provider["qwen"]["api_key_env"] = args.api_key_env
    elif provider_target == "api":
        if getattr(args, "api_base", None):
            provider["openai_compatible"]["api_base"] = args.api_base
        if getattr(args, "model", None):
            provider["openai_compatible"]["model"] = args.model
        if getattr(args, "api_key", None):
            provider["openai_compatible"]["api_key"] = args.api_key
        if getattr(args, "api_key_env", None):
            provider["openai_compatible"]["api_key_env"] = args.api_key_env
    return provider


def infer_provider_target(args: argparse.Namespace, provider_config: dict[str, Any]) -> str | None:
    explicit_provider = str(getattr(args, "provider", "") or "").strip().lower()
    if explicit_provider in {"local", "gemini", "qwen", "api"}:
        return explicit_provider

    explicit_type = str(getattr(args, "provider_type", "") or "").strip().lower()
    if explicit_type == "ollama":
        return "local"
    if explicit_type == "openai_compatible":
        return "api"

    api_base = str(getattr(args, "api_base", "") or "").strip().lower()
    api_key_env = str(getattr(args, "api_key_env", "") or "").strip().upper()
    if "generativelanguage.googleapis.com" in api_base or api_key_env == "GEMINI_API_KEY":
        return "gemini"
    if "dashscope" in api_base or api_key_env == "DASHSCOPE_API_KEY":
        return "qwen"
    if getattr(args, "api_base", None) or getattr(args, "api_key", None) or getattr(args, "api_key_env", None):
        return "api"

    model_name = str(getattr(args, "model", "") or "").strip().lower()
    if model_name.startswith("gemini-"):
        return "gemini"
    if model_name.startswith("qwen"):
        return "qwen"
    if model_name.startswith(("deepseek-", "gpt-", "claude-")):
        return "api"
    if ":" in model_name:
        return "local"

    routing = str(provider_config.get("routing", "auto")).strip().lower()
    if routing in {"local", "gemini", "qwen", "api"}:
        return routing
    return None


def is_ollama_available(*, api_base: str, model: str, timeout_seconds: float) -> tuple[bool, str]:
    if not api_base.strip():
        return False, "missing api_base"
    if not model.strip():
        return False, "missing model"
    try:
        response = requests.get(
            f"{api_base.rstrip('/')}/api/tags",
            timeout=min(max(timeout_seconds, 1.0), 3.0),
        )
        response.raise_for_status()
        payload = response.json()
        models = {
            str(item.get("name") or item.get("model") or "").strip()
            for item in payload.get("models", [])
            if isinstance(item, dict)
        }
        if models and model not in models:
            return False, f"model {model!r} not found in Ollama"
        return True, "reachable"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def resolve_api_key(api_config: dict[str, Any]) -> str:
    api_key = str(api_config.get("api_key", "")).strip()
    api_key_env = str(api_config.get("api_key_env", "")).strip()
    if not api_key and api_key_env:
        api_key = str(os.environ.get(api_key_env, "")).strip()
    return api_key


def sanitize_provider_snapshot(provider_config: dict[str, Any], *, route: str, provider_type: str) -> dict[str, Any]:
    snapshot = copy.deepcopy(provider_config)
    snapshot["routing"] = str(provider_config.get("routing", "auto")).strip().lower()
    snapshot["selected_route"] = route
    snapshot["selected_provider_type"] = provider_type
    if "gemini" in snapshot:
        snapshot["gemini"]["api_key"] = ""
    if "qwen" in snapshot:
        snapshot["qwen"]["api_key"] = ""
    if "openai_compatible" in snapshot:
        snapshot["openai_compatible"]["api_key"] = ""
    return snapshot


def validate_remote_provider(route: str, remote_config: dict[str, Any]) -> tuple[bool, str, str]:
    if not bool(remote_config.get("enabled", True)):
        return False, f"{route} disabled", ""
    if not bool(remote_config.get("supports_vision", True)):
        return False, f"{route} does not advertise vision support", ""
    api_key = resolve_api_key(remote_config)
    if not api_key:
        return False, f"{route} unavailable (missing api key)", ""
    api_base = str(remote_config.get("api_base", "")).strip()
    model = str(remote_config.get("model", "")).strip()
    if not api_base or not model:
        return False, f"{route} unavailable (missing api_base or model)", ""
    return True, "", api_key


def build_openai_compatible_provider(
    remote_config: dict[str, Any],
    *,
    temperature: float,
    timeout_seconds: float,
) -> OpenAICompatibleVisionProvider:
    return OpenAICompatibleVisionProvider(
        api_base=str(remote_config["api_base"]).strip(),
        model=str(remote_config["model"]).strip(),
        api_key=resolve_api_key(remote_config),
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        image_transport=str(remote_config.get("image_transport", "base64")).strip(),
        image_url_template=str(remote_config.get("image_url_template", "")).strip(),
        json_output=bool(remote_config.get("json_output", True)),
        extra_body=remote_config.get("extra_body") if isinstance(remote_config.get("extra_body"), dict) else None,
    )


def build_gemini_batch_provider(
    remote_config: dict[str, Any],
    *,
    temperature: float,
    timeout_seconds: float,
) -> GeminiBatchVisionProvider:
    return GeminiBatchVisionProvider(
        api_key=resolve_api_key(remote_config),
        model=str(remote_config["model"]).strip(),
        temperature=temperature,
        timeout_seconds=timeout_seconds,
    )


def resolve_execution_mode(route: str, provider_config: dict[str, Any], remote_config: dict[str, Any] | None) -> str:
    requested = str(provider_config.get("submission_mode", "auto")).strip().lower()
    if requested not in {"auto", "sync", "async"}:
        raise ValueError(f"Unsupported submission mode: {requested}")
    if route == "local":
        return "sync"
    if requested == "sync":
        return "sync"
    supports_async_batch = bool(remote_config and remote_config.get("supports_async_batch", False))
    if requested == "async":
        if not supports_async_batch:
            raise ValueError(f"{route} does not support async batch submission.")
        return "async_batch"
    if supports_async_batch and bool(remote_config and remote_config.get("prefer_async_batch", False)):
        return "async_batch"
    return "sync"


def build_provider(provider_config: dict[str, Any]) -> ProviderSelection:
    if not bool(provider_config.get("enabled", True)):
        raise ValueError("Provider is disabled in configuration.")

    provider_type = str(provider_config.get("type", "ollama")).strip()
    temperature = float(provider_config.get("temperature", 0.1))
    timeout_seconds = float(provider_config.get("timeout_seconds", 120))
    routing = str(provider_config.get("routing", "auto")).strip().lower()
    if routing not in {"auto", "local", "gemini", "qwen", "api"}:
        raise ValueError(f"Unsupported provider routing: {routing}")

    local_config = provider_config["ollama"]
    gemini_config = provider_config["gemini"]
    qwen_config = provider_config["qwen"]
    api_config = provider_config["openai_compatible"]

    candidates: list[tuple[str, str]] = []
    if routing in {"auto", "local"}:
        candidates.append(("local", "ollama"))
    if routing in {"auto", "gemini"}:
        candidates.append(("gemini", "gemini"))
    if routing in {"auto", "qwen"}:
        candidates.append(("qwen", "qwen"))
    if routing in {"auto", "api"}:
        candidates.append(("api", "openai_compatible"))

    failures: list[str] = []
    for route, resolved_provider_type in candidates:
        if route == "local":
            if not bool(local_config.get("enabled", True)):
                failures.append("local disabled")
                continue
            reachable, reason = is_ollama_available(
                api_base=str(local_config.get("api_base", "")).strip(),
                model=str(local_config.get("model", "")).strip(),
                timeout_seconds=timeout_seconds,
            )
            if not reachable:
                failures.append(f"local unavailable ({reason})")
                if routing == "local":
                    continue
                continue
            return ProviderSelection(
                route=route,
                provider_type=resolved_provider_type,
                profile=str(local_config.get("model", "")).strip() or "ollama",
                provider=OllamaVisionProvider(
                    api_base=str(local_config["api_base"]).strip(),
                    model=str(local_config["model"]).strip(),
                    temperature=temperature,
                    timeout_seconds=timeout_seconds,
                ),
                snapshot=sanitize_provider_snapshot(
                    provider_config,
                    route=route,
                    provider_type=resolved_provider_type,
                ),
            )

        if route == "gemini":
            remote_config = gemini_config
        elif route == "qwen":
            remote_config = qwen_config
        else:
            remote_config = api_config
        valid, failure, _ = validate_remote_provider(route, remote_config)
        if not valid:
            failures.append(failure)
            continue
        execution_mode = resolve_execution_mode(route, provider_config, remote_config)
        if route == "gemini" and execution_mode == "async_batch":
            provider_instance = build_gemini_batch_provider(
                remote_config,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
            )
        else:
            provider_instance = build_openai_compatible_provider(
                remote_config,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
            )
        return ProviderSelection(
            route=route,
            provider_type=resolved_provider_type,
            profile=str(remote_config.get("profile", "")).strip() or str(remote_config.get("model", "")).strip(),
            provider=provider_instance,
            snapshot=sanitize_provider_snapshot(
                provider_config,
                route=route,
                provider_type=resolved_provider_type,
            ),
            execution_mode=execution_mode,
        )

    detail = "; ".join(failures) if failures else f"unsupported provider type: {provider_type}"
    raise ValueError(f"No available provider for routing={routing}. {detail}")


def summarize_provider_exception(exc: Exception) -> str:
    message = str(exc).strip().replace("\r", " ").replace("\n", " ")
    return message[:240] if message else exc.__class__.__name__


def status_code_for_exception(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status
    return None


def retry_delay_for_exception(exc: Exception) -> float | None:
    status_code = status_code_for_exception(exc)
    if status_code != 429:
        return None
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after:
            try:
                return max(1.0, float(retry_after))
            except ValueError:
                pass
    return 15.0


def is_fatal_provider_exception(exc: Exception) -> tuple[bool, str]:
    status_code = status_code_for_exception(exc)
    message = summarize_provider_exception(exc).lower()
    fatal_markers = [
        "invalid api key",
        "api key not valid",
        "authentication",
        "unauthorized",
        "unauthenticated",
        "permission denied",
        "insufficient balance",
        "quota",
        "billing",
        "model not found",
        "unsupported model",
        "unknown variant `image_url`",
        "does not advertise vision support",
        "does not support image",
        "does not support vision",
        "not support image",
        "not support vision",
        "invalid_request_error",
    ]
    if status_code in {400, 401, 402, 403, 404}:
        return True, summarize_provider_exception(exc)
    if any(marker in message for marker in fatal_markers):
        return True, summarize_provider_exception(exc)
    return False, summarize_provider_exception(exc)


def ensure_extract_index(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    if getattr(args, "extract_index", None):
        return Path(args.extract_index).expanduser().resolve()

    video_path = Path(args.video).expanduser().resolve()
    video_output_dir = resolve_video_output_dir(config, getattr(args, "output_root", None), video_path)
    candidate_index = video_output_dir / "extract" / "index.json"
    if candidate_index.exists():
        return candidate_index

    logging.info("Extract index missing for %s, running extract stage first.", video_path)
    extract_settings = config["extract"]
    return extract_candidates_for_video(
        video_path,
        output_root=video_output_dir.parent,
        config=config,
        frame_interval_seconds=float(extract_settings["frame_interval_seconds"]),
        max_frames=int(extract_settings["max_frames"]),
        jpeg_quality=int(extract_settings["jpeg_quality"]),
        resize_for_llm=int(extract_settings["resize_for_llm"]),
    )


def provider_error_decision(reason: str) -> dict[str, Any]:
    return {
        "keep": False,
        "score": 0.0,
        "labels": [],
        "reason": "",
        "discard_reason": reason,
    }


def build_infer_records(
    extract_payload: dict[str, Any],
    *,
    decisions_by_frame_number: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for frame in extract_payload["frames"]:
        decision = decisions_by_frame_number.get(int(frame["frame_number"]))
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
    return records


def checkpoint_path_for_index(index_path: Path) -> Path:
    return index_path.parent.parent / "frame_decisions.checkpoint.jsonl"


def progress_path_for_index(index_path: Path) -> Path:
    return index_path.parent.parent / "infer.progress.json"


def extract_decision_fields(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "keep": bool(record.get("keep", False)),
        "score": float(record.get("score", 0.0) or 0.0),
        "labels": [str(label) for label in record.get("labels", []) if str(label).strip()],
        "reason": str(record.get("reason", "")).strip(),
        "discard_reason": str(record.get("discard_reason", "")).strip(),
    }


def build_decision_record(frame: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
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
    record.update(extract_decision_fields(decision))
    if record["keep"] and record["image_path"]:
        record["thumbnail_path"] = record["image_path"]
    return record


def load_checkpoint_decisions(index_path: Path) -> dict[int, dict[str, Any]]:
    checkpoint_path = checkpoint_path_for_index(index_path)
    if not checkpoint_path.exists():
        return {}
    decisions_by_frame_number: dict[int, dict[str, Any]] = {}
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            frame_number = item.get("frame_number")
            if frame_number is None:
                continue
            try:
                decisions_by_frame_number[int(frame_number)] = extract_decision_fields(item)
            except (TypeError, ValueError):
                continue
    return decisions_by_frame_number


def write_sync_progress(
    index_path: Path,
    *,
    provider_snapshot: dict[str, Any],
    total_candidate_frames: int,
    completed_candidate_frames: int,
    resumed_candidate_frames: int,
    last_frame: dict[str, Any] | None,
    status: str,
    analysis_path: Path | None = None,
    error: str = "",
) -> Path:
    progress_payload = {
        "stage": "infer.progress",
        "status": status,
        "source_extract_index": str(index_path),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "provider_snapshot": provider_snapshot,
        "checkpoint_path": str(checkpoint_path_for_index(index_path)),
        "analysis_path": str(analysis_path) if analysis_path else "",
        "total_candidate_frames": int(total_candidate_frames),
        "completed_candidate_frames": int(completed_candidate_frames),
        "remaining_candidate_frames": max(0, int(total_candidate_frames) - int(completed_candidate_frames)),
        "resumed_candidate_frames": int(resumed_candidate_frames),
        "last_frame": build_decision_record(last_frame["frame"], last_frame["decision"]) if last_frame else None,
        "error": str(error).strip(),
    }
    progress_path = progress_path_for_index(index_path)
    progress_path.write_text(json.dumps(progress_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return progress_path


def write_infer_outputs(
    index_path: Path,
    *,
    extract_payload: dict[str, Any],
    provider_snapshot: dict[str, Any],
    prompt_snapshot: dict[str, Any],
    selection_snapshot: dict[str, Any],
    decisions_by_frame_number: dict[int, dict[str, Any]],
) -> Path:
    video_info = dict(extract_payload["video"])
    video_output_dir = index_path.parent.parent
    records = build_infer_records(extract_payload, decisions_by_frame_number=decisions_by_frame_number)
    segments = merge_segments(
        records,
        float(video_info["duration_seconds"]),
        {"decision": selection_snapshot},
    )

    analysis_payload = {
        "stage": "infer",
        "video": video_info,
        "source_extract_index": str(index_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider_snapshot": provider_snapshot,
        "config_snapshot": {
            "prompt": prompt_snapshot,
            "selection": selection_snapshot,
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


def extract_gemini_batch_response_text(response_payload: dict[str, Any]) -> str:
    if not isinstance(response_payload, dict):
        return ""
    direct_text = response_payload.get("text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text
    text_parts: list[str] = []
    for candidate in response_payload.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content", {})
        if not isinstance(content, dict):
            continue
        for part in content.get("parts", []):
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if text:
                text_parts.append(str(text))
    return "\n".join(text_parts)


def summarize_batch_error_payload(error_payload: Any) -> str:
    if isinstance(error_payload, dict):
        for key in ("message", "status", "code"):
            value = error_payload.get(key)
            if value:
                return str(value)
        return json.dumps(error_payload, ensure_ascii=False)
    return str(error_payload)


def parse_gemini_batch_results(
    results_path: Path,
    *,
    selection_snapshot: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    decisions_by_frame_number: dict[int, dict[str, Any]] = {}
    for raw_line in results_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        item = json.loads(line)
        frame_number = parse_batch_request_key(str(item.get("key", "")).strip())
        if frame_number is None:
            continue
        error_payload = item.get("error")
        if error_payload:
            decisions_by_frame_number[frame_number] = provider_error_decision(
                f"provider_error: {summarize_batch_error_payload(error_payload)}"
            )
            continue
        response_payload = item.get("response", {})
        response_text = extract_gemini_batch_response_text(response_payload)
        if not response_text:
            decisions_by_frame_number[frame_number] = provider_error_decision("provider_error: empty batch response")
            continue
        try:
            decisions_by_frame_number[frame_number] = sanitize_decision(
                extract_json_block(response_text),
                {"decision": selection_snapshot},
            )
        except Exception as exc:  # noqa: BLE001
            decisions_by_frame_number[frame_number] = provider_error_decision(
                f"provider_error: {summarize_provider_exception(exc)}"
            )
    return decisions_by_frame_number


def infer_from_extract_index(
    index_path: Path,
    *,
    provider: VisionProvider,
    provider_snapshot: dict[str, Any],
    config: dict[str, Any],
    restart: bool = False,
) -> Path:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    video_info = dict(payload["video"])
    candidate_frames = [frame for frame in payload["frames"] if frame.get("candidate") and frame.get("image_path")]
    existing_decisions = {} if restart else load_checkpoint_decisions(index_path)
    remaining_frames = [frame for frame in candidate_frames if int(frame["frame_number"]) not in existing_decisions]
    progress = tqdm(
        remaining_frames,
        desc=f"infer:{video_info['filename']}",
        unit="frame",
        total=len(candidate_frames),
        initial=len(existing_decisions),
    )
    selected_route = str(provider_snapshot.get("selected_route", "")).strip()
    selected_provider_config = provider_snapshot.get(selected_route, {}) if selected_route else {}
    min_request_interval_seconds = float(selected_provider_config.get("min_request_interval_seconds", 0.0) or 0.0)
    last_request_started_at = 0.0

    decisions_by_frame_number: dict[int, dict[str, Any]] = dict(existing_decisions)
    checkpoint_path = checkpoint_path_for_index(index_path)
    if restart and checkpoint_path.exists():
        checkpoint_path.unlink()
    progress_path = write_sync_progress(
        index_path,
        provider_snapshot=provider_snapshot,
        total_candidate_frames=len(candidate_frames),
        completed_candidate_frames=len(decisions_by_frame_number),
        resumed_candidate_frames=len(existing_decisions),
        last_frame=None,
        status="running",
    )
    processed_this_run = 0
    last_completed_frame: dict[str, Any] | None = None
    fatal_provider_error: str | None = None
    try:
        with checkpoint_path.open("a", encoding="utf-8") as checkpoint_handle:
            for frame in progress:
                image_path = Path(frame["image_path"])
                image_bytes = image_path.read_bytes()
                should_abort_run = False
                while True:
                    if min_request_interval_seconds > 0:
                        elapsed = time.monotonic() - last_request_started_at
                        if elapsed < min_request_interval_seconds:
                            time.sleep(min_request_interval_seconds - elapsed)
                    last_request_started_at = time.monotonic()
                    try:
                        decision = sanitize_decision(
                            provider.infer(
                                image_bytes,
                                image_path=image_path,
                                timestamp_seconds=float(frame["timestamp_seconds"]),
                                config=config,
                            ),
                            {"decision": config["selection"]},
                        )
                        break
                    except Exception as exc:  # noqa: BLE001
                        retry_delay = retry_delay_for_exception(exc)
                        if retry_delay is not None:
                            logging.warning("Provider rate limited; sleeping %.1fs before retry.", retry_delay)
                            time.sleep(retry_delay)
                            continue
                        fatal, error_summary = is_fatal_provider_exception(exc)
                        if fatal:
                            fatal_provider_error = error_summary
                            should_abort_run = True
                            logging.warning("Stopping infer after fatal provider error: %s", fatal_provider_error)
                            break
                        decision = provider_error_decision(f"provider_error: {error_summary}")
                        break
                if should_abort_run:
                    break
                decisions_by_frame_number[int(frame["frame_number"])] = decision
                checkpoint_handle.write(json.dumps(build_decision_record(frame, decision), ensure_ascii=False) + "\n")
                checkpoint_handle.flush()
                processed_this_run += 1
                last_completed_frame = {"frame": frame, "decision": decision}
                if processed_this_run == 1 or processed_this_run % 10 == 0:
                    progress_path = write_sync_progress(
                        index_path,
                        provider_snapshot=provider_snapshot,
                        total_candidate_frames=len(candidate_frames),
                        completed_candidate_frames=len(decisions_by_frame_number),
                        resumed_candidate_frames=len(existing_decisions),
                        last_frame=last_completed_frame,
                        status="running",
                    )
        if fatal_provider_error:
            progress_path = write_sync_progress(
                index_path,
                provider_snapshot=provider_snapshot,
                total_candidate_frames=len(candidate_frames),
                completed_candidate_frames=len(decisions_by_frame_number),
                resumed_candidate_frames=len(existing_decisions),
                last_frame=last_completed_frame,
                status="blocked",
                error=fatal_provider_error,
            )
            logging.warning("Inference stopped and checkpoint preserved at %s", checkpoint_path)
            logging.info("Progress snapshot written to %s", progress_path)
            return progress_path
        analysis_path = write_infer_outputs(
            index_path,
            extract_payload=payload,
            provider_snapshot=provider_snapshot,
            prompt_snapshot=config["prompt"],
            selection_snapshot=config["selection"],
            decisions_by_frame_number=decisions_by_frame_number,
        )
    except KeyboardInterrupt:
        progress_path = write_sync_progress(
            index_path,
            provider_snapshot=provider_snapshot,
            total_candidate_frames=len(candidate_frames),
            completed_candidate_frames=len(decisions_by_frame_number),
            resumed_candidate_frames=len(existing_decisions),
            last_frame=last_completed_frame,
            status="interrupted",
        )
        logging.warning("Inference interrupted. Resume later with the same command; checkpoint saved at %s", checkpoint_path)
        logging.info("Progress snapshot written to %s", progress_path)
        raise
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    write_sync_progress(
        index_path,
        provider_snapshot=provider_snapshot,
        total_candidate_frames=len(candidate_frames),
        completed_candidate_frames=len(decisions_by_frame_number),
        resumed_candidate_frames=len(existing_decisions),
        last_frame=last_completed_frame,
        status="completed",
        analysis_path=analysis_path,
    )
    return analysis_path


def collect_gemini_batch_results(
    manifest_path: Path,
    *,
    provider: AsyncBatchVisionProvider,
) -> Path:
    results_path = provider.collect_batch(manifest_path)
    if results_path.resolve() == manifest_path.resolve():
        return manifest_path

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index_path = Path(str(manifest["source_extract_index"])).expanduser().resolve()
    extract_payload = json.loads(index_path.read_text(encoding="utf-8"))
    selection_snapshot = dict(manifest["config_snapshot"]["selection"])
    decisions_by_frame_number = parse_gemini_batch_results(
        results_path,
        selection_snapshot=selection_snapshot,
    )
    for frame in extract_payload["frames"]:
        if frame.get("candidate") and frame.get("image_path"):
            frame_number = int(frame["frame_number"])
            if frame_number not in decisions_by_frame_number:
                decisions_by_frame_number[frame_number] = provider_error_decision("provider_error: batch_missing_response")

    analysis_path = write_infer_outputs(
        index_path,
        extract_payload=extract_payload,
        provider_snapshot=dict(manifest["provider_snapshot"]),
        prompt_snapshot=dict(manifest["config_snapshot"]["prompt"]),
        selection_snapshot=selection_snapshot,
        decisions_by_frame_number=decisions_by_frame_number,
    )
    manifest["status"] = "collected"
    manifest["analysis_path"] = str(analysis_path)
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Updated %s", manifest_path)
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
        video_output_dir = resolve_video_output_dir(config, args.output_root, video_path)
        video_output_dir.parent.mkdir(parents=True, exist_ok=True)
        extract_candidates_for_video(
            video_path,
            output_root=video_output_dir.parent,
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
    provider_selection = build_provider(provider_config)
    provider = provider_selection.provider
    logging.info(
        "Using %s provider via %s (%s, %s).",
        provider_selection.provider_type,
        provider_selection.route,
        provider_selection.profile,
        provider_selection.execution_mode,
    )
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
        if provider_selection.execution_mode == "async_batch":
            if not hasattr(provider, "submit_batch"):
                raise RuntimeError("Selected provider does not implement async batch submission.")
            manifest_path = provider.submit_batch(
                index_path=index_path,
                provider_snapshot=provider_selection.snapshot,
                prompt_snapshot=config["prompt"],
                selection_snapshot=config["selection"],
                force_resubmit=bool(getattr(args, "force_resubmit", False)),
            )
            logging.info("Async batch submitted. Later collect with: python pipeline.py collect --manifest \"%s\"", manifest_path)
            continue
        try:
            infer_from_extract_index(
                index_path,
                provider=provider,
                provider_snapshot=provider_selection.snapshot,
                config=config,
                restart=bool(getattr(args, "restart", False)),
            )
        except KeyboardInterrupt:
            return 130
    return 0


def command_collect(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)
    config = load_pipeline_config(Path(args.config).expanduser())
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provider_snapshot = manifest.get("provider_snapshot", {})
    route = str(provider_snapshot.get("selected_route", "")).strip().lower()
    if route != "gemini":
        raise ValueError(f"Collect currently supports Gemini async manifests only, got route={route!r}")

    provider_config = copy.deepcopy(config["provider"])
    provider_config["routing"] = "gemini"
    provider_config["submission_mode"] = "async"
    provider_selection = build_provider(provider_config)
    if provider_selection.execution_mode != "async_batch" or not hasattr(provider_selection.provider, "collect_batch"):
        raise RuntimeError("Gemini provider is not configured for async batch collection.")

    result_path = collect_gemini_batch_results(
        manifest_path,
        provider=provider_selection.provider,
    )
    if result_path.resolve() == manifest_path.resolve():
        logging.info("Batch job is still pending. Re-run collect later.")
    return 0


def command_cancel(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)
    config = load_pipeline_config(Path(args.config).expanduser())
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provider_snapshot = manifest.get("provider_snapshot", {})
    route = str(provider_snapshot.get("selected_route", "")).strip().lower()
    if route != "gemini":
        raise ValueError(f"Cancel currently supports Gemini async manifests only, got route={route!r}")

    provider_config = copy.deepcopy(config["provider"])
    provider_config["routing"] = "gemini"
    provider_config["submission_mode"] = "async"
    provider_selection = build_provider(provider_config)
    if provider_selection.execution_mode != "async_batch" or not hasattr(provider_selection.provider, "cancel_batch"):
        raise RuntimeError("Gemini provider is not configured for async batch cancellation.")

    provider_selection.provider.cancel_batch(manifest_path)
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


def command_run(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)
    config = load_pipeline_config(Path(args.config).expanduser())
    input_path = Path(args.video).expanduser().resolve()
    videos = list_videos(input_path, config)
    if not videos:
        raise FileNotFoundError(f"No videos found in {input_path}")

    for video_path in videos:
        video_output_dir = resolve_video_output_dir(config, args.output_root, video_path)
        video_args = argparse.Namespace(**vars(args))
        video_args.video = str(video_path)

        command_extract(video_args)
        command_infer(video_args)

        if bool(getattr(args, "skip_review", False)):
            render_input = video_output_dir / "analysis.json"
        else:
            review_args = argparse.Namespace(**vars(video_args))
            review_args.input = str(video_output_dir / "analysis.json")
            review_args.output_dir = str(video_output_dir)
            command_review(review_args)
            review_stem = review_args.stem or default_stem(float(review_args.target_seconds or config["review"]["target_seconds"]))
            render_input = video_output_dir / f"{review_stem}.editable.json"

        render_args = argparse.Namespace(**vars(video_args))
        render_args.input = str(render_input)
        render_args.output_dir = str(video_output_dir)
        command_render(render_args)
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
    if args.command == "collect":
        return command_collect(args)
    if args.command == "cancel":
        return command_cancel(args)
    if args.command == "review":
        return command_review(args)
    if args.command == "render":
        return command_render(args)
    if args.command == "run":
        return command_run(args)
    if args.command == "edit":
        if args.edit_command == "update-segment":
            return command_edit_update_segment(args)
        if args.edit_command == "update-caption":
            return command_edit_update_caption(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
