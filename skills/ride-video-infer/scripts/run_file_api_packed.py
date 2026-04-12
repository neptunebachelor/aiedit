#!/usr/bin/env python
"""Run comparative packed inference with provider File API references."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MIN_KEEP_SCORE = 0.65


class ProviderRequestTimeoutError(TimeoutError):
    """Raised when a provider SDK request exceeds the per-pack timeout."""


def read_json(path: Path) -> Any:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{path} has a UTF-8 BOM; rewrite as UTF-8 without BOM")
    return json.loads(raw.decode("utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "file_api_infer"


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
            frame_numbers.add(int(item["frame_number"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return frame_numbers


def append_jsonl(path: Path, decisions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(decision, ensure_ascii=False) + "\n")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(message, flush=True)


def default_decisions_path(upload_manifest_path: Path, provider: str) -> Path:
    return upload_manifest_path.parent / f"{provider}_file_api.frame_decisions.jsonl"


def default_runs_dir(upload_manifest_path: Path) -> Path:
    return upload_manifest_path.parent / "file_api_runs"


def require_provider_ref(frame: dict[str, Any], provider: str) -> str:
    if provider == "openai":
        value = str(frame.get("openai_file_id", "") or "").strip()
    elif provider == "gemini":
        value = str(frame.get("gemini_file_uri", "") or "").strip()
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    if not value:
        raise ValueError(f"Frame {frame.get('frame_number')} is missing {provider} file reference")
    return value


def selected_frames_from_manifest(
    manifest: dict[str, Any],
    *,
    provider: str,
    include_all_frames: bool,
    max_frames: int,
    completed: set[int],
) -> list[dict[str, Any]]:
    frames = []
    for frame in manifest.get("frames", []):
        if not isinstance(frame, dict):
            continue
        try:
            frame_number = int(frame["frame_number"])
        except (KeyError, TypeError, ValueError):
            continue
        if frame_number in completed:
            continue
        if not include_all_frames and not bool(frame.get("candidate", False)):
            continue
        status = str(frame.get("upload_status", "") or "").strip()
        if status not in {"uploaded", "reused"}:
            continue
        require_provider_ref(frame, provider)
        frames.append(frame)
    if max_frames > 0:
        frames = frames[:max_frames]
    return frames


def make_packs(frames: list[dict[str, Any]], pack_size: int, *, start_pack: int, end_pack: int) -> list[dict[str, Any]]:
    size = max(1, int(pack_size))
    total = (len(frames) + size - 1) // size if frames else 0
    start = max(1, int(start_pack))
    end = int(end_pack) if int(end_pack) > 0 else total
    packs = []
    for pack_number in range(start, min(end, total) + 1):
        offset = (pack_number - 1) * size
        packs.append({"pack_number": pack_number, "frames": frames[offset : offset + size]})
    return packs


def build_prompt(pack: list[dict[str, Any]], *, prompt_variant: str, strict_retry: bool = False) -> str:
    max_keep = max(1, min(4, len(pack) // 5 + 1))
    variant = safe_stem(prompt_variant).lower()
    lines = [
        "Use the ride-video-infer File API Packed Inference contract.",
        "Compare these motorcycle ride candidate frames as separate uploaded file inputs.",
        "Do not assume chronological motion between still frames; judge visual highlight value frame by frame and relative to this group.",
        f"Keep only the strongest 1-{max_keep} frames unless many frames are genuinely strong.",
        f"Set keep to true exactly when score >= {MIN_KEEP_SCORE:.2f}; otherwise set keep to false.",
        "High scores: apex or lean, rapid transition, scenery reveal, nearby traffic, overtake, near pass, high speed, strong motion.",
        "Low scores: waiting, parking, boring straight, severe blur, repetitive low-value frames.",
        "",
        f"Prompt variant: {variant}",
        "Frames to analyze:",
    ]
    for frame in pack:
        lines.append(
            f"- frame_number {int(frame['frame_number'])}, timestamp {float(frame.get('timestamp_seconds', 0.0) or 0.0):.3f}s, "
            f"mime_type {str(frame.get('mime_type', '') or '')}, sha256 {str(frame.get('sha256', '') or '')}"
        )
    lines.extend(
        [
            "",
            "Return ONLY a valid JSON array. No markdown, no commentary, no code fences.",
            "Return exactly one object for every listed frame_number, with no missing or duplicate frame_number values.",
            "Object keys: frame_number, keep, score, labels, reason, discard_reason.",
            "score must be a number from 0.0 to 1.0. labels must be an array of short strings.",
            "Keep reason and discard_reason short and plain.",
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


def normalize_decision(item: dict[str, Any]) -> dict[str, Any]:
    labels = item.get("labels") or []
    if isinstance(labels, str):
        labels = [labels]
    score = float(item.get("score", 0.0) or 0.0)
    if 1.0 < score <= 100.0:
        score /= 100.0
    score = max(0.0, min(score, 1.0))
    return {
        "frame_number": int(item["frame_number"]),
        "keep": score >= MIN_KEEP_SCORE,
        "score": score,
        "labels": [str(label).strip() for label in labels if str(label).strip()],
        "reason": str(item.get("reason", "") or "").strip()[:160],
        "discard_reason": str(item.get("discard_reason", "") or "").strip()[:160],
    }


def reject_decision(frame: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "frame_number": int(frame["frame_number"]),
        "keep": False,
        "score": 0.0,
        "labels": [],
        "reason": "",
        "discard_reason": reason[:160],
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
        return parse_decision_array(value)
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
    expected_frames: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    expected_numbers = [int(frame["frame_number"]) for frame in expected_frames]
    expected_set = set(expected_numbers)
    by_frame: dict[int, dict[str, Any]] = {}
    duplicates: list[int] = []
    for item in decisions:
        try:
            normalized = normalize_decision(item)
        except (KeyError, TypeError, ValueError):
            continue
        frame_number = int(normalized["frame_number"])
        if frame_number not in expected_set:
            continue
        if frame_number in by_frame:
            duplicates.append(frame_number)
        by_frame[frame_number] = normalized
    missing = [frame_number for frame_number in expected_numbers if frame_number not in by_frame]
    ordered = [by_frame[frame_number] for frame_number in expected_numbers if frame_number in by_frame]
    return ordered, missing, duplicates


def validate_or_reject_pack(
    decisions: list[dict[str, Any]],
    expected_frames: list[dict[str, Any]],
    *,
    error_reason: str,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    normalized, missing, duplicates = validate_pack_decisions(decisions, expected_frames)
    if not missing and not duplicates:
        return normalized, missing, duplicates
    present = {int(item["frame_number"]) for item in normalized}
    for frame in expected_frames:
        if int(frame["frame_number"]) not in present:
            normalized.append(reject_decision(frame, error_reason))
    normalized.sort(key=lambda item: int(item["frame_number"]))
    return normalized, missing, duplicates


def extract_openai_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    if isinstance(response, dict):
        for key in ("output_text", "text"):
            if isinstance(response.get(key), str):
                return str(response[key])
        return json.dumps(response, ensure_ascii=False)
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        return json.dumps(model_dump(), ensure_ascii=False)
    return str(response)


def extract_gemini_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    if isinstance(response, dict):
        for key in ("text", "output_text"):
            if isinstance(response.get(key), str):
                return str(response[key])
        return json.dumps(response, ensure_ascii=False)
    return str(response)


def call_openai_pack(client: Any, *, model: str, prompt: str, pack: list[dict[str, Any]], temperature: float) -> str:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for frame in pack:
        content.append({"type": "input_image", "file_id": str(frame["openai_file_id"])})
    payload: dict[str, Any] = {
        "model": model,
        "input": [{"role": "user", "content": content}],
    }
    if temperature >= 0:
        payload["temperature"] = temperature
    return extract_openai_text(client.responses.create(**payload))


def call_gemini_pack(client: Any, *, model: str, prompt: str, pack: list[dict[str, Any]], temperature: float) -> str:
    from google.genai import types

    parts: list[Any] = [prompt]
    for frame in pack:
        parts.append(
            types.Part.from_uri(
                file_uri=str(frame["gemini_file_uri"]),
                mime_type=str(frame.get("mime_type", "") or "image/jpeg"),
            )
        )
    config = types.GenerateContentConfig(temperature=temperature, response_mime_type="application/json")
    response = client.models.generate_content(model=model, contents=parts, config=config)
    return extract_gemini_text(response)


def call_provider_pack(
    client: Any,
    *,
    provider: str,
    model: str,
    prompt: str,
    pack: list[dict[str, Any]],
    temperature: float,
) -> str:
    if provider == "openai":
        return call_openai_pack(client, model=model, prompt=prompt, pack=pack, temperature=temperature)
    if provider == "gemini":
        return call_gemini_pack(client, model=model, prompt=prompt, pack=pack, temperature=temperature)
    raise ValueError(f"Unsupported provider: {provider}")


def call_provider_pack_with_timeout(
    client: Any,
    *,
    provider: str,
    model: str,
    prompt: str,
    pack: list[dict[str, Any]],
    temperature: float,
    request_timeout: int = 180,
) -> str:
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def target() -> None:
        try:
            output = call_provider_pack(
                client,
                provider=provider,
                model=model,
                prompt=prompt,
                pack=pack,
                temperature=temperature,
            )
            result_queue.put(("ok", output))
        except Exception as exc:  # Propagate provider SDK exceptions to the main thread.
            result_queue.put(("error", exc))

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    timeout_seconds = max(1, int(request_timeout))
    thread.join(timeout_seconds)
    if thread.is_alive():
        raise ProviderRequestTimeoutError(f"provider request exceeded {timeout_seconds} seconds")

    status, payload = result_queue.get()
    if status == "ok":
        return str(payload)
    raise payload


def make_client(provider: str) -> Any:
    if provider == "openai":
        from openai import OpenAI

        return OpenAI()
    if provider == "gemini":
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY", "").strip() or None
        return genai.Client(api_key=api_key)
    raise ValueError(f"Unsupported provider: {provider}")


def run_apply_decisions(
    *,
    source_extract_index: str,
    decisions_path: Path,
    config: str,
    provider: str,
) -> dict[str, Any]:
    script_path = Path(__file__).with_name("apply_decisions.py")
    cmd = [
        sys.executable,
        str(script_path),
        "--index",
        str(Path(source_extract_index).expanduser().resolve()),
        "--decisions",
        str(decisions_path),
        "--config",
        str(config),
        "--provider",
        f"{provider}_file_api",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def build_run_manifest(
    *,
    upload_manifest_path: Path,
    upload_manifest: dict[str, Any],
    provider: str,
    model: str,
    pack_size: int,
    start_pack: int,
    end_pack: int,
    max_frames: int,
    prompt_variant: str,
    temperature: float,
    request_timeout: int = 180,
    include_all_frames: bool,
    restart: bool,
    dry_run: bool,
    decisions_path: Path,
    run_dir: Path,
    packs: list[dict[str, Any]],
) -> dict[str, Any]:
    prompts = [build_prompt(pack["frames"], prompt_variant=prompt_variant) for pack in packs]
    candidate_frame_numbers = [int(frame["frame_number"]) for pack in packs for frame in pack["frames"]]
    return {
        "stage": "infer.file_api_packed",
        "provider": provider,
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_extract_index": str(upload_manifest.get("source_extract_index", "") or ""),
        "upload_manifest_path": str(upload_manifest_path),
        "upload_manifest_hash": sha256_file(upload_manifest_path),
        "prompt_variant": safe_stem(prompt_variant).lower(),
        "prompt_hash": sha256_text("\n---PACK---\n".join(prompts)),
        "pack_size": int(pack_size),
        "start_pack": int(start_pack),
        "end_pack": int(end_pack),
        "max_frames": int(max_frames),
        "temperature": float(temperature),
        "request_timeout_seconds": int(request_timeout),
        "min_keep_score": MIN_KEEP_SCORE,
        "retry_policy": "retry once with strict prompt, then emit skill_error rejects for missing frames",
        "include_all_frames": bool(include_all_frames),
        "restart": bool(restart),
        "dry_run": bool(dry_run),
        "decisions_path": str(decisions_path),
        "run_dir": str(run_dir),
        "pack_count": len(packs),
        "candidate_frame_numbers": candidate_frame_numbers,
        "packs": [
            {
                "pack_number": int(pack["pack_number"]),
                "frame_numbers": [int(frame["frame_number"]) for frame in pack["frames"]],
            }
            for pack in packs
        ],
    }


def write_pack_state(
    pack_dir: Path,
    *,
    pack_number: int,
    pack_frames: list[dict[str, Any]],
    provider: str,
    model: str,
    attempt: str,
    request_timeout: int = 180,
) -> None:
    write_json(
        pack_dir / "pack_state.json",
        {
            "pack_number": int(pack_number),
            "frame_numbers": [int(frame["frame_number"]) for frame in pack_frames],
            "start_time": utc_now_iso(),
            "provider": provider,
            "model": model,
            "attempt": attempt,
            "request_timeout_seconds": int(request_timeout),
        },
    )


def write_provider_error(pack_dir: Path, *, reason: str, exc: BaseException | None = None) -> None:
    lines = [reason]
    if exc is not None:
        lines.extend(["", "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()])
    (pack_dir / "raw_response.error.txt").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def run_packed_inference(
    *,
    upload_manifest_path: Path,
    provider: str | None,
    model: str,
    pack_size: int,
    start_pack: int,
    end_pack: int,
    max_frames: int,
    prompt_variant: str,
    temperature: float,
    request_timeout: int = 180,
    output_decisions: Path | None,
    include_all_frames: bool,
    restart: bool,
    dry_run: bool,
    apply_outputs: bool,
    config: str,
    client: Any | None = None,
) -> dict[str, Any]:
    upload_manifest_path = upload_manifest_path.expanduser().resolve()
    upload_manifest = read_json(upload_manifest_path)
    resolved_provider = (provider or str(upload_manifest.get("provider", "") or "")).strip().lower()
    if resolved_provider not in {"openai", "gemini"}:
        raise ValueError("--provider must be openai or gemini")
    decisions_path = output_decisions.expanduser().resolve() if output_decisions else default_decisions_path(upload_manifest_path, resolved_provider)
    if restart and decisions_path.exists():
        decisions_path.unlink()
    completed = load_existing_frame_numbers(decisions_path, restart=restart)
    frames = selected_frames_from_manifest(
        upload_manifest,
        provider=resolved_provider,
        include_all_frames=include_all_frames,
        max_frames=max_frames,
        completed=completed,
    )
    packs = make_packs(frames, pack_size, start_pack=start_pack, end_pack=end_pack)
    run_name = f"{resolved_provider}_{safe_stem(prompt_variant)}_{int(time.time())}"
    run_dir = default_runs_dir(upload_manifest_path) / run_name
    run_manifest = build_run_manifest(
        upload_manifest_path=upload_manifest_path,
        upload_manifest=upload_manifest,
        provider=resolved_provider,
        model=model,
        pack_size=pack_size,
        start_pack=start_pack,
        end_pack=end_pack,
        max_frames=max_frames,
        prompt_variant=prompt_variant,
        temperature=temperature,
        request_timeout=request_timeout,
        include_all_frames=include_all_frames,
        restart=restart,
        dry_run=dry_run,
        decisions_path=decisions_path,
        run_dir=run_dir,
        packs=packs,
    )
    write_json(run_dir / "run_manifest.json", run_manifest)

    if dry_run:
        result = {
            "status": "dry_run",
            "run_manifest_path": str(run_dir / "run_manifest.json"),
            "decisions_path": str(decisions_path),
            "pack_count": len(packs),
            "frame_count": sum(len(pack["frames"]) for pack in packs),
        }
        write_json(run_dir / "summary.json", result)
        return result

    inference_client = client if client is not None else make_client(resolved_provider)
    processed = 0
    kept = 0
    failed_packs = 0
    for pack in packs:
        pack_number = int(pack["pack_number"])
        pack_frames = pack["frames"]
        pack_dir = run_dir / f"pack_{pack_number:04d}"
        pack_dir.mkdir(parents=True, exist_ok=True)
        prompt = build_prompt(pack_frames, prompt_variant=prompt_variant)
        frame_numbers = [int(frame["frame_number"]) for frame in pack_frames]
        log(f"pack {pack_number}/{len(packs)}: starting {len(pack_frames)} frames {frame_numbers}")
        write_pack_state(
            pack_dir,
            pack_number=pack_number,
            pack_frames=pack_frames,
            provider=resolved_provider,
            model=model,
            attempt="initial",
            request_timeout=request_timeout,
        )
        log(
            f"pack {pack_number}/{len(packs)}: requesting provider={resolved_provider} "
            f"model={model} timeout={int(request_timeout)}s"
        )
        try:
            output = call_provider_pack_with_timeout(
                inference_client,
                provider=resolved_provider,
                model=model,
                prompt=prompt,
                pack=pack_frames,
                temperature=temperature,
                request_timeout=request_timeout,
            )
        except ProviderRequestTimeoutError as exc:
            failed_packs += 1
            write_provider_error(pack_dir, reason=f"provider_error: timeout: {exc}", exc=exc)
            missing = frame_numbers
            duplicates: list[int] = []
            normalized = [reject_decision(frame, "provider_error: timeout") for frame in pack_frames]
            append_jsonl(decisions_path, normalized)
            keep_count = sum(1 for item in normalized if item.get("keep"))
            write_json(pack_dir / "summary.json", {"pack_number": pack_number, "frame_count": len(pack_frames), "decision_count": len(normalized), "missing": missing, "duplicates": duplicates, "keep_count": keep_count, "error": "provider_error: timeout"})
            processed += len(normalized)
            kept += keep_count
            log(f"pack {pack_number}/{len(packs)} complete: decisions={len(normalized)} keep={keep_count} missing={missing} duplicates={duplicates}")
            continue
        except Exception as exc:
            failed_packs += 1
            write_provider_error(pack_dir, reason=f"provider_error: exception: {type(exc).__name__}: {exc}", exc=exc)
            missing = frame_numbers
            duplicates = []
            normalized = [reject_decision(frame, f"provider_error: exception {type(exc).__name__}") for frame in pack_frames]
            append_jsonl(decisions_path, normalized)
            keep_count = sum(1 for item in normalized if item.get("keep"))
            write_json(pack_dir / "summary.json", {"pack_number": pack_number, "frame_count": len(pack_frames), "decision_count": len(normalized), "missing": missing, "duplicates": duplicates, "keep_count": keep_count, "error": f"provider_error: exception {type(exc).__name__}"})
            processed += len(normalized)
            kept += keep_count
            log(f"pack {pack_number}/{len(packs)} complete: decisions={len(normalized)} keep={keep_count} missing={missing} duplicates={duplicates}")
            continue
        (pack_dir / "raw_response.txt").write_text(output, encoding="utf-8")
        decisions = parse_decision_array(output) or []
        normalized, missing, duplicates = validate_pack_decisions(decisions, pack_frames)
        if missing or duplicates:
            retry_prompt = build_prompt(pack_frames, prompt_variant=prompt_variant, strict_retry=True)
            write_pack_state(
                pack_dir,
                pack_number=pack_number,
                pack_frames=pack_frames,
                provider=resolved_provider,
                model=model,
                attempt="retry",
                request_timeout=request_timeout,
            )
            log(
                f"pack {pack_number}/{len(packs)}: retrying provider={resolved_provider} "
                f"model={model} timeout={int(request_timeout)}s missing={missing} duplicates={duplicates}"
            )
            try:
                retry_output = call_provider_pack_with_timeout(
                    inference_client,
                    provider=resolved_provider,
                    model=model,
                    prompt=retry_prompt,
                    pack=pack_frames,
                    temperature=temperature,
                    request_timeout=request_timeout,
                )
            except ProviderRequestTimeoutError as exc:
                failed_packs += 1
                write_provider_error(pack_dir, reason=f"provider_error: timeout during retry: {exc}", exc=exc)
                missing = frame_numbers
                duplicates = []
                normalized = [reject_decision(frame, "provider_error: timeout") for frame in pack_frames]
                append_jsonl(decisions_path, normalized)
                keep_count = sum(1 for item in normalized if item.get("keep"))
                write_json(pack_dir / "summary.json", {"pack_number": pack_number, "frame_count": len(pack_frames), "decision_count": len(normalized), "missing": missing, "duplicates": duplicates, "keep_count": keep_count, "error": "provider_error: timeout"})
                processed += len(normalized)
                kept += keep_count
                log(f"pack {pack_number}/{len(packs)} complete: decisions={len(normalized)} keep={keep_count} missing={missing} duplicates={duplicates}")
                continue
            except Exception as exc:
                failed_packs += 1
                write_provider_error(pack_dir, reason=f"provider_error: exception during retry: {type(exc).__name__}: {exc}", exc=exc)
                missing = frame_numbers
                duplicates = []
                normalized = [reject_decision(frame, f"provider_error: exception {type(exc).__name__}") for frame in pack_frames]
                append_jsonl(decisions_path, normalized)
                keep_count = sum(1 for item in normalized if item.get("keep"))
                write_json(pack_dir / "summary.json", {"pack_number": pack_number, "frame_count": len(pack_frames), "decision_count": len(normalized), "missing": missing, "duplicates": duplicates, "keep_count": keep_count, "error": f"provider_error: exception {type(exc).__name__}"})
                processed += len(normalized)
                kept += keep_count
                log(f"pack {pack_number}/{len(packs)} complete: decisions={len(normalized)} keep={keep_count} missing={missing} duplicates={duplicates}")
                continue
            (pack_dir / "raw_response.retry.txt").write_text(retry_output, encoding="utf-8")
            decisions = parse_decision_array(retry_output) or []
            normalized, missing, duplicates = validate_pack_decisions(decisions, pack_frames)
        if missing or duplicates:
            failed_packs += 1
            normalized, missing, duplicates = validate_or_reject_pack(
                normalized,
                pack_frames,
                error_reason="skill_error: missing or duplicate frame in file api output",
            )
        append_jsonl(decisions_path, normalized)
        keep_count = sum(1 for item in normalized if item.get("keep"))
        write_json(pack_dir / "summary.json", {"pack_number": pack_number, "frame_count": len(pack_frames), "decision_count": len(normalized), "missing": missing, "duplicates": duplicates, "keep_count": keep_count})
        processed += len(normalized)
        kept += keep_count
        log(f"pack {pack_number}/{len(packs)} complete: decisions={len(normalized)} keep={keep_count} missing={missing} duplicates={duplicates}")

    result: dict[str, Any] = {
        "status": "done",
        "run_manifest_path": str(run_dir / "run_manifest.json"),
        "decisions_path": str(decisions_path),
        "processed_frames": processed,
        "kept_frames": kept,
        "failed_packs": failed_packs,
        "pack_size": int(pack_size),
    }
    if apply_outputs:
        result["apply"] = run_apply_decisions(source_extract_index=str(upload_manifest.get("source_extract_index", "") or ""), decisions_path=decisions_path, config=config, provider=resolved_provider)
    write_json(run_dir / "summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run packed inference over provider File API image references.")
    parser.add_argument("--upload-manifest", required=True, help="Path to file_uploads.<provider>.json")
    parser.add_argument("--provider", choices=["openai", "gemini"], help="Defaults to the provider in the upload manifest")
    parser.add_argument("--model", required=True)
    parser.add_argument("--pack-size", type=int, default=20)
    parser.add_argument("--start-pack", type=int, default=1)
    parser.add_argument("--end-pack", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--prompt-variant", default="default")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--request-timeout", type=int, default=180, help="Timeout in seconds for each provider request.")
    parser.add_argument("--output-decisions", help="Destination decisions JSONL")
    parser.add_argument("--include-all-frames", action="store_true", help="Infer all uploaded frames instead of candidate frames only")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Run apply_decisions.py after successful inference")
    parser.add_argument("--config", default="config.toml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_packed_inference(
        upload_manifest_path=Path(args.upload_manifest),
        provider=args.provider,
        model=str(args.model),
        pack_size=int(args.pack_size),
        start_pack=int(args.start_pack),
        end_pack=int(args.end_pack),
        max_frames=int(args.max_frames),
        prompt_variant=str(args.prompt_variant),
        temperature=float(args.temperature),
        request_timeout=int(args.request_timeout),
        output_decisions=Path(args.output_decisions) if args.output_decisions else None,
        include_all_frames=bool(args.include_all_frames),
        restart=bool(args.restart),
        dry_run=bool(args.dry_run),
        apply_outputs=bool(args.apply),
        config=str(args.config),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    apply_result = result.get("apply")
    if isinstance(apply_result, dict) and int(apply_result.get("returncode", 0)) != 0:
        return int(apply_result.get("returncode", 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
