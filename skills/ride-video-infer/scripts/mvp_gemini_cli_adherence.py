#!/usr/bin/env python3
"""MVP: measure Gemini CLI schema adherence on a real video extract.

Calls `gemini --yolo --output-format json` with N frames per pack, repeats each
pack R times. NO retries, NO repair, NO prompt babysitting --- we want the naked
failure rate. Reference: plans/todo/PLAN_gemini_schema_harness.md Phase 0.

Note: Gemini CLI 0.27 has no --temperature flag; determinism is requested in
the prompt only. Same-pack repeat consistency in stats.json captures the
practical effect.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


def find_repo_root(start: Path) -> Path:
    for cand in [start, *start.parents]:
        if (cand / "pipeline.py").exists():
            return cand
    raise RuntimeError("Could not find repo root containing pipeline.py")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
sys.path.insert(0, str(REPO_ROOT))

from video_data_paths import infer_dir_from_index, resolve_frame_image_path  # noqa: E402


class FrameDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame_number: int
    keep: bool
    score: float = Field(ge=0.0, le=1.0)
    labels: list[str]
    reason: str = Field(max_length=160)
    discard_reason: str = Field(max_length=160)


def build_prompt(pack: list[dict[str, Any]]) -> str:
    lines = [
        "Analyze the following motorcycle riding images and return a JSON array containing your decisions.",
        "CRITICAL INSTRUCTIONS FOR AGENT:",
        "1. DO NOT USE ANY TOOLS. You already have the images attached. Output the result immediately.",
        "2. You MUST output exactly one JSON array. No other text, no markdown blocks, no conversational fillers.",
        "3. You MUST use the EXACT keys specified below. DO NOT invent your own keys.",
        "4. Be deterministic and consistent across repeated calls.",
        "",
        "Example of the EXACT required output format:",
        '[{"frame_number": 230, "keep": false, "score": 0.1, "labels": [], "reason": "", "discard_reason": "blurry"}]',
        "",
        "Each item must strictly match this schema:",
        '{"frame_number": int, "keep": bool, "score": float in [0.0, 1.0], "labels": list[str], "reason": str (<=160 chars), "discard_reason": str (<=160 chars)}',
        "",
        "Field rules:",
        "- frame_number: MUST be the integer frame number from the list below.",
        "- score: 0.0 to 1.0.",
        "- keep: true ONLY for visually strong, useful, non-duplicate frames.",
        "- reason: brief reason why kept (or empty).",
        "- discard_reason: brief reason if discarded.",
        "",
        "Images to analyze:",
    ]
    for f in pack:
        lines.append(
            f"- Frame {f['frame_number']}, timestamp {f['timestamp_seconds']}s, image @images/{Path(f['image_path']).name}"
        )
    lines.append("")
    lines.append("Output exactly one JSON array of decisions. DO NOT USE TOOLS. NOTHING ELSE.")
    return "\n".join(lines)


def call_gemini_cli(
    prompt: str, *, cwd: Path, model: str, timeout: int
) -> tuple[int | None, str, str, float, bool]:
    exe = shutil.which("gemini")
    if exe is None:
        raise RuntimeError("gemini CLI not on PATH")
    cmd = [exe, "--yolo", "--output-format", "json", "--model", model, "-p", prompt]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return proc.returncode, proc.stdout, proc.stderr, elapsed_ms, False
    except subprocess.TimeoutExpired as e:
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        out = e.stdout if isinstance(e.stdout, str) else (e.stdout.decode("utf-8", "replace") if e.stdout else "")
        err = e.stderr if isinstance(e.stderr, str) else (e.stderr.decode("utf-8", "replace") if e.stderr else "")
        return None, out, err, elapsed_ms, True


FENCE_RE = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)


def analyze_response(raw_stdout: str, expected_frame_numbers: list[int]) -> dict[str, Any]:
    """Parse + validate the CLI response. Records every failure mode. No repair."""
    result: dict[str, Any] = {
        "outer_parse_ok": False,
        "cli_response_text": None,
        "had_markdown_fence": False,
        "needed_array_extraction": False,
        "inner_parse_ok": False,
        "inner_was_array": False,
        "validated_ok": False,
        "field_errors": [],
        "missing_frames": [],
        "extra_frames": [],
        "finish_reason": None,
        "decisions": None,
    }

    # 1. CLI outer envelope
    try:
        envelope = json.loads(raw_stdout)
        result["outer_parse_ok"] = True
    except json.JSONDecodeError as e:
        result["outer_error"] = str(e)
        return result

    if isinstance(envelope, dict):
        text = envelope.get("response") or envelope.get("text") or envelope.get("content") or ""
        stats = envelope.get("stats") or {}
        result["finish_reason"] = (
            envelope.get("finish_reason")
            or stats.get("finish_reason")
            or stats.get("finishReason")
        )
    elif isinstance(envelope, list):
        text = json.dumps(envelope)
    else:
        text = str(envelope)
    result["cli_response_text"] = text

    # 2. Markdown fence
    if "```" in text:
        result["had_markdown_fence"] = True
        m = FENCE_RE.search(text)
        if m:
            text = m.group(1)

    text_stripped = text.strip()
    parsed_inner: Any = None
    try:
        parsed_inner = json.loads(text_stripped)
        result["inner_parse_ok"] = True
    except json.JSONDecodeError:
        # brute-force [...] extraction; record that we needed it
        result["needed_array_extraction"] = True
        start = text_stripped.find("[")
        end = text_stripped.rfind("]")
        if start != -1 and end > start:
            try:
                parsed_inner = json.loads(text_stripped[start : end + 1])
                result["inner_parse_ok"] = True
            except json.JSONDecodeError as e:
                result["inner_error"] = str(e)
                return result
        else:
            return result

    if not isinstance(parsed_inner, list):
        result["inner_top_type"] = type(parsed_inner).__name__
        return result
    result["inner_was_array"] = True

    # 3. Per-element Pydantic validation
    validated: list[dict[str, Any]] = []
    field_errors: list[dict[str, Any]] = []
    seen_frame_numbers: set[int] = set()
    for idx, item in enumerate(parsed_inner):
        if not isinstance(item, dict):
            field_errors.append({"index": idx, "error": "item is not an object"})
            continue
        try:
            fd = FrameDecision.model_validate(item)
            validated.append(fd.model_dump())
            seen_frame_numbers.add(fd.frame_number)
        except ValidationError as e:
            for err in e.errors():
                field_errors.append(
                    {
                        "index": idx,
                        "raw_frame_number": item.get("frame_number"),
                        "field": ".".join(str(x) for x in err.get("loc", [])),
                        "type": err.get("type"),
                        "msg": err.get("msg"),
                    }
                )

    expected_set = set(expected_frame_numbers)
    result["field_errors"] = field_errors
    result["decisions"] = validated
    result["missing_frames"] = sorted(expected_set - seen_frame_numbers)
    result["extra_frames"] = sorted(seen_frame_numbers - expected_set)
    result["validated_ok"] = (
        not field_errors
        and not result["missing_frames"]
        and not result["extra_frames"]
        and len(validated) == len(expected_frame_numbers)
    )
    return result


def make_packs(
    frames: list[dict[str, Any]], pack_size: int, num_packs: int
) -> list[list[dict[str, Any]]]:
    needed = pack_size * num_packs
    if len(frames) < needed:
        raise RuntimeError(f"Need {needed} candidate frames but only have {len(frames)}")
    selected = frames[:needed]
    return [selected[i * pack_size : (i + 1) * pack_size] for i in range(num_packs)]


def compute_stats(runs: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(runs)
    if n == 0:
        return {}

    def rate(field: str) -> float:
        return round(sum(1 for r in runs if r.get(field)) / n, 4)

    finish_dist: dict[str, int] = {}
    for r in runs:
        fr = r.get("finish_reason") or "<none>"
        finish_dist[fr] = finish_dist.get(fr, 0) + 1

    by_pack: dict[int, list[dict[str, Any]]] = {}
    for r in runs:
        by_pack.setdefault(r["pack_id"], []).append(r)

    consistency_scores: list[float] = []
    for pack_runs in by_pack.values():
        if len(pack_runs) < 2:
            continue
        votes: dict[int, list[bool]] = {}
        for r in pack_runs:
            if not r.get("validated_ok"):
                continue
            for d in r.get("decisions") or []:
                votes.setdefault(int(d["frame_number"]), []).append(bool(d["keep"]))
        per_frame_agree: list[float] = []
        for vs in votes.values():
            if len(vs) < 2:
                continue
            yes = sum(vs)
            per_frame_agree.append(max(yes, len(vs) - yes) / len(vs))
        if per_frame_agree:
            consistency_scores.append(sum(per_frame_agree) / len(per_frame_agree))

    latencies = [r["latency_ms"] for r in runs if isinstance(r.get("latency_ms"), (int, float))]
    latencies.sort()

    def pct(p: float) -> float | None:
        if not latencies:
            return None
        idx = max(0, min(len(latencies) - 1, int(round(p * (len(latencies) - 1)))))
        return round(latencies[idx], 1)

    return {
        "total_runs": n,
        "outer_parse_ok_rate": rate("outer_parse_ok"),
        "inner_parse_ok_rate": rate("inner_parse_ok"),
        "needed_array_extraction_rate": rate("needed_array_extraction"),
        "validated_ok_rate": rate("validated_ok"),
        "had_markdown_fence_rate": rate("had_markdown_fence"),
        "timeout_rate": rate("timeout"),
        "finish_reason_distribution": finish_dist,
        "mean_per_pack_keep_consistency": (
            round(sum(consistency_scores) / len(consistency_scores), 4)
            if consistency_scores
            else None
        ),
        "n_packs_with_consistency_data": len(consistency_scores),
        "latency_ms_p50": pct(0.50),
        "latency_ms_p90": pct(0.90),
        "latency_ms_p99": pct(0.99),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, help="Path to extract/index.json")
    parser.add_argument("--pack-size", type=int, default=12)
    parser.add_argument("--num-packs", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()

    index_path = Path(args.index).resolve()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    frames = [f for f in payload["frames"] if f.get("candidate") and f.get("image_path")]
    for f in frames:
        f["image_path"] = str(resolve_frame_image_path(f, index_path=index_path, payload=payload))

    packs = make_packs(frames, args.pack_size, args.num_packs)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = infer_dir_from_index(index_path) / "mvp_runs" / ts
    raw_dir = run_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    runs_path = run_root / "runs.jsonl"
    stats_path = run_root / "stats.json"
    config_path = run_root / "config.json"

    config_path.write_text(
        json.dumps(
            {
                "index": str(index_path),
                "pack_size": args.pack_size,
                "num_packs": args.num_packs,
                "repeats": args.repeats,
                "model": args.model,
                "timeout_seconds": args.timeout_seconds,
                "started_at": ts,
                "note": "Gemini CLI 0.27 has no --temperature flag; determinism asked in prompt only.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    total_calls = args.num_packs * args.repeats
    print(
        f"MVP: {args.num_packs} packs x {args.pack_size} frames x {args.repeats} repeats = {total_calls} CLI calls"
    )
    print(f"Output: {run_root}")

    runs: list[dict[str, Any]] = []
    call_idx = 0
    for pack_id, pack in enumerate(packs):
        pack_dir = run_root / "packs" / f"pack_{pack_id:03d}"
        img_dir = pack_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        for f in pack:
            src = Path(f["image_path"])
            dst = img_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
        prompt = build_prompt(pack)
        (pack_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        expected = [int(f["frame_number"]) for f in pack]

        for rep in range(args.repeats):
            call_idx += 1
            print(f"[{call_idx}/{total_calls}] pack {pack_id} rep {rep}", flush=True)
            run_record: dict[str, Any] = {
                "pack_id": pack_id,
                "repeat_idx": rep,
                "pack_size": len(pack),
                "frame_numbers": expected,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            try:
                rc, stdout, stderr, latency_ms, timed_out = call_gemini_cli(
                    prompt, cwd=pack_dir, model=args.model, timeout=args.timeout_seconds
                )
                run_record["cli_returncode"] = rc
                run_record["latency_ms"] = round(latency_ms, 1)
                run_record["timeout"] = timed_out
                raw_path = raw_dir / f"pack{pack_id:03d}_r{rep}.stdout.txt"
                raw_path.write_text(stdout, encoding="utf-8")
                run_record["raw_stdout_path"] = str(raw_path.relative_to(run_root))
                if stderr.strip():
                    err_path = raw_dir / f"pack{pack_id:03d}_r{rep}.stderr.txt"
                    err_path.write_text(stderr, encoding="utf-8")
                    run_record["raw_stderr_path"] = str(err_path.relative_to(run_root))
                analysis = analyze_response(stdout, expected)
                run_record.update(analysis)
            except Exception as e:
                run_record["error"] = repr(e)

            runs.append(run_record)
            with runs_path.open("a", encoding="utf-8") as h:
                h.write(json.dumps(run_record, ensure_ascii=False) + "\n")

    stats = compute_stats(runs)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("---STATS---")
    print(json.dumps(stats, indent=2))
    print(f"Wrote {runs_path}")
    print(f"Wrote {stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
