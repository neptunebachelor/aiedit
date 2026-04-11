#!/usr/bin/env python
"""Validate Gemini pack responses and optionally append them to a decisions JSONL."""

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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_decision(item: dict[str, Any]) -> dict[str, Any]:
    labels = item.get("labels") or []
    if isinstance(labels, str):
        labels = [labels]
    score = float(item.get("score", 0.0) or 0.0)
    if 1.0 < score <= 100.0:
        score /= 100.0
    return {
        "frame_number": int(item["frame_number"]),
        "keep": bool(item.get("keep", False)),
        "score": max(0.0, min(score, 1.0)),
        "labels": [str(label).strip() for label in labels if str(label).strip()],
        "reason": str(item.get("reason", "") or "").strip()[:160],
        "discard_reason": str(item.get("discard_reason", "") or "").strip()[:160],
    }


def load_response(response_path: Path) -> list[dict[str, Any]]:
    payload = read_json(response_path)
    if isinstance(payload, dict):
        for key in ("decisions", "response", "frames"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError(f"{response_path} must contain a JSON array or an object with a decisions array")
    return [normalize_decision(item) for item in payload if isinstance(item, dict)]


def validate_pack(pack_dir: Path, *, min_keep_score: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = pack_dir / "manifest.json"
    response_path = pack_dir / "response.json"
    manifest = read_json(manifest_path)
    frames = manifest.get("frames") if isinstance(manifest, dict) else manifest
    if not isinstance(frames, list):
        raise ValueError(f"{manifest_path} does not contain a frames list")
    expected = [int(frame["frame_number"]) for frame in frames]
    decisions = load_response(response_path)
    by_frame: dict[int, dict[str, Any]] = {}
    duplicates: list[int] = []
    expected_set = set(expected)
    for decision in decisions:
        frame_number = int(decision["frame_number"])
        if frame_number in by_frame:
            duplicates.append(frame_number)
        if frame_number in expected_set:
            by_frame[frame_number] = decision
    missing = [frame_number for frame_number in expected if frame_number not in by_frame]
    extra = sorted({int(decision["frame_number"]) for decision in decisions} - expected_set)
    ordered = [by_frame[frame_number] for frame_number in expected if frame_number in by_frame]
    skill_errors = [
        int(decision["frame_number"])
        for decision in ordered
        if str(decision.get("discard_reason", "")).startswith("skill_error:")
    ]
    keep_score_mismatches = [
        {
            "frame_number": int(decision["frame_number"]),
            "score": float(decision.get("score", 0.0) or 0.0),
            "keep": bool(decision.get("keep", False)),
        }
        for decision in ordered
        if bool(decision.get("keep", False)) != (float(decision.get("score", 0.0) or 0.0) >= min_keep_score)
    ]
    summary = {
        "pack_dir": str(pack_dir),
        "manifest_path": str(manifest_path),
        "response_path": str(response_path),
        "expected_count": len(expected),
        "decision_count": len(decisions),
        "valid_decision_count": len(ordered),
        "missing": missing,
        "extra": extra,
        "duplicates": sorted(duplicates),
        "skill_errors": skill_errors,
        "keep_score_mismatches": keep_score_mismatches,
        "keep_count": sum(1 for decision in ordered if decision.get("keep")),
        "valid": (
            not missing
            and not extra
            and not duplicates
            and not skill_errors
            and not keep_score_mismatches
            and len(ordered) == len(expected)
        ),
    }
    return summary, ordered


def append_jsonl(path: Path, decisions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(decision, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate one or more Gemini pack responses.")
    parser.add_argument("--pack-dir", action="append", help="Pack directory containing manifest.json and response.json")
    parser.add_argument("--packs-dir", help="Directory containing pack_XXXX children")
    parser.add_argument("--start-pack", type=int, default=1)
    parser.add_argument("--end-pack", type=int, default=0)
    parser.add_argument("--append-decisions", help="Append valid decisions to this JSONL")
    parser.add_argument("--overwrite-decisions", action="store_true", help="Delete decisions JSONL before appending")
    parser.add_argument("--summary", help="Write validation summary JSON here")
    parser.add_argument("--min-keep-score", type=float, default=0.65, help="Score threshold that keep must match")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pack_dirs: list[Path] = []
    if args.pack_dir:
        pack_dirs.extend(Path(value).expanduser().resolve() for value in args.pack_dir)
    if args.packs_dir:
        packs_dir = Path(args.packs_dir).expanduser().resolve()
        end_pack = int(args.end_pack)
        if end_pack <= 0:
            pack_dirs.extend(sorted(path for path in packs_dir.iterdir() if path.is_dir() and path.name.startswith("pack_")))
        else:
            for pack_number in range(int(args.start_pack), end_pack + 1):
                pack_dirs.append(packs_dir / f"pack_{pack_number:04d}")
    if not pack_dirs:
        raise ValueError("Provide --pack-dir or --packs-dir")

    decisions_path = Path(args.append_decisions).expanduser().resolve() if args.append_decisions else None
    if decisions_path and args.overwrite_decisions and decisions_path.exists():
        decisions_path.unlink()

    summaries: list[dict[str, Any]] = []
    all_valid_decisions: list[dict[str, Any]] = []
    for pack_dir in pack_dirs:
        summary, decisions = validate_pack(pack_dir, min_keep_score=float(args.min_keep_score))
        summaries.append(summary)
        if summary["valid"]:
            all_valid_decisions.extend(decisions)
            if decisions_path:
                append_jsonl(decisions_path, decisions)

    result = {
        "packs_checked": len(summaries),
        "valid_packs": sum(1 for summary in summaries if summary["valid"]),
        "invalid_packs": [summary["pack_dir"] for summary in summaries if not summary["valid"]],
        "valid_decisions": len(all_valid_decisions),
        "keep_count": sum(1 for decision in all_valid_decisions if decision.get("keep")),
        "decisions_path": str(decisions_path) if decisions_path else "",
        "packs": summaries,
    }
    if args.summary:
        write_json(Path(args.summary).expanduser().resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not result["invalid_packs"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
