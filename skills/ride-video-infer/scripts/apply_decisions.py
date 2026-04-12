#!/usr/bin/env python
"""Apply Codex/Gemini CLI frame decisions to this project's pipeline outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pipeline.py").exists():
            return candidate
    raise RuntimeError("Could not find project root containing pipeline.py")


def load_decision_items(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("Decision JSON array must contain objects")
        return [item for item in payload if isinstance(item, dict)]

    items: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"Line {line_number} is not a JSON object")
        items.append(item)
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize pipeline outputs from external frame decisions.")
    parser.add_argument("--index", required=True, help="Path to extract/index.json")
    parser.add_argument("--decisions", required=True, help="JSONL or JSON array with frame_number decisions")
    parser.add_argument("--config", default="config.toml", help="Pipeline config path")
    parser.add_argument("--provider", default="codex", help="Provider label for output metadata")
    parser.add_argument("--restart", action="store_true", help="Replace existing checkpoint instead of merging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index_path = Path(args.index).expanduser().resolve()
    decisions_path = Path(args.decisions).expanduser().resolve()
    repo_root = find_repo_root(Path(__file__).resolve())
    sys.path.insert(0, str(repo_root))

    import pipeline  # pylint: disable=import-error,import-outside-toplevel

    config = pipeline.load_pipeline_config(Path(args.config).expanduser())
    extract_payload = json.loads(index_path.read_text(encoding="utf-8"))
    candidate_frame_numbers = {
        int(frame["frame_number"])
        for frame in extract_payload["frames"]
        if frame.get("candidate") and frame.get("image_path")
    }

    existing = {} if args.restart else pipeline.load_checkpoint_decisions(index_path)
    decisions_by_frame_number: dict[int, dict[str, Any]] = dict(existing)
    added = 0
    for item in load_decision_items(decisions_path):
        if "frame_number" not in item:
            raise ValueError(f"Decision missing frame_number: {item}")
        frame_number = int(item["frame_number"])
        if frame_number not in candidate_frame_numbers:
            continue
        decision = pipeline.sanitize_decision(item, {"decision": config["selection"]})
        decisions_by_frame_number[frame_number] = decision
        added += 1

    checkpoint_path = pipeline.checkpoint_path_for_index(index_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("w", encoding="utf-8") as handle:
        for frame_number in sorted(decisions_by_frame_number):
            record = {"frame_number": frame_number, **decisions_by_frame_number[frame_number]}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    provider_label = str(args.provider).strip() or "external_skill"
    provider_snapshot = {
        "routing": provider_label,
        "selected_route": provider_label,
        "selected_provider_type": provider_label,
        provider_label: {
            "enabled": True,
            "supports_vision": True,
            "execution": "skill",
        },
    }
    analysis_path = pipeline.write_infer_outputs(
        index_path,
        extract_payload=extract_payload,
        provider_snapshot=provider_snapshot,
        prompt_snapshot=config["prompt"],
        selection_snapshot=config["selection"],
        decisions_by_frame_number=decisions_by_frame_number,
    )
    kept = sum(1 for decision in decisions_by_frame_number.values() if decision.get("keep"))
    print(
        json.dumps(
            {
                "analysis_path": str(analysis_path),
                "checkpoint_path": str(checkpoint_path),
                "candidate_frames": len(candidate_frame_numbers),
                "decided_frames": len(decisions_by_frame_number),
                "added_or_updated_frames": added,
                "kept_frames": kept,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
