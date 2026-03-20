from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a manually editable highlight plan from a compact highlight JSON."
    )
    parser.add_argument("--input", required=True, help="Path to compact highlight JSON.")
    parser.add_argument("--output", help="Destination manual plan JSON path.")
    return parser.parse_args()


def normalize_segment(segment: dict[str, Any], index: int) -> dict[str, Any]:
    timeline_start = round(float(segment.get("timeline_start_seconds", segment.get("start_seconds", 0.0))), 3)
    timeline_end = round(float(segment.get("timeline_end_seconds", segment.get("end_seconds", timeline_start))), 3)
    source_start = round(float(segment.get("source_start_seconds", segment.get("start_seconds", timeline_start))), 3)
    source_end = round(float(segment.get("source_end_seconds", segment.get("end_seconds", timeline_end))), 3)
    duration = round(float(segment.get("duration_seconds", max(0.0, timeline_end - timeline_start))), 3)
    return {
        "rank": int(segment.get("rank", index)),
        "start_seconds": timeline_start,
        "end_seconds": timeline_end,
        "duration_seconds": duration,
        "source_start_seconds": source_start,
        "source_end_seconds": source_end,
        "score": round(float(segment.get("score", 0.0)), 3),
        "labels": [str(label).strip() for label in segment.get("labels", []) if str(label).strip()],
        "reason": str(segment.get("reason", "")).strip(),
        "caption": str(segment.get("caption", "")).strip(),
        "caption_detail": str(segment.get("caption_detail", "")).strip(),
        "notes": str(segment.get("notes", "")).strip(),
    }


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if "segments" not in payload:
        raise ValueError("Input JSON does not contain a 'segments' field.")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}.manual.json")
    )

    manual_payload = {
        "video": payload.get("video", {}),
        "source_input": str(input_path),
        "target_seconds": payload.get("actual_seconds", payload.get("target_seconds", 0.0)),
        "mode": "manual_edit_plan",
        "segments": [normalize_segment(segment, index) for index, segment in enumerate(payload["segments"], start=1)],
    }

    output_path.write_text(json.dumps(manual_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
