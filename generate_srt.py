from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_srt_entry(index: int, segment: dict[str, Any]) -> str:
    start = format_srt_timestamp(float(segment["start_seconds"]))
    end = format_srt_timestamp(float(segment["end_seconds"]))
    labels = ", ".join(segment.get("labels", [])) or "highlight"
    score = float(segment.get("score", 0.0))
    reason = str(segment.get("reason", "")).strip()
    body = f"{labels} | {score:.2f}"
    if reason:
        body = f"{body}\n{reason}"
    return f"{index}\n{start} --> {end}\n{body}\n"


def generate_srt_text(segments: list[dict[str, Any]]) -> str:
    chunks = [build_srt_entry(index, segment) for index, segment in enumerate(segments, start=1)]
    return "\n".join(chunks).strip() + ("\n" if chunks else "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate highlights.srt from highlights.json.")
    parser.add_argument("--input", required=True, help="Path to highlights.json or analysis.json")
    parser.add_argument("--output", help="Destination .srt path")
    return parser.parse_args()


def load_segments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "segments" in payload:
        return list(payload["segments"])
    raise ValueError("Input JSON does not contain a 'segments' field.")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else input_path.with_suffix(".srt")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    segments = load_segments(payload)
    output_path.write_text(generate_srt_text(segments), encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
