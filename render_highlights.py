from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def parse_srt_timestamp(value: str) -> float:
    hours, minutes, seconds_ms = value.strip().split(":")
    seconds, millis = seconds_ms.split(",")
    total = (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis) / 1000.0
    )
    return round(total, 3)


def build_srt_text(
    segments: list[dict[str, Any]],
    *,
    start_key: str = "start_seconds",
    end_key: str = "end_seconds",
) -> str:
    chunks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        caption = str(segment.get("caption", "")).strip()
        caption_detail = str(segment.get("caption_detail", "")).strip()
        if caption:
            body = caption
            if caption_detail:
                body = f"{body}\n{caption_detail}"
        else:
            labels = ", ".join(segment.get("labels", [])) or "highlight"
            score = float(segment.get("score", 0.0))
            reason = str(segment.get("reason", "")).strip()
            body = f"{labels} | {score:.2f}"
            if reason:
                body = f"{body}\n{reason}"
        chunks.append(
            f"{index}\n"
            f"{format_srt_timestamp(float(segment[start_key]))} --> {format_srt_timestamp(float(segment[end_key]))}\n"
            f"{body}\n"
        )
    return "\n".join(chunks).strip() + ("\n" if chunks else "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trim highlight segments to a target duration and optionally render a final video."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to analysis.json, highlights.json, or highlights.srt",
    )
    parser.add_argument(
        "--target-seconds",
        type=float,
        default=30.0,
        help="Desired total duration of the selected highlight clips.",
    )
    parser.add_argument(
        "--clip-before",
        type=float,
        default=1.0,
        help="Seconds kept before each highlight anchor when splitting long segments.",
    )
    parser.add_argument(
        "--clip-after",
        type=float,
        default=2.0,
        help="Seconds kept after each highlight anchor when splitting long segments.",
    )
    parser.add_argument(
        "--cluster-gap",
        type=float,
        default=2.0,
        help="Maximum gap between source timestamps before they become separate clip clusters.",
    )
    parser.add_argument(
        "--min-clip-seconds",
        type=float,
        default=2.0,
        help="Minimum duration for each exported highlight clip.",
    )
    parser.add_argument(
        "--max-clip-seconds",
        type=float,
        default=6.0,
        help="Maximum duration for each exported highlight clip.",
    )
    parser.add_argument(
        "--max-clips-per-source-segment",
        type=int,
        default=2,
        help="Maximum number of selected clips taken from the same original highlight segment. Use 0 to disable.",
    )
    parser.add_argument(
        "--source-video",
        help="Source video path. Required for SRT input if the video path cannot be inferred.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory where refined JSON/SRT/video files will be written. Defaults to the source video's artifact directory.",
    )
    parser.add_argument(
        "--stem",
        default="final_highlights",
        help="Base filename used for generated outputs.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render a final MP4 by cutting and concatenating the selected clips with ffmpeg.",
    )
    parser.add_argument(
        "--ffmpeg",
        help="Path to ffmpeg executable. If omitted, the script will look for ffmpeg in PATH.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="CRF used when re-encoding intermediate clips.",
    )
    parser.add_argument(
        "--preset",
        default="fast",
        help="ffmpeg preset used when re-encoding intermediate clips.",
    )
    return parser.parse_args()


def parse_srt_segments(text: str) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    blocks = [block.strip() for block in text.replace("\r\n", "\n").split("\n\n") if block.strip()]
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        time_line_index = 1 if "-->" in lines[1] else 0
        if "-->" not in lines[time_line_index]:
            continue

        start_text, end_text = [part.strip() for part in lines[time_line_index].split("-->")]
        body_lines = lines[time_line_index + 1 :]
        first_body = body_lines[0] if body_lines else "highlight | 0.00"
        labels_text, score_text = (first_body.rsplit("|", 1) + ["0.00"])[:2]
        labels = [label.strip() for label in labels_text.split(",") if label.strip()]
        try:
            score = float(score_text.strip())
        except ValueError:
            score = 0.0
        reason = " ".join(body_lines[1:]).strip() if len(body_lines) > 1 else ""

        segments.append(
            {
                "start_seconds": parse_srt_timestamp(start_text),
                "end_seconds": parse_srt_timestamp(end_text),
                "score": score,
                "labels": labels,
                "reason": reason,
                "source_timestamps": [],
            }
        )
    return segments


def load_payload(input_path: Path, source_override: str | None) -> dict[str, Any]:
    suffix = input_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if "segments" not in payload:
            raise ValueError("Input JSON does not contain a 'segments' field.")
        video_info = dict(payload.get("video") or {})
        if source_override:
            video_info["source_path"] = str(Path(source_override).expanduser().resolve())
        return {
            "video": video_info,
            "segments": list(payload["segments"]),
        }

    if suffix == ".srt":
        return {
            "video": {
                "source_path": str(Path(source_override).expanduser().resolve()) if source_override else "",
            },
            "segments": parse_srt_segments(input_path.read_text(encoding="utf-8")),
        }

    raise ValueError(f"Unsupported input type: {input_path.suffix}")


def resolve_output_dir(input_path: Path, payload: dict[str, Any], output_override: str | None) -> Path:
    if output_override:
        return Path(output_override).expanduser().resolve()
    source_path_text = str(payload.get("video", {}).get("source_path", "")).strip()
    if source_path_text:
        source_path = Path(source_path_text).expanduser().resolve()
        return source_path.parent / source_path.stem
    return input_path.parent


def normalize_segment(segment: dict[str, Any], index: int) -> dict[str, Any]:
    start = round(float(segment["start_seconds"]), 3)
    end = round(float(segment["end_seconds"]), 3)
    source_timestamps = sorted(
        {
            round(float(value), 3)
            for value in segment.get("source_timestamps", [])
            if start <= float(value) <= end
        }
    )
    labels = [str(label).strip() for label in segment.get("labels", []) if str(label).strip()]
    reason = str(segment.get("reason", "")).strip()
    score = max(0.0, min(1.0, float(segment.get("score", 0.0))))
    return {
        "segment_index": index,
        "start_seconds": start,
        "end_seconds": end,
        "duration_seconds": round(max(0.0, end - start), 3),
        "score": score,
        "labels": labels,
        "reason": reason,
        "source_timestamps": source_timestamps,
        "hit_count": max(1, len(source_timestamps)),
    }


def cluster_timestamps(timestamps: list[float], max_gap_seconds: float) -> list[list[float]]:
    if not timestamps:
        return []
    clusters: list[list[float]] = [[timestamps[0]]]
    for timestamp in timestamps[1:]:
        if timestamp - clusters[-1][-1] <= max_gap_seconds:
            clusters[-1].append(timestamp)
        else:
            clusters.append([timestamp])
    return clusters


def clamp_window(
    start: float,
    end: float,
    *,
    lower_bound: float,
    upper_bound: float,
    anchor_seconds: float,
    min_clip_seconds: float,
    max_clip_seconds: float,
) -> tuple[float, float]:
    start = max(lower_bound, start)
    end = min(upper_bound, end)
    duration = max(0.0, end - start)

    if max_clip_seconds > 0 and duration > max_clip_seconds:
        target = max_clip_seconds
        start = max(lower_bound, anchor_seconds - (target / 2.0))
        end = min(upper_bound, start + target)
        start = max(lower_bound, end - target)

    duration = max(0.0, end - start)
    available = max(0.0, upper_bound - lower_bound)
    if min_clip_seconds > 0 and duration < min_clip_seconds and available > 0:
        target = min(min_clip_seconds, available)
        start = max(lower_bound, anchor_seconds - (target / 2.0))
        end = min(upper_bound, start + target)
        start = max(lower_bound, end - target)

    return round(start, 3), round(end, 3)


def build_candidate(
    segment: dict[str, Any],
    *,
    start_seconds: float,
    end_seconds: float,
    anchor_seconds: float,
    hit_count: int,
    candidate_index: int,
    min_clip_seconds: float,
    max_clip_seconds: float,
) -> dict[str, Any]:
    start_seconds, end_seconds = clamp_window(
        start_seconds,
        end_seconds,
        lower_bound=float(segment["start_seconds"]),
        upper_bound=float(segment["end_seconds"]),
        anchor_seconds=anchor_seconds,
        min_clip_seconds=min_clip_seconds,
        max_clip_seconds=max_clip_seconds,
    )
    duration_seconds = round(max(0.0, end_seconds - start_seconds), 3)
    priority = (
        float(segment["score"])
        + min(0.05, 0.01 * max(0, hit_count - 1))
        + min(0.03, 0.01 * max(0, len(segment["labels"]) - 1))
    )
    return {
        "candidate_index": candidate_index,
        "segment_index": segment["segment_index"],
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "duration_seconds": duration_seconds,
        "anchor_seconds": round(anchor_seconds, 3),
        "score": float(segment["score"]),
        "labels": list(segment["labels"]),
        "reason": str(segment["reason"]),
        "hit_count": int(hit_count),
        "priority": round(priority, 4),
    }


def build_candidates(
    segments: list[dict[str, Any]],
    *,
    clip_before: float,
    clip_after: float,
    cluster_gap: float,
    min_clip_seconds: float,
    max_clip_seconds: float,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    candidate_index = 0

    for segment in segments:
        source_timestamps = list(segment["source_timestamps"])
        if source_timestamps:
            for cluster in cluster_timestamps(source_timestamps, cluster_gap):
                anchor_seconds = cluster[len(cluster) // 2]
                candidate = build_candidate(
                    segment,
                    start_seconds=cluster[0] - clip_before,
                    end_seconds=cluster[-1] + clip_after,
                    anchor_seconds=anchor_seconds,
                    hit_count=len(cluster),
                    candidate_index=candidate_index,
                    min_clip_seconds=min_clip_seconds,
                    max_clip_seconds=max_clip_seconds,
                )
                candidates.append(candidate)
                candidate_index += 1
            continue

        duration = float(segment["duration_seconds"])
        if max_clip_seconds > 0 and duration > max_clip_seconds:
            window_start = float(segment["start_seconds"])
            while window_start < float(segment["end_seconds"]) - 0.001:
                window_end = min(float(segment["end_seconds"]), window_start + max_clip_seconds)
                anchor_seconds = window_start + ((window_end - window_start) / 2.0)
                candidate = build_candidate(
                    segment,
                    start_seconds=window_start,
                    end_seconds=window_end,
                    anchor_seconds=anchor_seconds,
                    hit_count=segment["hit_count"],
                    candidate_index=candidate_index,
                    min_clip_seconds=min_clip_seconds,
                    max_clip_seconds=max_clip_seconds,
                )
                candidates.append(candidate)
                candidate_index += 1
                if window_end >= float(segment["end_seconds"]):
                    break
                window_start = round(window_end, 3)
        else:
            anchor_seconds = float(segment["start_seconds"]) + (duration / 2.0)
            candidate = build_candidate(
                segment,
                start_seconds=float(segment["start_seconds"]),
                end_seconds=float(segment["end_seconds"]),
                anchor_seconds=anchor_seconds,
                hit_count=segment["hit_count"],
                candidate_index=candidate_index,
                min_clip_seconds=min_clip_seconds,
                max_clip_seconds=max_clip_seconds,
            )
            candidates.append(candidate)
            candidate_index += 1

    return candidates


def windows_overlap(first: dict[str, Any], second: dict[str, Any], tolerance: float = 0.15) -> bool:
    return min(float(first["end_seconds"]), float(second["end_seconds"])) - max(
        float(first["start_seconds"]), float(second["start_seconds"])
    ) > tolerance


def trim_candidate(candidate: dict[str, Any], target_duration: float, min_clip_seconds: float) -> dict[str, Any] | None:
    if target_duration <= 0:
        return None
    if target_duration < min_clip_seconds and float(candidate["duration_seconds"]) > target_duration:
        return None

    trimmed = dict(candidate)
    start_seconds, end_seconds = clamp_window(
        float(candidate["start_seconds"]),
        float(candidate["start_seconds"]) + target_duration,
        lower_bound=float(candidate["start_seconds"]),
        upper_bound=float(candidate["end_seconds"]),
        anchor_seconds=float(candidate["anchor_seconds"]),
        min_clip_seconds=min_clip_seconds,
        max_clip_seconds=target_duration,
    )
    trimmed["start_seconds"] = start_seconds
    trimmed["end_seconds"] = end_seconds
    trimmed["duration_seconds"] = round(max(0.0, end_seconds - start_seconds), 3)
    return trimmed


def select_candidates(
    candidates: list[dict[str, Any]],
    *,
    target_seconds: float,
    min_clip_seconds: float,
    max_clips_per_source_segment: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item["priority"]),
            -float(item["score"]),
            -int(item["hit_count"]),
            float(item["duration_seconds"]),
            float(item["start_seconds"]),
        ),
    )

    selected: list[dict[str, Any]] = []
    per_segment_counts: dict[int, int] = {}
    remaining = target_seconds

    for candidate in ranked:
        segment_index = int(candidate["segment_index"])
        if max_clips_per_source_segment > 0 and per_segment_counts.get(segment_index, 0) >= max_clips_per_source_segment:
            continue
        if any(windows_overlap(candidate, existing) for existing in selected):
            continue

        duration = float(candidate["duration_seconds"])
        chosen = candidate
        if remaining > 0 and duration > remaining:
            maybe_trimmed = trim_candidate(candidate, remaining, min_clip_seconds)
            if maybe_trimmed is None:
                continue
            chosen = maybe_trimmed
            duration = float(chosen["duration_seconds"])

        selected.append(chosen)
        per_segment_counts[segment_index] = per_segment_counts.get(segment_index, 0) + 1
        if remaining > 0:
            remaining = round(max(0.0, remaining - duration), 3)
            if remaining <= 0.05:
                break

    return sorted(selected, key=lambda item: (float(item["start_seconds"]), float(item["end_seconds"])))


def build_export_segments(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    timeline_cursor = 0.0
    for index, clip in enumerate(selected, start=1):
        duration_seconds = round(float(clip["duration_seconds"]), 3)
        timeline_start_seconds = round(timeline_cursor, 3)
        timeline_end_seconds = round(timeline_start_seconds + duration_seconds, 3)
        exported.append(
            {
                "rank": index,
                "start_seconds": timeline_start_seconds,
                "end_seconds": timeline_end_seconds,
                "duration_seconds": duration_seconds,
                "source_start_seconds": round(float(clip["start_seconds"]), 3),
                "source_end_seconds": round(float(clip["end_seconds"]), 3),
                "timeline_start_seconds": timeline_start_seconds,
                "timeline_end_seconds": timeline_end_seconds,
                "score": round(float(clip["score"]), 3),
                "labels": list(clip["labels"]),
                "reason": str(clip["reason"]),
                "hit_count": int(clip["hit_count"]),
                "anchor_seconds": round(float(clip["anchor_seconds"]), 3),
                "source_segment_index": int(clip["segment_index"]),
            }
        )
        timeline_cursor = timeline_end_seconds
    return exported


def normalize_prebuilt_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    timeline_cursor = 0.0

    for index, segment in enumerate(segments, start=1):
        source_start_seconds = round(float(segment["source_start_seconds"]), 3)
        source_end_seconds = round(float(segment["source_end_seconds"]), 3)
        if source_end_seconds <= source_start_seconds:
            raise ValueError(f"Invalid source range for segment {index}: {source_start_seconds} -> {source_end_seconds}")

        duration_seconds = round(
            float(segment.get("duration_seconds", source_end_seconds - source_start_seconds)),
            3,
        )
        start_seconds = round(float(segment.get("start_seconds", timeline_cursor)), 3)
        end_seconds = round(float(segment.get("end_seconds", start_seconds + duration_seconds)), 3)
        if end_seconds <= start_seconds:
            end_seconds = round(start_seconds + duration_seconds, 3)

        exported.append(
            {
                "rank": int(segment.get("rank", index)),
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "duration_seconds": round(end_seconds - start_seconds, 3),
                "source_start_seconds": source_start_seconds,
                "source_end_seconds": source_end_seconds,
                "timeline_start_seconds": start_seconds,
                "timeline_end_seconds": end_seconds,
                "score": round(float(segment.get("score", 1.0)), 3),
                "labels": [str(label).strip() for label in segment.get("labels", []) if str(label).strip()],
                "reason": str(segment.get("reason", "")).strip(),
                "hit_count": int(segment.get("hit_count", 1)),
                "anchor_seconds": round(float(segment.get("anchor_seconds", (source_start_seconds + source_end_seconds) / 2.0)), 3),
                "source_segment_index": int(segment.get("source_segment_index", index - 1)),
                "caption": str(segment.get("caption", "")).strip(),
                "caption_detail": str(segment.get("caption_detail", "")).strip(),
            }
        )
        timeline_cursor = end_seconds

    return exported


def find_ffmpeg(ffmpeg_override: str | None) -> str:
    if ffmpeg_override:
        path = Path(ffmpeg_override).expanduser()
        if path.exists():
            return str(path.resolve())
        raise FileNotFoundError(f"ffmpeg executable not found: {path}")

    local_tools_dir = Path(__file__).resolve().parent / ".tools" / "ffmpeg"
    if local_tools_dir.exists():
        local_matches = sorted(local_tools_dir.rglob("ffmpeg.exe"))
        if local_matches:
            return str(local_matches[0].resolve())

    detected = shutil.which("ffmpeg")
    if detected:
        return detected
    raise FileNotFoundError("ffmpeg was not found in PATH. Pass --ffmpeg C:\\path\\to\\ffmpeg.exe")


def run_ffmpeg(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def render_video(
    *,
    ffmpeg_path: str,
    source_video: Path,
    output_video: Path,
    clips: list[dict[str, Any]],
    preset: str,
    crf: int,
    scale_height: int | None = None,
) -> None:
    temp_dir = output_video.parent / f"{output_video.stem}_parts"
    temp_dir.mkdir(parents=True, exist_ok=True)

    concat_entries: list[str] = []
    for index, clip in enumerate(clips, start=1):
        clip_path = temp_dir / f"clip_{index:03d}.mp4"
        concat_entries.append(f"file '{clip_path.name}'")
        run_ffmpeg(
            [
                ffmpeg_path,
                "-y",
                "-ss",
                f"{float(clip['source_start_seconds']):.3f}",
                "-to",
                f"{float(clip['source_end_seconds']):.3f}",
                "-i",
                str(source_video),
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                *(
                    [
                        "-vf",
                        f"scale=-2:{int(scale_height)}:flags=lanczos",
                    ]
                    if scale_height and scale_height > 0
                    else []
                ),
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(clip_path),
            ]
        )

    concat_list_path = temp_dir / "concat.txt"
    concat_list_path.write_text("\n".join(concat_entries) + "\n", encoding="utf-8")
    run_ffmpeg(
        [
            ffmpeg_path,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_path.name),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_video),
        ],
        cwd=temp_dir,
    )


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    payload = load_payload(input_path, args.source_video)

    output_dir = resolve_output_dir(input_path, payload, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_segments = list(payload["segments"])
    if raw_segments and "source_start_seconds" in raw_segments[0] and "source_end_seconds" in raw_segments[0]:
        exported_segments = normalize_prebuilt_segments(raw_segments)
    else:
        segments = [normalize_segment(segment, index) for index, segment in enumerate(raw_segments)]
        candidates = build_candidates(
            segments,
            clip_before=float(args.clip_before),
            clip_after=float(args.clip_after),
            cluster_gap=float(args.cluster_gap),
            min_clip_seconds=float(args.min_clip_seconds),
            max_clip_seconds=float(args.max_clip_seconds),
        )
        selected = select_candidates(
            candidates,
            target_seconds=float(args.target_seconds),
            min_clip_seconds=float(args.min_clip_seconds),
            max_clips_per_source_segment=int(args.max_clips_per_source_segment),
        )
        exported_segments = build_export_segments(selected)

    total_duration = round(sum(float(segment["duration_seconds"]) for segment in exported_segments), 3)

    output_payload = {
        "video": payload["video"],
        "source_input": str(input_path),
        "target_seconds": round(float(args.target_seconds), 3),
        "actual_seconds": total_duration,
        "clip_before": round(float(args.clip_before), 3),
        "clip_after": round(float(args.clip_after), 3),
        "cluster_gap": round(float(args.cluster_gap), 3),
        "min_clip_seconds": round(float(args.min_clip_seconds), 3),
        "max_clip_seconds": round(float(args.max_clip_seconds), 3),
        "max_clips_per_source_segment": int(args.max_clips_per_source_segment),
        "segments": exported_segments,
    }

    json_path = output_dir / f"{args.stem}.json"
    srt_path = output_dir / f"{args.stem}.srt"
    source_srt_path = output_dir / f"{args.stem}_source.srt"
    json_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    srt_path.write_text(build_srt_text(exported_segments), encoding="utf-8")
    source_srt_path.write_text(
        build_srt_text(exported_segments, start_key="source_start_seconds", end_key="source_end_seconds"),
        encoding="utf-8",
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {srt_path}")
    print(f"Wrote {source_srt_path}")
    print(f"Selected {len(exported_segments)} clips totaling {total_duration:.3f}s")

    if args.render:
        source_path_text = str(payload["video"].get("source_path") or "").strip()
        if not source_path_text:
            raise ValueError("Rendering requires a source video path. Pass --source-video when using SRT input.")
        source_path = Path(source_path_text).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(f"Source video not found: {source_path}")
        ffmpeg_path = find_ffmpeg(args.ffmpeg)
        video_path = output_dir / f"{args.stem}.mp4"
        render_video(
            ffmpeg_path=ffmpeg_path,
            source_video=source_path.resolve(),
            output_video=video_path,
            clips=exported_segments,
            preset=str(args.preset),
            crf=int(args.crf),
        )
        print(f"Wrote {video_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
