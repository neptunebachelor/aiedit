"""
Remote pipeline client — run on Pi / Mac to trigger jobs on the home PC over LAN.

Usage:
    # Extract frames from a video
    python remote_infer.py extract \\
        --host 192.168.1.100:8765 \\
        --video "/abs/path/on/pc/videos/ride01.mp4"

    # Run inference on extracted frames
    python remote_infer.py infer \\
        --host 192.168.1.100:8765 \\
        --index "/abs/path/on/pc/.video_data/videos/<slug>/extract/index.json" \\
        [--shutdown]

No external dependencies — stdlib only (urllib, json, argparse).
Progress is streamed via Server-Sent Events.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

TERMINAL_STATUSES = {"finished", "failed"}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post(url: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def _fmt_eta(seconds: int | None) -> str:
    if seconds is None:
        return ""
    if seconds < 60:
        return f"  ETA {seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"  ETA {m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"  ETA {h}h{m:02d}m"


def _stream_sse(url: str, label: str) -> None:
    """Open an SSE stream and print progress until a terminal status is received."""
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").rstrip("\r\n")
                if not line.startswith("data:"):
                    continue
                payload = json.loads(line[5:].strip())
                job_status = payload.get("status", "?")
                stage = payload.get("stage") or "?"
                progress = payload.get("progress", 0)
                eta = _fmt_eta(payload.get("eta_seconds"))
                print(
                    f"\r[{job_status}]  stage={stage:<12} progress={progress:>3}%{eta:<12}   ",
                    end="",
                    flush=True,
                )
                if job_status in TERMINAL_STATUSES:
                    print()
                    if job_status == "finished":
                        print(f"{label} completed successfully.")
                        result = payload.get("result") or {}
                        if "index_path" in result:
                            print(f"index_path: {result['index_path']}")
                        sys.exit(0)
                    else:
                        error = payload.get("error") or "unknown error"
                        print(f"{label} FAILED: {error}")
                        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"\nERROR: Lost SSE connection: {exc.reason}")
        sys.exit(1)


def _submit_and_stream(base: str, endpoint: str, body: dict, label: str) -> None:
    print(f"Submitting {label.lower()} job to {base} ...")
    try:
        result = _post(f"{base}/{endpoint}", body)
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode(errors="replace")
        print(f"ERROR: Server returned {exc.code}: {body_text}")
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"ERROR: Cannot reach server at {base}: {exc.reason}")
        sys.exit(1)

    job_id = result["job_id"]
    print(f"Job submitted — id: {job_id}")
    print()

    try:
        _stream_sse(f"{base}/sse/jobs/{job_id}", label)
    except KeyboardInterrupt:
        print(f"\nInterrupted. Job is still running on the PC (id: {job_id}).")
        sys.exit(0)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_extract(args: argparse.Namespace) -> None:
    _submit_and_stream(
        base=f"http://{args.host}",
        endpoint="extract-jobs",
        body={"video_path": args.video},
        label="Extract",
    )


def cmd_infer(args: argparse.Namespace) -> None:
    if args.shutdown:
        print("NOTE: PC will shut down after infer completes.")
    _submit_and_stream(
        base=f"http://{args.host}",
        endpoint="infer-jobs",
        body={"index_path": args.index, "shutdown": args.shutdown},
        label="Infer",
    )


def cmd_ls(args: argparse.Namespace) -> None:
    base = f"http://{args.host}"
    try:
        data = _get(f"{base}/ls/videos")
    except urllib.error.URLError as exc:
        print(f"ERROR: Cannot reach server at {base}: {exc.reason}")
        sys.exit(1)

    videos_dir = data.get("videos_dir", "None")
    videos = data.get("videos", [])

    if videos_dir == "None" or not videos:
        if videos_dir == "None":
            print("WARNING: Server has no --videos-dir configured.")
        else:
            print(f"Remote videos dir: {videos_dir}")
            print("No video files found.")
        return

    print(f"Remote videos dir: {videos_dir}")
    print()
    name_w = max(len(v["name"]) for v in videos)
    name_w = max(name_w, 4)
    print(f"{'NAME':<{name_w}}  {'SIZE':>10}  PATH")
    print("-" * (name_w + 2 + 10 + 2 + 60))
    for v in videos:
        print(f"{v['name']:<{name_w}}  {str(v['size_mb']) + ' MB':>10}  {v['path']}")


def cmd_ls_local(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else Path.cwd() / ".video_data"
    videos_dir = data_root / "videos"

    print(f"Local data root: {data_root}")

    if not videos_dir.is_dir():
        print("No .video_data/videos/ directory found.")
        return

    entries = []
    for slug_dir in sorted(videos_dir.iterdir()):
        if not slug_dir.is_dir():
            continue
        index_path = slug_dir / "extract" / "index.json"
        infer_dir = slug_dir / "infer"
        if not index_path.is_file():
            continue

        source_path = "-"
        frame_count = "-"
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            source_path = (payload.get("video") or {}).get("source_path") or "-"
            frames = payload.get("frames") or []
            frame_count = str(len(frames))
        except Exception:
            pass

        has_infer = infer_dir.is_dir() and any(infer_dir.iterdir())
        entries.append((slug_dir.name, frame_count, "yes" if has_infer else "no", source_path))

    if not entries:
        print("No extracted videos found.")
        return

    slug_w = max(len(e[0]) for e in entries)
    slug_w = max(slug_w, 4)
    print()
    print(f"{'SLUG':<{slug_w}}  {'FRAMES':>6}  {'INFER':>5}  SOURCE")
    print("-" * (slug_w + 2 + 6 + 2 + 5 + 2 + 60))
    for slug, frames, infer, source in entries:
        print(f"{slug:<{slug_w}}  {frames:>6}  {infer:>5}  {source}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trigger remote pipeline jobs on the PC over LAN."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # extract subcommand
    p_extract = sub.add_parser("extract", help="Extract frames from a video on the PC")
    p_extract.add_argument("--host", required=True, help="PC address:port, e.g. 192.168.1.100:8765")
    p_extract.add_argument("--video", required=True, help="Absolute path to video ON THE PC")

    # infer subcommand
    p_infer = sub.add_parser("infer", help="Run inference on extracted frames on the PC")
    p_infer.add_argument("--host", required=True, help="PC address:port, e.g. 192.168.1.100:8765")
    p_infer.add_argument("--index", required=True, help="Absolute path to extract/index.json ON THE PC")
    p_infer.add_argument("--shutdown", action="store_true", help="Shut down the PC after infer completes")

    # ls subcommand
    p_ls = sub.add_parser("ls", help="List raw video files on the remote PC")
    p_ls.add_argument("--host", required=True, help="PC address:port, e.g. 192.168.1.100:8765")

    # ls-local subcommand
    p_ls_local = sub.add_parser("ls-local", help="List extracted/inferred videos in local .video_data")
    p_ls_local.add_argument("--data-root", default=None,
                             help="Path to .video_data root (default: ./.video_data)")

    args = parser.parse_args()

    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "infer":
        cmd_infer(args)
    elif args.command == "ls":
        cmd_ls(args)
    elif args.command == "ls-local":
        cmd_ls_local(args)


if __name__ == "__main__":
    main()
