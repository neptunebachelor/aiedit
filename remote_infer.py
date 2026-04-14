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

TERMINAL_STATUSES = {"finished", "failed"}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post(url: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
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

    args = parser.parse_args()

    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "infer":
        cmd_infer(args)


if __name__ == "__main__":
    main()
