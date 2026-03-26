from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast


REQUIRED_JOB_KEYS = {"projectTitle", "sourceMedia", "results", "clips"}


class ReviewDevServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class: type[BaseHTTPRequestHandler],
        *,
        job_file: Path,
        state_file: Path,
        allowed_origin: str,
    ) -> None:
        super().__init__(server_address, request_handler_class)
        self.job_file = job_file
        self.state_file = state_file
        self.allowed_origin = allowed_origin


class ReviewRequestHandler(BaseHTTPRequestHandler):
    server_version = "ReviewDevServer/1.0"

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_default_headers()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/healthz":
            payload = {
                "ok": True,
                "jobFile": str(self._server.job_file),
                "stateFile": str(self._server.state_file),
                "activeSource": str(self._resolve_active_job_file()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._send_json(HTTPStatus.OK, payload)
            return

        if self.path == "/api/review-job":
            try:
                payload = self._read_review_job()
            except FileNotFoundError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "missing_job", "message": str(exc)})
                return
            except ValueError as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "invalid_job", "message": str(exc)})
                return

            self._send_json(HTTPStatus.OK, payload)
            return

        self._send_json(
            HTTPStatus.NOT_FOUND,
            {"error": "not_found", "message": f"Unknown path: {self.path}"},
        )

    def do_PUT(self) -> None:
        if self.path != "/api/review-job":
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": "not_found", "message": f"Unknown path: {self.path}"},
            )
            return

        try:
            payload = self._read_request_json()
            validate_review_job(payload)
            self._server.state_file.parent.mkdir(parents=True, exist_ok=True)
            self._server.state_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_job", "message": str(exc)})
            return
        except OSError as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "save_failed", "message": str(exc)})
            return

        response = {
            "ok": True,
            "savedTo": str(self._server.state_file),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._send_json(HTTPStatus.OK, response)

    def log_message(self, format: str, *args: Any) -> None:
        message = format % args
        print(f"[{self.log_date_time_string()}] {self.command} {self.path} -> {message}")

    @property
    def _server(self) -> ReviewDevServer:
        return cast(ReviewDevServer, self.server)

    def _send_default_headers(self) -> None:
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", self._server.allowed_origin)
        self.send_header("Access-Control-Allow-Methods", "GET, PUT, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")

    def _send_json(self, status: HTTPStatus, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self._send_default_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_request_json(self) -> Any:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            raise ValueError("Request body is required.")

        raw_body = self.rfile.read(content_length)
        try:
            return json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Request body is not valid JSON: {exc.msg}.") from exc

    def _resolve_active_job_file(self) -> Path:
        if self._server.state_file.exists():
            return self._server.state_file
        return self._server.job_file

    def _read_review_job(self) -> dict[str, Any]:
        active_file = self._resolve_active_job_file()
        if not active_file.exists():
            raise FileNotFoundError(f"Review job file does not exist: {active_file}")

        payload = json.loads(active_file.read_text(encoding="utf-8"))
        validate_review_job(payload)
        return payload


def validate_review_job(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Review job must be a JSON object.")

    missing_keys = sorted(REQUIRED_JOB_KEYS.difference(payload.keys()))
    if missing_keys:
        joined = ", ".join(missing_keys)
        raise ValueError(f"Review job is missing required keys: {joined}.")

    for key in ("sourceMedia", "results", "clips"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"Review job field '{key}' must be a JSON array.")

    if not isinstance(payload.get("projectTitle"), str) or not str(payload.get("projectTitle")).strip():
        raise ValueError("Review job field 'projectTitle' must be a non-empty string.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a lightweight review API for local runner testing.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind.")
    parser.add_argument(
        "--job-file",
        default="devdata/review-job.json",
        help="Seed review job JSON file served before any save happens.",
    )
    parser.add_argument(
        "--state-file",
        default="output/dev-review/review-job.json",
        help="Writable JSON file used to persist edits from the frontend.",
    )
    parser.add_argument(
        "--allowed-origin",
        default="*",
        help="CORS Access-Control-Allow-Origin header value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    job_file = Path(args.job_file).resolve()
    state_file = Path(args.state_file).resolve()

    if not job_file.exists():
        raise SystemExit(f"Seed review job file does not exist: {job_file}")

    validate_review_job(json.loads(job_file.read_text(encoding="utf-8")))

    server = ReviewDevServer(
        (args.host, args.port),
        ReviewRequestHandler,
        job_file=job_file,
        state_file=state_file,
        allowed_origin=args.allowed_origin,
    )

    print(f"Review dev server listening on http://{args.host}:{args.port}")
    print(f"Seed job file: {job_file}")
    print(f"State file: {state_file}")
    server.serve_forever()


if __name__ == "__main__":
    main()
