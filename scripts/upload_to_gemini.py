from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from gemini_files_common import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_JOBS_DIR,
    SOURCE_NAME,
    copy_original_file,
    detect_mime_type,
    fail,
    get_value,
    make_client,
    make_job_id,
    normalize_state,
    now_iso,
    print_json,
    resolve_existing_file,
    to_jsonable,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a local file to the Gemini Files API and write a manifest.")
    parser.add_argument("--file", required=True, help="Local file to upload.")
    parser.add_argument("--job-id", help="Stable job id. Defaults to a timestamp plus a short random suffix.")
    parser.add_argument("--mime-type", help="MIME type to send to Gemini. Inferred from the file extension if omitted.")
    parser.add_argument("--display-name", help="Display name stored with the Gemini file. Defaults to the file name.")
    parser.add_argument("--jobs-dir", default=str(DEFAULT_JOBS_DIR), help="Directory for job manifests.")
    parser.add_argument("--api-key", help="Gemini API key. Prefer --api-key-env or GEMINI_API_KEY for normal use.")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV, help="Environment variable that contains the API key.")
    parser.add_argument("--copy-original", action="store_true", help="Also copy the source bytes to original.bin.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing job directory.")
    return parser.parse_args()


def upload_to_gemini(
    *,
    file_path: Path,
    job_id: str,
    mime_type: str,
    display_name: str,
    jobs_dir: Path,
    api_key: str | None,
    api_key_env: str,
    copy_original: bool,
    force: bool,
) -> dict[str, Any]:
    job_dir = (jobs_dir / job_id).resolve()
    manifest_path = job_dir / "manifest.json"
    upload_response_path = job_dir / "upload_response.json"

    if job_dir.exists() and not force:
        raise FileExistsError(f"Job directory already exists: {job_dir}. Pass --force to overwrite metadata files.")

    job_dir.mkdir(parents=True, exist_ok=True)

    client = make_client(api_key=api_key, api_key_env=api_key_env)
    upload_kwargs: dict[str, Any] = {"file": str(file_path)}
    upload_config = {"display_name": display_name, "mime_type": mime_type}
    if upload_config:
        upload_kwargs["config"] = upload_config

    created_at = now_iso()
    uploaded_file = client.files.upload(**upload_kwargs)
    gemini_file_name = str(get_value(uploaded_file, "name") or "")
    if not gemini_file_name:
        raise RuntimeError(f"Gemini upload response did not include a file name: {uploaded_file!r}")

    verified_file = client.files.get(name=gemini_file_name)
    verified_at = now_iso()

    gemini_file_uri = str(get_value(verified_file, "uri") or get_value(uploaded_file, "uri") or "")
    if not gemini_file_uri:
        raise RuntimeError(f"Gemini file metadata did not include a URI for {gemini_file_name}")

    verified_mime_type = str(get_value(verified_file, "mime_type", "mimeType") or mime_type)
    state = normalize_state(get_value(verified_file, "state") or get_value(uploaded_file, "state"))

    local_copy_path = ""
    if copy_original:
        local_copy_path = str(copy_original_file(file_path, job_dir))

    manifest = {
        "job_id": job_id,
        "local_path": str(file_path),
        "local_size_bytes": file_path.stat().st_size,
        "local_copy_path": local_copy_path,
        "mime_type": verified_mime_type,
        "display_name": display_name,
        "gemini_file_name": gemini_file_name,
        "gemini_file_uri": gemini_file_uri,
        "state": state,
        "created_at": created_at,
        "verified_at": verified_at,
        "source": SOURCE_NAME,
    }

    upload_response = {
        "uploaded": to_jsonable(uploaded_file),
        "verified": to_jsonable(verified_file),
    }
    write_json(upload_response_path, upload_response)
    write_json(manifest_path, manifest)

    return {
        "ok": True,
        "job_id": job_id,
        "manifest_path": str(manifest_path),
        "upload_response_path": str(upload_response_path),
        "gemini_file_name": gemini_file_name,
        "gemini_file_uri": gemini_file_uri,
        "state": state,
    }


def main() -> int:
    args = parse_args()
    try:
        file_path = resolve_existing_file(args.file)
        job_id = make_job_id(args.job_id)
        mime_type = detect_mime_type(file_path, args.mime_type)
        display_name = (args.display_name or file_path.name).strip()
        if not display_name:
            raise ValueError("display name cannot be empty")

        result = upload_to_gemini(
            file_path=file_path,
            job_id=job_id,
            mime_type=mime_type,
            display_name=display_name,
            jobs_dir=Path(args.jobs_dir).expanduser(),
            api_key=args.api_key,
            api_key_env=args.api_key_env,
            copy_original=args.copy_original,
            force=args.force,
        )
    except Exception as exc:
        return fail(str(exc))

    print_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
