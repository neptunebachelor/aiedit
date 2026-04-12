from __future__ import annotations

import argparse
from pathlib import Path

from gemini_files_common import DEFAULT_API_KEY_ENV, fail, make_client, now_iso, print_json, read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete a Gemini Files API file by name or manifest.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--name", help="Gemini file name, for example files/abc123xyz.")
    target.add_argument("--manifest", help="Manifest written by upload_to_gemini.py.")
    parser.add_argument("--api-key", help="Gemini API key. Prefer --api-key-env or GEMINI_API_KEY for normal use.")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV, help="Environment variable that contains the API key.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.manifest:
            manifest = read_json(Path(args.manifest).expanduser())
            name = str(manifest.get("gemini_file_name", "")).strip()
            if not name:
                raise ValueError(f"Manifest does not contain gemini_file_name: {args.manifest}")
        else:
            name = str(args.name).strip()

        client = make_client(api_key=args.api_key, api_key_env=args.api_key_env)
        client.files.delete(name=name)
    except Exception as exc:
        return fail(str(exc))

    print_json({"ok": True, "deleted": name, "deleted_at": now_iso()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
