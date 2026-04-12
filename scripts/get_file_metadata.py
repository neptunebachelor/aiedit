from __future__ import annotations

import argparse
from pathlib import Path

from gemini_files_common import DEFAULT_API_KEY_ENV, fail, make_client, print_json, to_jsonable, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Gemini Files API metadata by file name.")
    parser.add_argument("--name", required=True, help="Gemini file name, for example files/abc123xyz.")
    parser.add_argument("--api-key", help="Gemini API key. Prefer --api-key-env or GEMINI_API_KEY for normal use.")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV, help="Environment variable that contains the API key.")
    parser.add_argument("--out", help="Optional path to write the metadata JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        client = make_client(api_key=args.api_key, api_key_env=args.api_key_env)
        metadata = to_jsonable(client.files.get(name=args.name))
        if args.out:
            write_json(Path(args.out).expanduser(), metadata)
    except Exception as exc:
        return fail(str(exc))

    print_json(metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
