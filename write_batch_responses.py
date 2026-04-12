#!/usr/bin/env python
"""Write a response.json file for a prepared inference pack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pipeline.py").exists():
            return candidate
    raise RuntimeError("Could not find project root containing pipeline.py")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
sys.path.insert(0, str(REPO_ROOT))

from video_data_paths import infer_dir_from_index  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write pack response JSON without hardcoded video paths.")
    parser.add_argument("--index", required=True, help="Source extract/index.json used to prepare packs")
    parser.add_argument("--pack", required=True, help="Pack name such as pack_0001")
    parser.add_argument("--response", required=True, help="JSON array/object file to copy into response.json")
    parser.add_argument("--packs-dir", help="Prepared packs directory. Defaults to the canonical infer/packs directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index_path = Path(args.index).expanduser().resolve()
    packs_dir = Path(args.packs_dir).expanduser().resolve() if args.packs_dir else infer_dir_from_index(index_path) / "packs"
    response_source = Path(args.response).expanduser().resolve()
    payload = json.loads(response_source.read_text(encoding="utf-8"))
    output_path = packs_dir / str(args.pack) / "response.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"response_path": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
