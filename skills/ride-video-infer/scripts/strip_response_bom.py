#!/usr/bin/env python
"""Strip UTF-8 BOM bytes from Gemini pack response.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


UTF8_BOM = b"\xef\xbb\xbf"


def response_paths_from_args(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.response:
        paths.extend(Path(value).expanduser().resolve() for value in args.response)
    if args.pack_dir:
        paths.extend(Path(value).expanduser().resolve() / "response.json" for value in args.pack_dir)
    if args.packs_dir:
        packs_dir = Path(args.packs_dir).expanduser().resolve()
        end_pack = int(args.end_pack)
        if end_pack <= 0:
            pack_dirs = sorted(path for path in packs_dir.iterdir() if path.is_dir() and path.name.startswith("pack_"))
        else:
            pack_dirs = [packs_dir / f"pack_{pack_number:04d}" for pack_number in range(int(args.start_pack), end_pack + 1)]
        paths.extend(pack_dir / "response.json" for pack_dir in pack_dirs)
    return paths


def strip_bom(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    had_bom = raw.startswith(UTF8_BOM)
    if had_bom:
        path.write_bytes(raw[len(UTF8_BOM) :])
    return {
        "path": str(path),
        "exists": True,
        "had_bom": had_bom,
        "stripped": had_bom,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove UTF-8 BOM from response.json files.")
    parser.add_argument("--response", action="append", help="Direct response.json path")
    parser.add_argument("--pack-dir", action="append", help="Pack directory containing response.json")
    parser.add_argument("--packs-dir", help="Directory containing pack_XXXX children")
    parser.add_argument("--start-pack", type=int, default=1)
    parser.add_argument("--end-pack", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = response_paths_from_args(args)
    if not paths:
        raise ValueError("Provide --response, --pack-dir, or --packs-dir")

    seen: set[Path] = set()
    results: list[dict[str, object]] = []
    missing: list[str] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            missing.append(str(path))
            results.append({"path": str(path), "exists": False, "had_bom": False, "stripped": False})
            continue
        results.append(strip_bom(path))

    summary = {
        "responses_checked": len(results),
        "stripped_count": sum(1 for item in results if item["stripped"]),
        "missing": missing,
        "responses": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
