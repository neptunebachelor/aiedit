"""Guardrail: fail if banned artifact paths get tracked.

Rules live in AGENTS.md §2. This test scans `git ls-files` output and rejects
any tracked file whose path matches one of the forbidden segment patterns.

Whitelist is tight: `tests/_tmp/` (test isolation) and `.video_data/` (which
shouldn't be tracked at all, but if it leaks in, that's a separate bug).

For the canonical way to produce artifact paths, use the helpers in
`video_data_paths.py`: `video_artifact_dir`, `video_frames_dir`,
`infer_dir_from_index`, `artifact_dir_from_index`.
"""
from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Match on any path segment.
FORBIDDEN_SEGMENT_PATTERNS = [
    re.compile(r"^tmp_"),                # tmp_batch_infer, tmp_verify, etc.
    re.compile(r"^temp_"),               # temp_visualize, temp_foo
    re.compile(r"_tmp$"),                # .ride-video-infer-tmp
    re.compile(r"^inference_tmp$"),
    re.compile(r"^\.gemini_temp"),       # .gemini_temp_bulk, .gemini_temp_pack
    re.compile(r"^\.ride-video-infer-"),
    re.compile(r"^infer_calib_"),
    re.compile(r"^tmpo"),                # tmpo1, tmpout, ...
    re.compile(r"^\.codex_backlog"),
]

# Match only at repo root (first path segment == full path).
FORBIDDEN_ROOT_FILES = {"temp.json", "temp_index.json"}
FORBIDDEN_ROOT_FILE_PATTERNS = [
    re.compile(r"^gemini_prompt.*\.txt$"),
    re.compile(r".*\.frame_decisions\.jsonl$"),
]

WHITELISTED_PREFIXES = (
    "tests/_tmp/",
    ".video_data/",
)


def _list_tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _is_violation(path: str) -> str | None:
    if any(path.startswith(prefix) for prefix in WHITELISTED_PREFIXES):
        return None
    segments = path.split("/")
    for seg in segments:
        for pattern in FORBIDDEN_SEGMENT_PATTERNS:
            if pattern.search(seg):
                return f"segment {seg!r} matches {pattern.pattern}"
    if "/" not in path:
        if path in FORBIDDEN_ROOT_FILES:
            return f"root file {path!r} is on the forbidden list"
        for pattern in FORBIDDEN_ROOT_FILE_PATTERNS:
            if pattern.match(path):
                return f"root file {path!r} matches {pattern.pattern}"
    return None


class ForbiddenPathsTests(unittest.TestCase):
    def test_no_tracked_file_matches_forbidden_pattern(self) -> None:
        violations = []
        for path in _list_tracked_files():
            reason = _is_violation(path)
            if reason:
                violations.append(f"  {path}  [{reason}]")
        if violations:
            self.fail(
                "Tracked files violate AGENTS.md artifact-path rules:\n"
                + "\n".join(violations)
                + "\n\nUse video_data_paths.py helpers (video_artifact_dir, "
                  "video_frames_dir, infer_dir_from_index) to place artifacts "
                  "under resolve_video_data_root()."
            )

    def test_violation_detector_matches_known_bad_examples(self) -> None:
        bad = [
            "tmp_batch_infer/foo.json",
            "tmp_verify_123/out.log",
            "temp_visualize/x.png",
            "inference_tmp/a",
            ".gemini_temp_bulk/x",
            ".gemini_temp_pack/y",
            ".ride-video-infer-tmp/z",
            ".codex_backlog_01/notes",
            "infer_calib_v1/run.log",
            "tmpo1/data",
            "subdir/.ride-video-infer-test/out",
            "temp.json",
            "temp_index.json",
            "gemini_prompt_abc.txt",
            "foo.frame_decisions.jsonl",
        ]
        for path in bad:
            self.assertIsNotNone(_is_violation(path), f"Should flag {path}")

    def test_violation_detector_passes_known_good_examples(self) -> None:
        good = [
            "AGENTS.md",
            "pipeline.py",
            "tests/test_video_data_paths.py",
            "tests/_tmp/throwaway.json",
            "skills/ride-video-infer/scripts/prepare_packs.py",
            "plans/todo/PLAN_artifact_path_rules.md",
            ".video_data/videos/ride01/extract/index.json",
            "config.toml",
        ]
        for path in good:
            self.assertIsNone(_is_violation(path), f"Should not flag {path}")


if __name__ == "__main__":
    unittest.main()
