import base64
import copy
import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import video_data_paths
from analyze_video import build_packed_user_prompt, validate_pack_decisions
from pipeline import (
    DEFAULT_PIPELINE_CONFIG,
    OpenAICompatibleVisionProvider,
    infer_from_extract_index,
)


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2Z6xQAAAAASUVORK5CYII="
)


def _make_temp_dir(test_id: str) -> Path:
    base = Path(__file__).resolve().parent / "_tmp"
    base.mkdir(parents=True, exist_ok=True)
    target = base / f"case_{os.getpid()}_{test_id}"
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    return target


class BuildPackedUserPromptTests(unittest.TestCase):
    def test_prompt_lists_each_frame_and_wraps_schema_in_decisions(self) -> None:
        config = copy.deepcopy(DEFAULT_PIPELINE_CONFIG)
        frames = [
            {"frame_number": 10, "timestamp_seconds": 12.40},
            {"frame_number": 20, "timestamp_seconds": 12.80},
            {"frame_number": 30, "timestamp_seconds": 13.20},
            {"frame_number": 40, "timestamp_seconds": 13.60},
            {"frame_number": 50, "timestamp_seconds": 14.00},
        ]

        prompt = build_packed_user_prompt(config, frames)

        for frame in frames:
            self.assertIn(
                f"Frame {frame['frame_number']}, t={frame['timestamp_seconds']:.2f}s",
                prompt,
            )
        self.assertIn('"decisions"', prompt)
        self.assertIn('"frame_number"', prompt)
        self.assertIn("5 frames", prompt)


class ValidatePackDecisionsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {"decision": {"max_reason_chars": 80}}
        self.frames = [
            {"frame_number": 1, "timestamp_seconds": 1.0},
            {"frame_number": 2, "timestamp_seconds": 2.0},
            {"frame_number": 3, "timestamp_seconds": 3.0},
        ]

    def test_clean_response_preserves_expected_order(self) -> None:
        raw = [
            {"frame_number": 3, "keep": True, "score": 0.7, "labels": [], "reason": "c"},
            {"frame_number": 1, "keep": True, "score": 0.9, "labels": [], "reason": "a"},
            {"frame_number": 2, "keep": False, "score": 0.1, "labels": [], "discard_reason": "b"},
        ]

        out = validate_pack_decisions(self.frames, raw, self.config)

        self.assertEqual(len(out), 3)
        self.assertEqual(out[0]["reason"], "a")
        self.assertFalse(out[1]["keep"])
        self.assertEqual(out[2]["reason"], "c")

    def test_missing_frame_gets_provider_error_fallback(self) -> None:
        raw = [
            {"frame_number": 1, "keep": True, "score": 0.9, "labels": [], "reason": "a"},
            {"frame_number": 3, "keep": True, "score": 0.7, "labels": [], "reason": "c"},
        ]

        out = validate_pack_decisions(self.frames, raw, self.config)

        self.assertEqual(len(out), 3)
        self.assertFalse(out[1]["keep"])
        self.assertIn("missing from output", out[1]["discard_reason"])

    def test_duplicate_frame_number_keeps_first_entry(self) -> None:
        raw = [
            {"frame_number": 1, "keep": True, "score": 0.9, "labels": [], "reason": "first"},
            {"frame_number": 1, "keep": False, "score": 0.1, "labels": [], "reason": "second"},
            {"frame_number": 2, "keep": True, "score": 0.5, "labels": [], "reason": "x"},
            {"frame_number": 3, "keep": True, "score": 0.8, "labels": [], "reason": "y"},
        ]

        out = validate_pack_decisions(self.frames, raw, self.config)

        self.assertEqual(out[0]["reason"], "first")
        self.assertTrue(out[0]["keep"])

    def test_unexpected_frame_number_is_dropped(self) -> None:
        raw = [
            {"frame_number": 1, "keep": True, "score": 0.9, "labels": [], "reason": "a"},
            {"frame_number": 999, "keep": True, "score": 0.9, "labels": [], "reason": "unexpected"},
            {"frame_number": 2, "keep": True, "score": 0.5, "labels": [], "reason": "b"},
            {"frame_number": 3, "keep": True, "score": 0.8, "labels": [], "reason": "c"},
        ]

        out = validate_pack_decisions(self.frames, raw, self.config)

        reasons = [d["reason"] for d in out]
        self.assertEqual(reasons, ["a", "b", "c"])


class OpenAICompatiblePackedPayloadTests(unittest.TestCase):
    def test_infer_pack_builds_multi_image_user_content(self) -> None:
        provider = OpenAICompatibleVisionProvider(
            api_base="https://api.test.local",
            model="qwen3-vl-flash",
            api_key="test-key",
            temperature=0.1,
            timeout_seconds=30,
            image_transport="base64",
            image_url_template="",
            json_output=False,
            extra_body={"enable_thinking": False},
        )
        frames = [
            {"frame_number": 10, "timestamp_seconds": 1.0, "image_bytes": PNG_BYTES, "image_path": None},
            {"frame_number": 20, "timestamp_seconds": 2.0, "image_bytes": PNG_BYTES, "image_path": None},
            {"frame_number": 30, "timestamp_seconds": 3.0, "image_bytes": PNG_BYTES, "image_path": None},
        ]
        captured: dict = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            msg = MagicMock()
            msg.content = json.dumps({
                "decisions": [
                    {"frame_number": 10, "keep": True, "score": 0.9, "labels": [], "reason": "r"},
                    {"frame_number": 20, "keep": False, "score": 0.1, "labels": [], "discard_reason": "x"},
                    {"frame_number": 30, "keep": True, "score": 0.7, "labels": [], "reason": "s"},
                ]
            })
            choice = MagicMock()
            choice.message = msg
            response = MagicMock()
            response.choices = [choice]
            return response

        provider.client.chat.completions.create = fake_create
        config = copy.deepcopy(DEFAULT_PIPELINE_CONFIG)

        out = provider.infer_pack(frames, config=config)

        messages = captured["messages"]
        self.assertEqual(messages[0]["role"], "system")
        user_content = messages[1]["content"]
        self.assertEqual(len(user_content), 1 + len(frames))
        self.assertEqual(user_content[0]["type"], "text")
        for image_part in user_content[1:]:
            self.assertEqual(image_part["type"], "image_url")
            self.assertTrue(image_part["image_url"]["url"].startswith("data:image/"))

        self.assertEqual(len(out), 3)
        self.assertTrue(out[0]["keep"])
        self.assertFalse(out[1]["keep"])
        self.assertTrue(out[2]["keep"])


class InferFromExtractIndexPackedTests(unittest.TestCase):
    def test_pack_size_three_dispatches_ceil_of_ten_over_three_packs(self) -> None:
        tmp_dir = _make_temp_dir(self.id().split(".")[-1])
        try:
            frames_dir = tmp_dir / "frames" / "test_slug"
            frames_dir.mkdir(parents=True)
            frame_records = []
            for i in range(1, 11):
                img = frames_dir / f"frame_{i:08d}.png"
                img.write_bytes(PNG_BYTES)
                frame_records.append({
                    "timestamp_seconds": float(i),
                    "timestamp_srt": f"00:00:{i:02d},000",
                    "frame_number": i,
                    "candidate": True,
                    "image_path": str(img),
                    "blur_score": 100.0,
                    "frame_diff": 10.0,
                    "hash_distance": 10,
                })

            index_dir = tmp_dir / "videos" / "test_slug" / "extract"
            index_dir.mkdir(parents=True)
            index_path = index_dir / "index.json"
            index_path.write_text(json.dumps({
                "video": {
                    "filename": "test.mp4",
                    "source_path": str(tmp_dir / "test.mp4"),
                    "duration_seconds": 30.0,
                    "frame_count": 10,
                    "width": 1920,
                    "height": 1080,
                },
                "frames": frame_records,
            }), encoding="utf-8")

            class StubPackedProvider:
                def __init__(self) -> None:
                    self.calls: list[list[int]] = []

                def infer(self, *args, **kwargs):  # noqa: ANN001, ANN002
                    raise AssertionError("single-frame infer must not be called when pack_size > 1")

                def infer_pack(self, pack, *, config):  # noqa: ANN001
                    self.calls.append([int(f["frame_number"]) for f in pack])
                    return [
                        {"keep": True, "score": 0.9, "labels": [], "reason": "ok", "discard_reason": ""}
                        for _ in pack
                    ]

            provider = StubPackedProvider()
            provider_snapshot = {
                "selected_route": "qwen",
                "qwen": {"pack_size": 3, "min_request_interval_seconds": 0.0},
            }
            config = copy.deepcopy(DEFAULT_PIPELINE_CONFIG)

            with patch.object(video_data_paths, "resolve_video_data_root", return_value=tmp_dir):
                analysis_path = infer_from_extract_index(
                    index_path,
                    provider=provider,
                    provider_snapshot=provider_snapshot,
                    config=config,
                )

            self.assertEqual(len(provider.calls), 4)
            self.assertEqual([len(c) for c in provider.calls], [3, 3, 3, 1])
            self.assertEqual(
                [n for call in provider.calls for n in call],
                list(range(1, 11)),
            )
            self.assertTrue(analysis_path.exists())
            analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
            kept = [r for r in analysis["frames"] if r.get("keep")]
            self.assertEqual(len(kept), 10)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
