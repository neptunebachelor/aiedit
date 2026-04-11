import base64
import copy
import json
import os
import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import (
    DEFAULT_PIPELINE_CONFIG,
    OpenAICompatibleBatchVisionProvider,
    build_provider,
    parse_openai_batch_results,
)


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2Z6xQAAAAASUVORK5CYII="
)


class QwenBatchTests(unittest.TestCase):
    def make_temp_dir(self) -> Path:
        base_dir = Path(__file__).resolve().parent / "_tmp"
        base_dir.mkdir(parents=True, exist_ok=True)
        target = base_dir / f"case_{os.getpid()}_{self.id().split('.')[-1]}"
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def test_build_provider_selects_qwen_async_batch(self) -> None:
        provider_config = copy.deepcopy(DEFAULT_PIPELINE_CONFIG["provider"])
        provider_config["routing"] = "qwen"
        provider_config["submission_mode"] = "async"
        provider_config["qwen"]["supports_async_batch"] = True
        provider_config["qwen"]["api_key"] = "test-key"
        provider_config["qwen"]["model"] = "qwen3.5-plus"

        selection = build_provider(provider_config)

        self.assertEqual(selection.route, "qwen")
        self.assertEqual(selection.execution_mode, "async_batch")
        self.assertIsInstance(selection.provider, OpenAICompatibleBatchVisionProvider)

    def test_qwen_batch_request_inlines_extra_body(self) -> None:
        tmp_dir = self.make_temp_dir()
        try:
            image_path = tmp_dir / "frame.png"
            image_path.write_bytes(PNG_BYTES)

            provider = OpenAICompatibleBatchVisionProvider(
                route_name="qwen",
                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                model="qwen3.5-plus",
                api_key="test-key",
                temperature=0.1,
                timeout_seconds=30,
                image_transport="base64",
                image_url_template="",
                json_output=False,
                extra_body={"enable_thinking": False},
            )
            record = provider._build_request_record(
                {
                    "frame_number": 7,
                    "timestamp_seconds": 3.5,
                    "image_path": str(image_path),
                },
                prompt_snapshot=copy.deepcopy(DEFAULT_PIPELINE_CONFIG["prompt"]),
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertEqual(record["custom_id"], "frame_000000007")
        self.assertEqual(record["url"], "/v1/chat/completions")
        self.assertFalse(record["body"]["enable_thinking"])
        self.assertNotIn("extra_body", record["body"])
        self.assertEqual(record["body"]["messages"][1]["content"][1]["type"], "image_url")

    def test_parse_openai_batch_results_maps_frames_from_success_and_error_files(self) -> None:
        tmp_dir = self.make_temp_dir()
        try:
            results_path = tmp_dir / "results.jsonl"
            errors_path = tmp_dir / "errors.jsonl"
            results_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "custom_id": "frame_000000010",
                                "response": {
                                    "status_code": 200,
                                    "body": {
                                        "choices": [
                                            {
                                                "message": {
                                                    "content": '{"keep": true, "score": 0.82, "labels": ["apex"], "reason": "nice corner"}'
                                                }
                                            }
                                        ]
                                    },
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "custom_id": "frame_000000003",
                                "response": {
                                    "status_code": 200,
                                    "body": {
                                        "choices": [
                                            {
                                                "message": {
                                                    "content": '{"keep": false, "score": 0.12, "labels": [], "discard_reason": "blur"}'
                                                }
                                            }
                                        ]
                                    },
                                },
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            errors_path.write_text(
                json.dumps(
                    {
                        "custom_id": "frame_000000005",
                        "error": {
                            "message": "image fetch failed",
                        },
                    }
                ),
                encoding="utf-8",
            )

            decisions = parse_openai_batch_results(
                results_path,
                selection_snapshot=copy.deepcopy(DEFAULT_PIPELINE_CONFIG["selection"]),
                error_results_path=errors_path,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertTrue(decisions[10]["keep"])
        self.assertAlmostEqual(decisions[10]["score"], 0.82)
        self.assertFalse(decisions[3]["keep"])
        self.assertIn("blur", decisions[3]["discard_reason"])
        self.assertIn("image fetch failed", decisions[5]["discard_reason"])


if __name__ == "__main__":
    unittest.main()
