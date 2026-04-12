import base64
import importlib.util
import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
UPLOAD_SCRIPT = ROOT / "skills" / "ride-video-infer" / "scripts" / "upload_frame_files.py"
RUN_SCRIPT = ROOT / "skills" / "ride-video-infer" / "scripts" / "run_file_api_packed.py"
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2Z6xQAAAAASUVORK5CYII="
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


upload_module = load_module("upload_frame_files_test", UPLOAD_SCRIPT)
run_module = load_module("run_file_api_packed_test", RUN_SCRIPT)


class FileApiInferTests(unittest.TestCase):
    def make_temp_dir(self) -> Path:
        base_dir = Path(__file__).resolve().parent / "_tmp"
        base_dir.mkdir(parents=True, exist_ok=True)
        target = base_dir / f"file_api_{os.getpid()}_{self.id().split('.')[-1]}"
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def write_index(self, tmp_dir: Path) -> Path:
        video_dir = tmp_dir / "ride01"
        frames_dir = video_dir / "extract" / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = []
        for frame_number in range(1, 4):
            image_path = frames_dir / f"frame_{frame_number:09d}.png"
            image_path.write_bytes(PNG_BYTES)
            frame_paths.append(image_path)
        index_path = video_dir / "extract" / "index.json"
        index_path.write_text(
            json.dumps(
                {
                    "video": {"filename": "ride01.mp4"},
                    "frames": [
                        {
                            "frame_number": 1,
                            "candidate": True,
                            "timestamp_seconds": 1.0,
                            "image_path": str(frame_paths[0]),
                        },
                        {
                            "frame_number": 2,
                            "candidate": False,
                            "timestamp_seconds": 2.0,
                            "image_path": str(frame_paths[1]),
                        },
                        {
                            "frame_number": 3,
                            "candidate": True,
                            "timestamp_seconds": 3.0,
                            "image_path": str(frame_paths[2]),
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return index_path

    def test_upload_manifest_dry_run_uses_candidate_default(self) -> None:
        tmp_dir = self.make_temp_dir()
        try:
            index_path = self.write_index(tmp_dir)
            manifest = upload_module.build_manifest(
                index_path,
                provider="openai",
                include="candidates",
                dry_run=True,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertEqual(manifest["provider"], "openai")
        self.assertEqual(manifest["frame_count"], 2)
        self.assertEqual(manifest["planned_count"], 2)
        self.assertEqual([frame["frame_number"] for frame in manifest["frames"]], [1, 3])
        self.assertTrue(all(frame["upload_status"] == "planned" for frame in manifest["frames"]))
        self.assertIn("sha256", manifest["frames"][0])

    def test_upload_openai_uses_vision_purpose(self) -> None:
        tmp_dir = self.make_temp_dir()
        calls = []

        class Files:
            def create(self, **kwargs: Any) -> Any:
                calls.append(kwargs)
                return SimpleNamespace(id="file_123")

        try:
            image_path = tmp_dir / "frame.png"
            image_path.write_bytes(PNG_BYTES)
            refs = upload_module.upload_openai_frame(SimpleNamespace(files=Files()), image_path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertEqual(refs["openai_file_id"], "file_123")
        self.assertEqual(calls[0]["purpose"], "vision")
        self.assertTrue(hasattr(calls[0]["file"], "read"))

    def test_openai_pack_request_uses_input_image_file_ids(self) -> None:
        captured = {}

        class Responses:
            def create(self, **kwargs: Any) -> Any:
                captured.update(kwargs)
                return SimpleNamespace(output_text='[{"frame_number": 7, "keep": true, "score": 0.8, "labels": []}]')

        output = run_module.call_openai_pack(
            SimpleNamespace(responses=Responses()),
            model="gpt-test",
            prompt="prompt",
            temperature=0.1,
            pack=[{"frame_number": 7, "openai_file_id": "file_abc"}],
        )

        self.assertIn("frame_number", output)
        self.assertEqual(captured["model"], "gpt-test")
        content = captured["input"][0]["content"]
        self.assertEqual(content[1]["type"], "input_image")
        self.assertEqual(content[1]["file_id"], "file_abc")

    def test_gemini_pack_request_uses_file_uri_parts(self) -> None:
        captured = {}

        class Models:
            def generate_content(self, **kwargs: Any) -> Any:
                captured.update(kwargs)
                return SimpleNamespace(text='[{"frame_number": 8, "keep": true, "score": 0.8, "labels": []}]')

        output = run_module.call_gemini_pack(
            SimpleNamespace(models=Models()),
            model="gemini-test",
            prompt="prompt",
            temperature=0.1,
            pack=[{"frame_number": 8, "gemini_file_uri": "https://files/test", "mime_type": "image/png"}],
        )

        self.assertIn("frame_number", output)
        self.assertEqual(captured["model"], "gemini-test")
        self.assertEqual(captured["contents"][0], "prompt")
        self.assertEqual(captured["contents"][1].file_data.file_uri, "https://files/test")
        self.assertEqual(captured["contents"][1].file_data.mime_type, "image/png")

    def test_dry_run_records_control_variables_and_hashes(self) -> None:
        tmp_dir = self.make_temp_dir()
        try:
            index_path = self.write_index(tmp_dir)
            upload_manifest = upload_module.build_manifest(
                index_path,
                provider="openai",
                include="all",
                dry_run=True,
            )
            for frame in upload_manifest["frames"]:
                frame["upload_status"] = "uploaded"
                frame["openai_file_id"] = f"file_{frame['frame_number']}"
            upload_manifest_path = index_path.parent.parent / "infer" / "file_uploads.openai.json"
            upload_module.write_json(upload_manifest_path, upload_manifest)

            result = run_module.run_packed_inference(
                upload_manifest_path=upload_manifest_path,
                provider="openai",
                model="gpt-test",
                pack_size=5,
                start_pack=1,
                end_pack=1,
                max_frames=20,
                prompt_variant="calibration",
                temperature=0.2,
                output_decisions=None,
                include_all_frames=False,
                restart=True,
                dry_run=True,
                apply_outputs=False,
                config="config.toml",
            )
            run_manifest = json.loads(Path(result["run_manifest_path"]).read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(run_manifest["pack_size"], 5)
        self.assertEqual(run_manifest["max_frames"], 20)
        self.assertEqual(run_manifest["prompt_variant"], "calibration")
        self.assertEqual(run_manifest["candidate_frame_numbers"], [1, 3])
        self.assertIn("upload_manifest_hash", run_manifest)
        self.assertIn("prompt_hash", run_manifest)

    def test_invalid_pack_response_emits_skill_error_rejects(self) -> None:
        frames = [{"frame_number": 1}, {"frame_number": 2}]
        decisions, missing, duplicates = run_module.validate_or_reject_pack(
            [{"frame_number": 1, "keep": True, "score": 0.8, "labels": []}],
            frames,
            error_reason="skill_error: missing or duplicate frame in file api output",
        )

        self.assertEqual(missing, [2])
        self.assertEqual(duplicates, [])
        self.assertEqual(len(decisions), 2)
        self.assertFalse(decisions[1]["keep"])
        self.assertTrue(decisions[1]["discard_reason"].startswith("skill_error:"))


if __name__ == "__main__":
    unittest.main()
