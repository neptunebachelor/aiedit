import json
import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import upload_to_gemini
from gemini_files_common import detect_mime_type, make_job_id


class FakeGeminiFiles:
    def __init__(self) -> None:
        self.upload_kwargs = {}

    def upload(self, **kwargs):
        self.upload_kwargs = kwargs
        return {
            "name": "files/fake123",
            "uri": "https://generativelanguage.googleapis.com/v1beta/files/fake123",
            "mime_type": kwargs["config"]["mime_type"],
            "state": "ACTIVE",
        }

    def get(self, *, name: str):
        return {
            "name": name,
            "uri": "https://generativelanguage.googleapis.com/v1beta/files/fake123",
            "mime_type": "image/png",
            "state": "ACTIVE",
        }


class FakeGeminiClient:
    def __init__(self) -> None:
        self.files = FakeGeminiFiles()


class GeminiFilesScriptsTests(unittest.TestCase):
    def make_temp_dir(self) -> Path:
        base_dir = Path(__file__).resolve().parent / "_tmp"
        base_dir.mkdir(parents=True, exist_ok=True)
        target = base_dir / f"gemini_files_{self.id().split('.')[-1]}"
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def test_detect_mime_type_from_extension(self) -> None:
        self.assertEqual(detect_mime_type(Path("sample.png"), None), "image/png")

    def test_make_job_id_sanitizes_path_unsafe_text(self) -> None:
        self.assertEqual(make_job_id("  demo job/001  "), "demo_job_001")

    def test_upload_to_gemini_writes_manifest_and_raw_response(self) -> None:
        tmp_dir = self.make_temp_dir()
        original_make_client = upload_to_gemini.make_client
        fake_client = FakeGeminiClient()
        try:
            source = tmp_dir / "frame.png"
            source.write_bytes(b"fake-image")
            jobs_dir = tmp_dir / "jobs"

            upload_to_gemini.make_client = lambda **kwargs: fake_client
            result = upload_to_gemini.upload_to_gemini(
                file_path=source,
                job_id="job001",
                mime_type="image/png",
                display_name="frame.png",
                jobs_dir=jobs_dir,
                api_key=None,
                api_key_env="GEMINI_API_KEY",
                copy_original=False,
                force=False,
            )

            manifest_path = Path(result["manifest_path"])
            upload_response_path = Path(result["upload_response_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["gemini_file_name"], "files/fake123")
            self.assertEqual(
                manifest["gemini_file_uri"],
                "https://generativelanguage.googleapis.com/v1beta/files/fake123",
            )
            self.assertEqual(manifest["mime_type"], "image/png")
            self.assertEqual(manifest["source"], "gemini_files_api")
            self.assertFalse((jobs_dir / "job001" / "original.bin").exists())
            self.assertTrue(upload_response_path.exists())
            self.assertEqual(fake_client.files.upload_kwargs["config"]["display_name"], "frame.png")
        finally:
            upload_to_gemini.make_client = original_make_client
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
