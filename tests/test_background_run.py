import json
import os
import shutil
import sys
import unittest
from contextlib import redirect_stdout
from argparse import Namespace
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline


def _make_temp_dir(test_id: str) -> Path:
    base = Path(__file__).resolve().parent / "_tmp"
    base.mkdir(parents=True, exist_ok=True)
    target = base / f"background_{os.getpid()}_{test_id}"
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    return target


class BackgroundRunTests(unittest.TestCase):
    def test_launcher_writes_manifest_and_strips_background_flag(self) -> None:
        tmp_dir = _make_temp_dir(self.id().split(".")[-1])
        captured: dict = {}

        class FakeProcess:
            pid = 4242

            def __init__(self, command, **kwargs):  # noqa: ANN001
                captured["command"] = command
                captured["kwargs"] = kwargs

        args = Namespace(
            video=str(tmp_dir / "ride01.mp4"),
            output_root=None,
        )
        artifact_dir = tmp_dir / "data" / "videos" / "ride01"
        try:
            with (
                patch.object(pipeline.sys, "argv", ["pipeline.py", "run", "--video", str(tmp_dir / "ride01.mp4"), "--background", "--target-seconds", "30"]),
                patch.object(pipeline.subprocess, "Popen", FakeProcess),
                patch.object(pipeline, "resolve_video_output_dir", return_value=artifact_dir),
            ):
                manifest_path = pipeline.launch_background_run(args, pipeline.DEFAULT_PIPELINE_CONFIG)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["pid"], 4242)
            self.assertEqual(manifest["stage"], "run.background")
            self.assertEqual(manifest["status"], "started")
            self.assertEqual(manifest["artifact_dir"], str(artifact_dir))
            self.assertEqual(manifest_path.parent.parent, artifact_dir / "runs")
            self.assertNotIn("--background", captured["command"])
            self.assertIn("--target-seconds", captured["command"])
            self.assertEqual(captured["kwargs"]["env"]["AIEDIT_BACKGROUND_JOB_MANIFEST"], str(manifest_path))
            self.assertTrue((manifest_path.parent / "pid").exists())
            self.assertTrue((manifest_path.parent / "stdout.log").exists())
            self.assertTrue((manifest_path.parent / "stderr.log").exists())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_background_manifest_wrapper_records_success_and_failure(self) -> None:
        tmp_dir = _make_temp_dir(self.id().split(".")[-1])
        manifest_path = tmp_dir / "job.json"
        parsed_args = Namespace(command="run", background=False)
        try:
            with patch.dict(pipeline.os.environ, {"AIEDIT_BACKGROUND_JOB_MANIFEST": str(manifest_path)}):
                with patch.object(pipeline, "parse_args", return_value=parsed_args), patch.object(pipeline, "dispatch_command", return_value=0):
                    exit_code = pipeline.main()

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["exit_code"], 0)

            with patch.dict(pipeline.os.environ, {"AIEDIT_BACKGROUND_JOB_MANIFEST": str(manifest_path)}):
                with patch.object(pipeline, "parse_args", return_value=parsed_args), patch.object(pipeline, "dispatch_command", side_effect=RuntimeError("boom")):
                    with self.assertRaises(RuntimeError):
                        pipeline.main()

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["exit_code"], 1)
            self.assertEqual(manifest["error_type"], "RuntimeError")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_status_command_prints_latest_job_for_video(self) -> None:
        tmp_dir = _make_temp_dir(self.id().split(".")[-1])
        video_path = tmp_dir / "ride01.mp4"
        artifact_dir = tmp_dir / "data" / "videos" / "ride01"
        job_dir = artifact_dir / "runs" / "run_20260101T000000Z"
        job_dir.mkdir(parents=True)
        job_path = job_dir / "job.json"
        job_path.write_text(
            json.dumps(
                {
                    "stage": "run.background",
                    "status": "completed",
                    "pid": 123,
                    "exit_code": 0,
                    "video": str(video_path),
                    "artifact_dir": str(artifact_dir),
                    "stdout": str(job_dir / "stdout.log"),
                    "stderr": str(job_dir / "stderr.log"),
                }
            ),
            encoding="utf-8",
        )
        try:
            args = Namespace(
                job=None,
                video=str(video_path),
                output_root=None,
                config="config.toml",
                json=False,
                log_level="INFO",
            )
            output = StringIO()
            with (
                patch.object(pipeline, "load_pipeline_config", return_value=pipeline.DEFAULT_PIPELINE_CONFIG),
                patch.object(pipeline, "resolve_video_output_dir", return_value=artifact_dir),
                redirect_stdout(output),
            ):
                exit_code = pipeline.command_status(args)

            self.assertEqual(exit_code, 0)
            text = output.getvalue()
            self.assertIn("status: completed", text)
            self.assertIn("exit_code: 0", text)
            self.assertIn(str(job_path), text)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_run_reuses_existing_extract_index_for_resume(self) -> None:
        tmp_dir = _make_temp_dir(self.id().split(".")[-1])
        video_path = tmp_dir / "ride01.mp4"
        index_path = tmp_dir / "data" / "videos" / "ride01" / "extract" / "index.json"
        stage_output_dir = index_path.parent.parent
        index_path.parent.mkdir(parents=True)
        index_path.write_text(json.dumps({"video": {"filename": "ride01.mp4"}, "frames": []}), encoding="utf-8")
        try:
            args = Namespace(
                command="run",
                video=str(video_path),
                output_root=None,
                config="config.toml",
                extract_index=None,
                target_seconds=30,
                top_highlights=1,
                skip_review=True,
                log_level="INFO",
                background=False,
            )
            infer_extract_indexes: list[str] = []

            def fake_command_infer(infer_args):  # noqa: ANN001
                infer_extract_indexes.append(infer_args.extract_index)
                (stage_output_dir / "analysis.json").write_text(json.dumps({"frames": []}), encoding="utf-8")
                return 0

            with (
                patch.object(pipeline, "load_pipeline_config", return_value=pipeline.DEFAULT_PIPELINE_CONFIG),
                patch.object(pipeline, "list_videos", return_value=[video_path]),
                patch.object(pipeline, "ensure_extract_index", return_value=index_path),
                patch.object(pipeline, "command_extract") as command_extract,
                patch.object(pipeline, "command_infer", side_effect=fake_command_infer),
                patch.object(pipeline, "resolve_video_dir_for_index", return_value=stage_output_dir),
                patch.object(pipeline, "command_temporal", return_value=0),
                patch.object(pipeline, "command_render", return_value=0),
            ):
                exit_code = pipeline.command_run(args)

            self.assertEqual(exit_code, 0)
            command_extract.assert_not_called()
            self.assertEqual(infer_extract_indexes, [str(index_path)])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_run_skips_infer_when_analysis_exists_unless_restart(self) -> None:
        tmp_dir = _make_temp_dir(self.id().split(".")[-1])
        video_path = tmp_dir / "ride01.mp4"
        index_path = tmp_dir / "data" / "videos" / "ride01" / "extract" / "index.json"
        stage_output_dir = index_path.parent.parent
        index_path.parent.mkdir(parents=True)
        index_path.write_text(json.dumps({"video": {"filename": "ride01.mp4"}, "frames": []}), encoding="utf-8")
        (stage_output_dir / "analysis.json").write_text(json.dumps({"frames": []}), encoding="utf-8")
        try:
            args = Namespace(
                command="run",
                video=str(video_path),
                output_root=None,
                config="config.toml",
                extract_index=None,
                target_seconds=30,
                top_highlights=1,
                skip_review=True,
                log_level="INFO",
                background=False,
                restart=False,
            )

            with (
                patch.object(pipeline, "load_pipeline_config", return_value=pipeline.DEFAULT_PIPELINE_CONFIG),
                patch.object(pipeline, "list_videos", return_value=[video_path]),
                patch.object(pipeline, "ensure_extract_index", return_value=index_path),
                patch.object(pipeline, "command_infer") as command_infer,
                patch.object(pipeline, "resolve_video_dir_for_index", return_value=stage_output_dir),
                patch.object(pipeline, "command_temporal", return_value=0),
                patch.object(pipeline, "command_render", return_value=0),
            ):
                exit_code = pipeline.command_run(args)

            self.assertEqual(exit_code, 0)
            command_infer.assert_not_called()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_viral_run_defaults_feed_infer_temporal_and_render(self) -> None:
        tmp_dir = _make_temp_dir(self.id().split(".")[-1])
        video_path = tmp_dir / "ride01.mp4"
        index_path = tmp_dir / "data" / "videos" / "ride01" / "extract" / "index.json"
        stage_output_dir = index_path.parent.parent
        index_path.parent.mkdir(parents=True)
        index_path.write_text(json.dumps({"video": {"filename": "ride01.mp4"}, "frames": []}), encoding="utf-8")
        captured: dict = {}
        try:
            args = Namespace(
                command="run",
                video=str(video_path),
                output_root=None,
                config="config.toml",
                extract_index=None,
                target_seconds=None,
                selection_mode=None,
                single_top_k=None,
                top_highlights=None,
                caption_mode=None,
                caption_style=None,
                prompt_preset=None,
                skip_review=True,
                log_level="INFO",
                background=False,
                restart=False,
                viral=True,
            )

            def fake_command_infer(infer_args):  # noqa: ANN001
                captured["infer_prompt_preset"] = infer_args.prompt_preset
                (stage_output_dir / "analysis.json").write_text(json.dumps({"frames": []}), encoding="utf-8")
                return 0

            def fake_command_temporal(temporal_args):  # noqa: ANN001
                captured["temporal_final_duration"] = temporal_args.final_duration_seconds
                return 0

            def fake_command_render(render_args):  # noqa: ANN001
                captured["render_selection_mode"] = render_args.selection_mode
                captured["render_caption_style"] = render_args.caption_style
                return 0

            with (
                patch.object(pipeline, "load_pipeline_config", return_value=pipeline.DEFAULT_PIPELINE_CONFIG),
                patch.object(pipeline, "list_videos", return_value=[video_path]),
                patch.object(pipeline, "ensure_extract_index", return_value=index_path),
                patch.object(pipeline, "command_infer", side_effect=fake_command_infer),
                patch.object(pipeline, "resolve_video_dir_for_index", return_value=stage_output_dir),
                patch.object(pipeline, "command_temporal", side_effect=fake_command_temporal),
                patch.object(pipeline, "command_render", side_effect=fake_command_render),
            ):
                exit_code = pipeline.command_run(args)

            self.assertEqual(exit_code, 0)
            self.assertEqual(captured["infer_prompt_preset"], "douyin_riding")
            self.assertEqual(captured["temporal_final_duration"], 30.0)
            self.assertEqual(captured["render_selection_mode"], "single_continuous")
            self.assertEqual(captured["render_caption_style"], "douyin")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
