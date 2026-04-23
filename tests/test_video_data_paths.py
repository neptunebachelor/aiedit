import os
import unittest
from pathlib import Path
from unittest import mock

from video_data_paths import (
    resolve_video_data_root,
    safe_existing_slug,
    safe_video_slug,
)


class VideoDataPathsTests(unittest.TestCase):
    def test_safe_video_slug_removes_last_suffix_from_paths(self) -> None:
        self.assertEqual(safe_video_slug(Path("/tmp/ride.v1.mp4")), "ride.v1")

    def test_safe_existing_slug_preserves_dotted_slug_text(self) -> None:
        self.assertEqual(safe_existing_slug("ride.v1"), "ride.v1")

    def test_safe_existing_slug_uses_basename_when_path_is_passed(self) -> None:
        self.assertEqual(safe_existing_slug("videos/ride.v1"), "ride.v1")


class ResolveVideoDataRootEnvTests(unittest.TestCase):
    def test_env_override_used_when_no_explicit_override(self) -> None:
        with mock.patch.dict(os.environ, {"RIDE_VIDEO_DATA_ROOT": "/tmp/vd_x"}, clear=False):
            self.assertEqual(resolve_video_data_root(), Path("/tmp/vd_x"))

    def test_explicit_override_beats_env(self) -> None:
        with mock.patch.dict(os.environ, {"RIDE_VIDEO_DATA_ROOT": "/tmp/vd_x"}, clear=False):
            self.assertEqual(resolve_video_data_root(override="/tmp/vd_y"), Path("/tmp/vd_y"))

    def test_env_override_tolerates_nonexistent_path(self) -> None:
        with mock.patch.dict(os.environ, {"RIDE_VIDEO_DATA_ROOT": "/tmp/definitely_not_here_xyz"}, clear=False):
            resolved = resolve_video_data_root()
        self.assertEqual(resolved, Path("/tmp/definitely_not_here_xyz"))

    def test_env_override_expanduser(self) -> None:
        with mock.patch.dict(os.environ, {"RIDE_VIDEO_DATA_ROOT": "~/vd_home"}, clear=False):
            resolved = resolve_video_data_root()
        self.assertEqual(resolved, Path.home() / "vd_home")

    def test_no_override_uses_repo_default(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "RIDE_VIDEO_DATA_ROOT"}
        with mock.patch.dict(os.environ, env, clear=True):
            resolved = resolve_video_data_root()
        self.assertTrue(resolved.name == ".video_data")
        self.assertTrue(resolved.is_absolute())


if __name__ == "__main__":
    unittest.main()
