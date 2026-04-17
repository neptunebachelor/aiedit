import unittest
from pathlib import Path

from video_data_paths import safe_existing_slug, safe_video_slug


class VideoDataPathsTests(unittest.TestCase):
    def test_safe_video_slug_removes_last_suffix_from_paths(self) -> None:
        self.assertEqual(safe_video_slug(Path("/tmp/ride.v1.mp4")), "ride.v1")

    def test_safe_existing_slug_preserves_dotted_slug_text(self) -> None:
        self.assertEqual(safe_existing_slug("ride.v1"), "ride.v1")

    def test_safe_existing_slug_uses_basename_when_path_is_passed(self) -> None:
        self.assertEqual(safe_existing_slug("videos/ride.v1"), "ride.v1")


if __name__ == "__main__":
    unittest.main()
