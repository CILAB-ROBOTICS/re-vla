import csv
import json
import tempfile
import unittest
from pathlib import Path

from build_rollout_viewer import build_viewer, parse_byte_range


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class RolloutViewerTest(unittest.TestCase):
    def test_explicit_video_path_builds_review_ready_viewer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = root / "dataset"
            (videos / "clips").mkdir(parents=True)
            (videos / "clips" / "episode_000001.mp4").write_bytes(b"video-bytes")
            review = root / "review.csv"
            fields = [
                "review_id",
                "suite",
                "episode_index",
                "review_clip_path",
                "classification",
                "review_recommended",
            ]
            write_csv(
                review,
                fields,
                [
                    {
                        "review_id": "r1",
                        "suite": "libero_object",
                        "episode_index": "1",
                        "review_clip_path": "clips/episode_000001.mp4",
                        "classification": "false_complete",
                        "review_recommended": "True",
                    }
                ],
            )
            output = root / "viewer"
            result = build_viewer(review, videos, output)
            self.assertEqual(result["episodes"], 1)
            self.assertEqual(result["videos"], 1)
            self.assertEqual(result["videos_copied"], 0)
            for asset in ("index.html", "app.js", "styles.css", "episodes.json"):
                self.assertTrue((output / asset).is_file())
            payload = json.loads((output / "episodes.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["episodes"][0]["key"], "r1")
            self.assertEqual(
                payload["episodes"][0]["videos"][0]["url"],
                "/media/clips/episode_000001.mp4",
            )
            self.assertIn("human_label", payload["fields"])
            self.assertEqual(payload["episodes"][0]["row"]["human_label"], "")

    def test_discovers_multiple_cameras_by_suite_and_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = root / "videos"
            for camera in ("agentview", "wrist"):
                path = videos / "libero_goal" / camera / "episode_000003.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(camera.encode())
            review = root / "review.csv"
            write_csv(
                review,
                ["suite", "episode_index", "classification"],
                [{"suite": "libero_goal", "episode_index": "3", "classification": "failure"}],
            )
            output = root / "viewer"
            result = build_viewer(review, videos, output)
            self.assertEqual(result["videos"], 2)
            payload = json.loads((output / "episodes.json").read_text(encoding="utf-8"))
            urls = [item["url"] for item in payload["episodes"][0]["videos"]]
            self.assertEqual(len(urls), 2)
            self.assertTrue(all("libero_goal" in url for url in urls))

    def test_forbidden_unblinded_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = root / "videos"
            videos.mkdir()
            (videos / "episode_000000.mp4").write_bytes(b"x")
            review = root / "review.csv"
            write_csv(
                review,
                ["suite", "episode_index", "outcome"],
                [{"suite": "libero_goal", "episode_index": "0", "outcome": "success"}],
            )
            with self.assertRaisesRegex(ValueError, "unblinded"):
                build_viewer(review, videos, root / "viewer")

    def test_duplicate_episode_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = root / "videos"
            videos.mkdir()
            (videos / "episode_000000.mp4").write_bytes(b"x")
            review = root / "review.csv"
            write_csv(
                review,
                ["suite", "episode_index"],
                [
                    {"suite": "libero_goal", "episode_index": "0"},
                    {"suite": "libero_goal", "episode_index": "0"},
                ],
            )
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                build_viewer(review, videos, root / "viewer")

    def test_empty_video_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = root / "videos"
            videos.mkdir()
            (videos / "episode_000000.mp4").touch()
            review = root / "review.csv"
            write_csv(
                review,
                ["suite", "episode_index", "video_path"],
                [
                    {
                        "suite": "libero_goal",
                        "episode_index": "0",
                        "video_path": "episode_000000.mp4",
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "empty"):
                build_viewer(review, videos, root / "viewer")

    def test_range_parser_supports_seek_and_suffix(self):
        self.assertEqual(parse_byte_range(None, 100), None)
        self.assertEqual(parse_byte_range("bytes=10-19", 100), (10, 19))
        self.assertEqual(parse_byte_range("bytes=90-", 100), (90, 99))
        self.assertEqual(parse_byte_range("bytes=-10", 100), (90, 99))
        with self.assertRaises(ValueError):
            parse_byte_range("bytes=100-101", 100)


if __name__ == "__main__":
    unittest.main()
