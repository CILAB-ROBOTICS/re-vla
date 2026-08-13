import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))

from build_false_complete_review_queue import build_queue


class BuildFalseCompleteReviewQueueTest(unittest.TestCase):
    def test_queue_is_blinded_unlabeled_and_video_addressable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "dataset"
            episode_dir = dataset / "meta" / "episodes" / "chunk-000"
            episode_dir.mkdir(parents=True)
            (dataset / "meta" / "info.json").write_text(
                json.dumps(
                    {
                        "fps": 20,
                        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                    }
                ),
                encoding="utf-8",
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "episode_index": 0,
                            "length": 10,
                            "videos/observation.images.image/chunk_index": 0,
                            "videos/observation.images.image/file_index": 0,
                            "videos/observation.images.image/from_timestamp": 1.0,
                            "videos/observation.images.image/to_timestamp": 1.5,
                            "videos/observation.images.image2/chunk_index": 0,
                            "videos/observation.images.image2/file_index": 0,
                            "videos/observation.images.image2/from_timestamp": 1.0,
                            "videos/observation.images.image2/to_timestamp": 1.5,
                        }
                    ]
                ),
                episode_dir / "file-000.parquet",
            )
            review = [
                {
                    "source_dir": str(Path(temp_dir) / "collection"),
                    "dataset_dir": str(dataset),
                    "episode_index": "0",
                    "suite": "libero_goal",
                    "task_id": "2",
                    "task_description": "test task",
                    "seed": "1000",
                    "outcome": "failure",
                    "num_frames": "10",
                }
            ]

            blinded, key = build_queue(review)

            self.assertNotIn("outcome", blinded[0])
            self.assertNotIn("seed", blinded[0])
            self.assertNotIn("episode_index", blinded[0])
            self.assertNotIn("num_frames", blinded[0])
            self.assertEqual(blinded[0]["false_complete"], "")
            self.assertEqual(blinded[0]["completion_like_behavior"], "")
            self.assertEqual(blinded[0]["agent_from_timestamp"], 1.0)
            self.assertTrue(blinded[0]["agent_video_path"].endswith("file-000.mp4"))
            self.assertEqual(key[0]["outcome"], "failure")
            self.assertEqual(key[0]["review_id"], blinded[0]["review_id"])


if __name__ == "__main__":
    unittest.main()
