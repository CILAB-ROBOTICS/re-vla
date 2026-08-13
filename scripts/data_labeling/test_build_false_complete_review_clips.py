import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))

from build_false_complete_review_clips import (
    FORBIDDEN_REVIEW_FIELDS,
    build_clips,
    clip_bounds,
    ffmpeg_command,
    select_rows,
)


def base_row(review_id: str = "abc") -> dict[str, str]:
    return {
        "review_id": review_id,
        "suite": "libero_goal",
        "task_id": "2",
        "task_description": "put the bowl on the stove",
        "fps": "20",
        "agent_video_path": "/source/agent.mp4",
        "agent_from_timestamp": "10.0",
        "agent_to_timestamp": "15.0",
        "wrist_video_path": "/source/wrist.mp4",
        "wrist_from_timestamp": "10.1",
        "wrist_to_timestamp": "14.8",
        "task_complete_visual": "",
        "completion_like_behavior": "",
        "false_complete": "",
        "false_complete_subtypes": "",
        "confidence": "",
        "evidence_start_timestamp": "",
        "evidence_end_timestamp": "",
        "reviewer": "",
        "review_notes": "",
    }


class BuildFalseCompleteReviewClipsTest(unittest.TestCase):
    def test_clip_bounds_use_shorter_stream_without_outcome(self) -> None:
        self.assertEqual(clip_bounds(base_row()), (10.0, 10.1, 4.700000000000001))

    def test_ffmpeg_command_is_side_by_side_and_reencoded(self) -> None:
        command = ffmpeg_command(
            base_row(), Path("clip.mp4"), ffmpeg="ffmpeg", width=256, crf=25
        )
        self.assertIn("hstack=inputs=2", " ".join(command))
        self.assertIn("libx264", command)
        self.assertNotIn("outcome", " ".join(command))

    def test_selection_is_stable_and_rejects_unknown_id(self) -> None:
        rows = [base_row("b"), base_row("a")]
        self.assertEqual([row["review_id"] for row in select_rows(rows, [], 1)], ["a"])
        with self.assertRaises(ValueError):
            select_rows(rows, ["missing"], None)

    def test_packet_omits_source_and_scientific_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "packet"

            def fake_runner(command: list[str], check: bool, **kwargs: object) -> object:
                self.assertTrue(check)
                if command[0] == "ffmpeg":
                    Path(command[-1]).write_bytes(b"fake-mp4")
                    return SimpleNamespace(stdout="")
                self.assertEqual(command[0], "ffprobe")
                return SimpleNamespace(
                    stdout=json.dumps(
                        {
                            "streams": [
                                {
                                    "width": 512,
                                    "height": 256,
                                    "duration": "4.7",
                                    "nb_read_frames": "94",
                                }
                            ]
                        }
                    )
                )

            packet, manifest = build_clips(
                [base_row()],
                output_dir,
                ffmpeg="ffmpeg",
                ffprobe="ffprobe",
                width=256,
                crf=25,
                overwrite=False,
                packet_path_root="/nas/review_packet",
                runner=fake_runner,
            )

            self.assertFalse(set(packet[0]) & FORBIDDEN_REVIEW_FIELDS)
            self.assertEqual(packet[0]["false_complete"], "")
            self.assertEqual(
                packet[0]["review_clip_path"], "/nas/review_packet/clips/abc.mp4"
            )
            self.assertFalse(manifest["scientific_labels_assigned"])
            self.assertFalse(manifest["outcome_key_used"])
            with (output_dir / "false_complete_review_packet.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                saved = list(csv.DictReader(handle))
            self.assertEqual(saved[0]["review_id"], "abc")
            saved_manifest = json.loads(
                (output_dir / "clip_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_manifest["clips"][0]["size_bytes"], 8)
            self.assertEqual(saved_manifest["clips"][0]["decoded_frames"], 94)

    def test_refuses_to_overwrite_existing_clip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "packet"
            clips_dir = output_dir / "clips"
            clips_dir.mkdir(parents=True)
            (clips_dir / "abc.mp4").write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                build_clips(
                    [base_row()],
                    output_dir,
                    ffmpeg="ffmpeg",
                    ffprobe="ffprobe",
                    width=256,
                    crf=25,
                    overwrite=False,
                )


if __name__ == "__main__":
    unittest.main()
