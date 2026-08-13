import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from prepare_false_complete_review import build_review_rows, schema_document, write_outputs


class PrepareFalseCompleteReviewTest(unittest.TestCase):
    def test_template_preserves_outcome_but_assigns_no_false_complete_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            collection = Path(temp_dir) / "collection"
            collection.mkdir()
            manifest = [
                {
                    "episode_index": 3,
                    "suite": "libero_10",
                    "task_id": 0,
                    "task_description": "test task",
                    "seed": 42,
                    "outcome": "failure",
                    "num_frames": 980,
                    "standard_max_steps": 520,
                    "extended_max_steps": 980,
                    "first_success_step": None,
                }
            ]
            (collection / "rollout_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            rows = build_review_rows([collection])

            self.assertEqual(rows[0]["outcome"], "failure")
            self.assertEqual(rows[0]["false_complete"], "")
            self.assertEqual(rows[0]["false_complete_subtypes"], "")
            self.assertEqual(rows[0]["review_status"], "unreviewed")

    def test_schema_keeps_false_complete_independent_and_nullable(self) -> None:
        schema = schema_document()
        self.assertEqual(schema["fields"]["false_complete"]["type"], "nullable_boolean")
        self.assertIn("false_complete is independent of time-budget outcome", schema["invariants"])

    def test_existing_review_artifacts_are_not_overwritten_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_outputs([], output_dir, overwrite=False)
            with self.assertRaises(FileExistsError):
                write_outputs([], output_dir, overwrite=False)


if __name__ == "__main__":
    unittest.main()
