import unittest

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from validate_false_complete_annotations import validate_rows


def base_row() -> dict[str, str]:
    return {
        "review_id": "abc",
        "agent_from_timestamp": "10.0",
        "agent_to_timestamp": "20.0",
        "wrist_from_timestamp": "10.0",
        "wrist_to_timestamp": "20.0",
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


class ValidateFalseCompleteAnnotationsTest(unittest.TestCase):
    def test_unreviewed_row_and_one_to_one_key_are_valid(self) -> None:
        summary = validate_rows([base_row()], [{"review_id": "abc", "outcome": "failure"}])
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["unreviewed_rows"], 1)

    def test_subtype_requires_positive_false_complete(self) -> None:
        row = base_row()
        row.update({"false_complete": "false", "false_complete_subtypes": "slip", "reviewer": "r1"})
        summary = validate_rows([row])
        self.assertFalse(summary["valid"])
        self.assertTrue(any("subtypes require" in error for error in summary["errors"]))

    def test_evidence_must_be_inside_clip(self) -> None:
        row = base_row()
        row.update(
            {
                "task_complete_visual": "false",
                "evidence_start_timestamp": "9.0",
                "evidence_end_timestamp": "11.0",
                "reviewer": "r1",
            }
        )
        summary = validate_rows([row])
        self.assertFalse(summary["valid"])
        self.assertTrue(any("outside the episode clip" in error for error in summary["errors"]))

    def test_review_key_must_match_exactly(self) -> None:
        summary = validate_rows([base_row()], [{"review_id": "different", "outcome": "success"}])
        self.assertFalse(summary["valid"])
        self.assertTrue(any("not one-to-one" in error for error in summary["errors"]))


if __name__ == "__main__":
    unittest.main()
