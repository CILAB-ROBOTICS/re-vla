import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_false_complete_annotation_assignments import build_assignments


class BuildAssignmentsTest(unittest.TestCase):
    def test_200_rows_make_50_balanced_overlap_assignments(self) -> None:
        rows = []
        for suite in ("a", "b", "c", "d"):
            for index in range(50):
                rows.append(
                    {
                        "review_id": f"{suite}{index:02d}",
                        "suite": suite,
                        "task_id": str(index % 10),
                        "task_description": "task",
                        "review_clip_path": f"/clips/{suite}{index:02d}.mp4",
                    }
                )
        assignments = build_assignments(rows, "ann_a", "ann_b", 50)
        self.assertEqual(len(assignments), 250)
        counts = Counter(row["annotator_id"] for row in assignments)
        self.assertEqual(counts, {"ann_a": 200, "ann_b": 50})
        overlap = Counter(row["review_id"] for row in assignments)
        suite_overlap = Counter(
            row["suite"] for row in assignments if overlap[row["review_id"]] == 2 and row["annotator_id"] == "ann_b"
        )
        self.assertEqual(sum(suite_overlap.values()), 50)
        self.assertLessEqual(max(suite_overlap.values()) - min(suite_overlap.values()), 1)
        self.assertTrue(all(row["terminal_like_human"] == "" for row in assignments))

    def test_annotators_must_be_distinct(self) -> None:
        with self.assertRaises(ValueError):
            build_assignments([], "same", "same", 0)


if __name__ == "__main__":
    unittest.main()
