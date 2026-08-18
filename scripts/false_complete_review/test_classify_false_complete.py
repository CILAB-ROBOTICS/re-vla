import json
import tempfile
import unittest
from pathlib import Path

from classify_false_complete import (
    classify,
    classify_rows,
    detect_profile,
    load_analysis_root,
    summarize,
)


def evidence(**updates):
    row = {
        "task_incomplete": True,
        "failure_event_detected": True,
        "next_phase_entry": True,
        "terminal_like": True,
        "valid_recovery_attempt": False,
        "failure_types": ["missed_grasp"],
    }
    row.update(updates)
    return row


class FalseCompleteSeparatorTest(unittest.TestCase):
    def test_robust_core_pattern_is_false_complete(self):
        label, reason, confidence, score = classify(evidence(terminal_like=False), "robust")
        self.assertEqual(label, "false_complete")
        self.assertEqual(reason, "robust_core_pattern+terminal_not_observed")
        self.assertEqual(confidence, "medium")
        self.assertEqual(score, 0.8)

    def test_terminal_support_raises_confidence(self):
        label, _, confidence, score = classify(evidence())
        self.assertEqual(label, "false_complete")
        self.assertEqual(confidence, "high")
        self.assertEqual(score, 1.0)

    def test_terminal_unknown_does_not_erase_core_pattern(self):
        label, _, confidence, score = classify(evidence(terminal_like=None))
        self.assertEqual(label, "false_complete")
        self.assertEqual(confidence, "medium")
        self.assertEqual(score, 0.8)

    def test_strict_policy_remains_available(self):
        self.assertEqual(classify(evidence(), "strict")[0], "false_complete")
        self.assertEqual(classify(evidence(terminal_like=False), "strict")[0], "failure")
        self.assertEqual(classify(evidence(terminal_like=None), "strict")[0], "uncertain")

    def test_recovery_or_no_next_phase_is_ordinary_failure(self):
        self.assertEqual(classify(evidence(valid_recovery_attempt=True))[0], "failure")
        self.assertEqual(classify(evidence(next_phase_entry=False))[0], "failure")

    def test_unknown_core_evidence_is_uncertain(self):
        self.assertEqual(classify(evidence(failure_event_detected=None))[0], "uncertain")
        self.assertEqual(classify(evidence(task_incomplete=None))[0], "uncertain")

    def test_completed_task_is_not_failure(self):
        self.assertEqual(classify(evidence(task_incomplete=False))[0], "not_failure")

    def test_detector_summary_accepts_only_automatic_fields(self):
        row = {
            "review_id": "r1",
            "suite": "libero_object",
            "task_id": "0",
            "episode_index": "1",
            "seed": "1001",
            "task_incomplete": "true",
            "detector_failure_event_count": "1",
            "detector_next_phase_state": "true",
            "detector_terminal_like_state": "false",
            "detector_valid_recovery_attempt_count": "0",
            "detector_failure_types": "missed_grasp",
            "collection_outcome_posthoc": "success",
            "frame_count": "999",
        }
        fields = set(row)
        profile = detect_profile(fields, "auto")
        self.assertEqual(profile, "detector-summary")
        result = classify_rows([row], profile)
        self.assertEqual(result[0]["classification"], "false_complete")
        self.assertTrue(result[0]["review_recommended"])
        self.assertEqual(result[0]["review_priority"], "high")
        self.assertEqual(result[0]["human_label"], "")
        self.assertEqual(result[0]["human_failure_type"], "")
        self.assertEqual(result[0]["human_notes"], "")
        summary = summarize(result, profile, fields)
        self.assertEqual(
            summary["forbidden_proxy_columns_ignored"],
            ["collection_outcome_posthoc", "frame_count"],
        )

    def test_human_annotation_profile_is_rejected(self):
        human_fields = {
            "task_complete_visual_A",
            "terminal_like_human_A",
            "next_phase_entry_human_A",
            "detector_failure_event_count",
            "detector_valid_recovery_attempt_count",
        }
        with self.assertRaises(ValueError):
            detect_profile(human_fields, "auto")

    def test_normalized_profile(self):
        row = {
            "task_incomplete": "true",
            "failure_event_detected": "true",
            "next_phase_entry": "true",
            "terminal_like": "false",
            "valid_recovery_attempt": "false",
            "failure_types": "slip",
        }
        profile = detect_profile(set(row), "auto")
        self.assertEqual(profile, "normalized")
        result = classify_rows([row], profile)[0]
        self.assertEqual(result["classification"], "false_complete")
        self.assertEqual(result["confidence"], "medium")

    def test_analysis_root_adapter_uses_detector_json_only(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory) / "libero_object"
            for name in ("taxonomy", "failure_recovery", "terminal_like"):
                (suite / name / "episodes").mkdir(parents=True)
            filename = "episode_000003.json"
            (suite / "taxonomy" / "episodes" / filename).write_text(
                json.dumps({"episode_index": 3, "task_complete_simulator": False}),
                encoding="utf-8",
            )
            (suite / "failure_recovery" / "episodes" / filename).write_text(
                json.dumps(
                    {
                        "episode_index": 3,
                        "failure_events": [],
                    }
                ),
                encoding="utf-8",
            )
            (suite / "terminal_like" / "episodes" / filename).write_text(
                json.dumps(
                    {
                        "episode_index": 3,
                        "sequences": [{"terminal_like": False}],
                    }
                ),
                encoding="utf-8",
            )
            rows = load_analysis_root(Path(directory))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["suite"], "libero_object")
            result = classify_rows(rows, "normalized", "robust")[0]
            self.assertEqual(result["classification"], "failure")
            self.assertEqual(result["failure_types"], "")


if __name__ == "__main__":
    unittest.main()
