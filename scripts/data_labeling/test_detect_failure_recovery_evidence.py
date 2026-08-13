import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from detect_failure_recovery_evidence import DEFAULT_CONFIG, detect_episode, run_collection


def feature(step, distance=0.15, speed=0.0, predicate=False, width=0.01):
    return {
        "record_type": "geometry_kinematics_features",
        "episode_index": 0,
        "timestep": step,
        "timestamp": step / 20,
        "gripper_width_proxy": width,
        "bindings": [{
            "binding_index": 0,
            "binding_kind": "spatial",
            "eef_to_target_distance": distance,
            "target_speed": speed,
            "predicate_value": predicate,
        }],
    }


def event(event_type, step, **evidence):
    return {"event_type": event_type, "timestep": step, "timestamp": step / 20, "evidence": evidence}


def phase(events):
    return {
        "episode_index": 0,
        "bindings": [{"binding_index": 0, "evidence_status": "available", "events": events}],
    }


class FailureRecoveryEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        self.config["recovery"]["minimum_post_event_observation_frames"] = 2

    def test_missed_grasp_then_valid_successful_recovery(self):
        frames = [feature(step, predicate=step >= 15) for step in range(20)]
        events = [
            event("grasp_attempt_entry", 2),
            event("place_attempt_entry", 6, target_transport_observed=False),
            event("approach_entry", 9),
            event("grasp_attempt_entry", 11),
            event("transport_entry", 14),
        ]
        result = detect_episode(frames, phase(events), self.config)
        self.assertEqual(result["failure_event_count"], 1)
        failure = result["failure_events"][0]
        self.assertEqual(failure["failure_event_type"], "missed_grasp")
        self.assertTrue(failure["valid_recovery_attempt"])
        self.assertTrue(failure["recovery_succeeded"])

    def test_missing_post_event_window_stays_unknown(self):
        frames = [feature(step) for step in range(4)]
        events = [event("grasp_attempt_entry", 2), event("place_attempt_entry", 3, target_transport_observed=False)]
        result = detect_episode(frames, phase(events), self.config)
        failure = result["failure_events"][0]
        self.assertIsNone(failure["valid_recovery_attempt"])
        self.assertIsNone(failure["recovery_succeeded"])

    def test_slip_requires_closed_gripper_motion_and_separation(self):
        frames = [feature(step) for step in range(15)]
        frames[8] = feature(8, distance=0.30, speed=0.05, width=0.01)
        result = detect_episode(frames, phase([event("transport_entry", 4)]), self.config)
        self.assertEqual(result["failure_events"][0]["failure_event_type"], "slip")

    def test_collection_is_versioned_distinct_and_label_free(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            feature_dir, phase_dir = root / "features", root / "phases"
            feature_dir.mkdir(); phase_dir.mkdir()
            frames = [feature(step) for step in range(5)]
            (feature_dir / "episode_000000.jsonl").write_text("\n".join(json.dumps(item) for item in frames) + "\n", encoding="utf-8")
            (phase_dir / "episode_000000.json").write_text(json.dumps(phase([])), encoding="utf-8")
            output_dir = root / "output"
            manifest = json.loads(run_collection(feature_dir, phase_dir, output_dir).read_text(encoding="utf-8"))
            self.assertEqual(manifest["forbidden_proxies_used"], [])
            self.assertFalse(manifest["sealed_reference_used"])
            self.assertFalse(manifest["scientific_episode_labels_assigned"])
            self.assertFalse(manifest["false_complete_assigned"])
            with self.assertRaises(FileExistsError):
                run_collection(feature_dir, phase_dir, output_dir)


if __name__ == "__main__":
    unittest.main()
