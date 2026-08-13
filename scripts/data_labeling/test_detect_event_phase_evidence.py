import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from detect_event_phase_evidence import DEFAULT_CONFIG, REPO_ROOT, detect_episode, run_collection


def binding(index, eef_distance, target_x, goal_distance):
    return {
        "binding_index": index,
        "predicate": "in",
        "binding_kind": "spatial",
        "target_position": [target_x, 0.0, 0.0],
        "goal_position": [1.0, 0.0, 0.0],
        "eef_to_target_distance": eef_distance,
        "target_to_goal_distance": goal_distance,
        "target_speed": 0.0,
        "predicate_value": False,
    }


def feature_frame(step, width, eef_distance, target_x, goal_distance, *, settled=False):
    return {
        "record_type": "geometry_kinematics_features",
        "episode_index": 0,
        "timestep": step,
        "timestamp": step / 20,
        "eef_position": [target_x - eef_distance, 0.0, 0.0],
        "eef_speed": 0.001 if settled else 0.04,
        "gripper_width_proxy": width,
        "action_translation_norm": 0.001 if settled else 0.2,
        "action_rotation_norm": 0.001 if settled else 0.04,
        "bindings": [binding(0, eef_distance, target_x, goal_distance)],
    }


def development_config():
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["gripper"]["minimum_state_frames"] = 2
    for key in ("approach", "transport", "goal_approach"):
        config[key]["window_frames"] = 3
    config["retract"]["window_frames"] = 4
    config["settle"]["window_frames"] = 2
    return config


def v02_config():
    path = REPO_ROOT / "research" / "false_complete" / "EVENT_PHASE_DETECTOR_V0_2.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["gripper"]["minimum_state_frames"] = 2
    config["gripper"]["minimum_transition_separation_frames"] = 2
    for key in ("approach", "transport", "goal_approach", "eef_goal_approach"):
        config[key]["window_frames"] = 3
    config["eef_goal_approach"]["near_goal_max"] = 0.35
    config["eef_goal_approach"]["minimum_frames_after_grasp"] = 2
    config["retract"]["window_frames"] = 4
    config["settle"]["window_frames"] = 2
    return config


class EventPhaseEvidenceTest(unittest.TestCase):
    def test_detects_ordered_geometry_kinematics_evidence(self):
        values = [
            (0.08, 0.50, 0.00, 1.00),
            (0.08, 0.30, 0.00, 1.00),
            (0.08, 0.15, 0.00, 1.00),
            (0.01, 0.14, 0.00, 1.00),
            (0.01, 0.13, 0.00, 1.00),
            (0.01, 0.12, 0.00, 1.00),
            (0.01, 0.12, 0.03, 0.70),
            (0.01, 0.11, 0.06, 0.40),
            (0.01, 0.10, 0.09, 0.20),
            (0.08, 0.10, 0.09, 0.20),
            (0.08, 0.11, 0.09, 0.20),
            (0.08, 0.17, 0.09, 0.20),
            (0.08, 0.18, 0.09, 0.20),
        ]
        frames = [feature_frame(step, *value, settled=step >= 11) for step, value in enumerate(values)]
        result = detect_episode(frames, development_config())
        event_types = [event["event_type"] for item in result["bindings"] for event in item["events"]]
        for expected in ("approach_entry", "grasp_attempt_entry", "transport_entry", "goal_approach_entry", "place_attempt_entry", "retract_entry"):
            self.assertIn(expected, event_types)
        self.assertIn("settle_entry", [event["event_type"] for event in result["global_phase_events"]])
        self.assertFalse(result["scientific_labels_assigned"])
        self.assertFalse(result["false_complete_assigned"])

    def test_non_spatial_binding_stays_explicitly_unsupported(self):
        frame = feature_frame(0, 0.08, 0.2, 0.0, 1.0)
        frame["bindings"][0].update({"predicate": "close", "binding_kind": "non_spatial"})
        result = detect_episode([frame], development_config())
        self.assertEqual(result["bindings"][0]["evidence_status"], "non_spatial_not_supported_v0.1")
        self.assertEqual(result["bindings"][0]["events"], [])

    def test_v02_uses_eef_goal_motion_and_debounces_crossings(self):
        values = [
            (0.08, 0.15, 0.00, 1.00),
            (0.08, 0.14, 0.00, 1.00),
            (0.01, 0.13, 0.00, 1.00),
            (0.01, 0.12, 0.00, 1.00),
            (0.08, 0.90, 0.00, 1.00),
            (0.08, 0.85, 0.00, 1.00),
        ]
        frames = [feature_frame(step, *value) for step, value in enumerate(values)]
        for step in (4, 5):
            frames[step]["eef_position"] = [0.90, 0.0, 0.0]
        result = detect_episode(frames, v02_config())
        events = result["bindings"][0]["events"]
        place = [event for event in events if event["event_type"] == "place_attempt_entry"]
        self.assertEqual(len(place), 1)
        self.assertFalse(place[0]["evidence"]["target_transport_observed"])
        self.assertAlmostEqual(place[0]["evidence"]["eef_to_goal_distance"], 0.1)

    def test_collection_is_distinct_versioned_and_label_free(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir = root / "features"
            input_dir.mkdir()
            frames = [feature_frame(step, 0.08, 0.2, 0.0, 1.0) for step in range(3)]
            source = input_dir / "episode_000000.jsonl"
            source.write_text("\n".join(json.dumps(frame) for frame in frames) + "\n", encoding="utf-8")
            output_dir = root / "events"
            config_path = root / "config.json"
            config_path.write_text(json.dumps(development_config()), encoding="utf-8")
            manifest = json.loads(run_collection(input_dir, output_dir, config_path).read_text(encoding="utf-8"))
            self.assertEqual(manifest["episode_count"], 1)
            self.assertEqual(manifest["forbidden_proxies_used"], [])
            self.assertFalse(manifest["sealed_reference_used"])
            self.assertFalse(manifest["scientific_labels_assigned"])
            with self.assertRaises(FileExistsError):
                run_collection(input_dir, output_dir, config_path)


if __name__ == "__main__":
    unittest.main()
