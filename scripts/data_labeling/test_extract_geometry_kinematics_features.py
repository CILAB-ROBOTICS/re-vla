import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from extract_geometry_kinematics_features import (
    DEFAULT_CONFIG,
    build_goal_bindings,
    extract_frame_features,
    run_collection,
)


def start_record():
    return {
        "record_type": "episode_start",
        "episode_index": 0,
        "task_id": 0,
        "seed": 1000,
        "task_mapping": {
            "goal_state": [
                ["in", "soup_1", "basket_1_contain_region"],
                ["close", "cabinet_1_bottom_region"],
            ]
        },
    }


def frame(timestep, target_x, *, forbidden_values=False):
    record = {
        "record_type": "frame",
        "episode_index": 0,
        "timestep": timestep,
        "timestamp": timestep / 20,
        "action": [0.1, 0.0, 0.0, 0.0, 0.2, 0.0, -1.0],
        "eef_position": [target_x - 0.1, 0.0, 0.0],
        "gripper_position": [0.02, -0.02],
        "gripper_velocity": None if timestep == 0 else [0.1, -0.1],
        "objects": {
            "soup_1": {
                "position": [target_x, 0.0, 0.0],
                "linear_velocity": None,
            }
        },
        "sites": {"basket_1_contain_region": {"position": [1.0, 0.0, 0.0]}},
        "contacts": [{"geom1": "gripper0_finger", "geom2": "soup_1_collision"}],
        "goal_predicates": [
            {"predicate": "in", "arguments": ["soup_1", "basket_1_contain_region"], "value": False, "error": None},
            {"predicate": "close", "arguments": ["cabinet_1_bottom_region"], "value": False, "error": None},
        ],
    }
    if forbidden_values:
        record.update(
            {
                "reward_recorded_not_used_as_detector_proxy": 999,
                "done_recorded_not_used_as_detector_proxy": True,
                "task_success": True,
                "false_complete": True,
            }
        )
    return record


class GeometryKinematicsFeaturesTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))

    def test_goal_binding_separates_spatial_and_non_spatial_predicates(self):
        bindings = build_goal_bindings(start_record(), self.config)
        self.assertEqual(bindings[0].binding_kind, "spatial")
        self.assertEqual(bindings[0].target_entity, "soup_1")
        self.assertEqual(bindings[0].goal_entity, "basket_1_contain_region")
        self.assertEqual(bindings[1].binding_kind, "non_spatial")
        self.assertIsNone(bindings[1].goal_entity)

    def test_raw_geometry_features_and_initial_velocity_null(self):
        bindings = build_goal_bindings(start_record(), self.config)
        first = extract_frame_features(frame(0, 0.0), None, bindings, 20)
        second = extract_frame_features(frame(1, 0.01), frame(0, 0.0), bindings, 20)
        spatial = second["bindings"][0]
        self.assertAlmostEqual(spatial["eef_to_target_distance"], 0.1)
        self.assertAlmostEqual(spatial["target_to_goal_distance"], 0.99)
        self.assertAlmostEqual(spatial["target_speed"], 0.2)
        self.assertEqual(spatial["target_speed_source"], "finite_difference")
        self.assertEqual(spatial["target_contact_count"], 1)
        self.assertTrue(spatial["geometry_available"])
        self.assertIsNone(first["eef_speed"])
        self.assertIsNone(first["gripper_velocity_norm"])

    def test_forbidden_fields_cannot_change_feature_output(self):
        bindings = build_goal_bindings(start_record(), self.config)
        plain = extract_frame_features(frame(1, 0.01), frame(0, 0.0), bindings, 20)
        forbidden = extract_frame_features(
            frame(1, 0.01, forbidden_values=True),
            frame(0, 0.0, forbidden_values=True),
            bindings,
            20,
        )
        self.assertEqual(plain, forbidden)

    def test_collection_output_is_distinct_versioned_and_label_free(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir = root / "telemetry" / "episodes"
            input_dir.mkdir(parents=True)
            records = [start_record(), frame(0, 0.0), frame(1, 0.01), {"record_type": "episode_end"}]
            source = input_dir / "episode_000000.jsonl"
            source.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
            output_dir = root / "derived"
            manifest_path = run_collection(input_dir, output_dir, DEFAULT_CONFIG)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["episode_count"], 1)
            self.assertEqual(manifest["frame_count"], 2)
            self.assertFalse(manifest["scientific_labels_assigned"])
            self.assertFalse(manifest["detector_events_assigned"])
            self.assertEqual(manifest["forbidden_fields_used"], [])
            output_text = (output_dir / "episodes" / source.name).read_text(encoding="utf-8")
            for forbidden in ("task_success", "reward", "done", "false_complete"):
                self.assertNotIn(forbidden, output_text)
            with self.assertRaises(FileExistsError):
                run_collection(input_dir, output_dir, DEFAULT_CONFIG)


if __name__ == "__main__":
    unittest.main()
