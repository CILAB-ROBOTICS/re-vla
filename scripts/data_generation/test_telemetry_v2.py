import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from telemetry_v2 import (
    SCHEMA_VERSION,
    TelemetryV2Writer,
    build_frame_record,
    capture_rng_state_hashes,
    capture_libero_semantics,
    capture_simulator_state,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "research" / "false_complete" / "TELEMETRY_SCHEMA_V2.json"


def raw_observation(gripper=(0.1, 0.2)):
    return {
        "robot_state": {
            "eef": {"pos": [np.array([1, 2, 3])], "quat": [np.array([0, 0, 0, 1])]},
            "gripper": {"qpos": [np.array(gripper)]},
        }
    }


class TelemetryV2Test(unittest.TestCase):
    def test_rng_hash_capture_does_not_consume_python_numpy_or_env_rng(self):
        class FakeCuda:
            @staticmethod
            def is_initialized():
                return False

        class FakeTensor:
            def cpu(self):
                return self

            def numpy(self):
                return np.array([3, 1, 4], dtype=np.uint8)

        fake_torch = type("FakeTorch", (), {"cuda": FakeCuda(), "get_rng_state": staticmethod(FakeTensor)})()
        core = type("Core", (), {"sim": object(), "np_random": np.random.default_rng(17)})()
        wrapper = type("OffScreen", (), {"env": core})()
        member = type("LeRobot", (), {"_env": wrapper, "np_random": np.random.default_rng(19)})()
        vector = type("SyncVector", (), {"envs": [member]})()
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        env_before = core.np_random.bit_generator.state
        first = capture_rng_state_hashes(vector, torch_module=fake_torch)
        second = capture_rng_state_hashes(vector, torch_module=fake_torch)
        self.assertEqual(first, second)
        self.assertIsNone(first["env_unavailable_reason"])
        self.assertEqual(python_before, random.getstate())
        self.assertEqual(numpy_before[0], np.random.get_state()[0])
        np.testing.assert_array_equal(numpy_before[1], np.random.get_state()[1])
        self.assertEqual(env_before, core.np_random.bit_generator.state)

    def test_semantic_capture_preserves_objects_contacts_and_predicates(self):
        class Model:
            @staticmethod
            def geom_id2name(identifier):
                return {1: "gripper", 2: "bowl_geom"}[identifier]

        class Contact:
            geom1, geom2, dist = 1, 2, -0.01

        class Data:
            body_xpos = {7: np.array([0.1, 0.2, 0.3])}
            body_xquat = {7: np.array([1.0, 0.0, 0.0, 0.0])}
            ncon = 1
            contact = [Contact()]

        core = type(
            "Core",
            (),
            {
                "sim": type("Sim", (), {"data": Data(), "model": Model()})(),
                "parsed_problem": {"goal_state": [["In", "bowl", "basket"]]},
                "obj_of_interest": ["bowl"],
                "obj_body_id": {"bowl": 7},
                "objects_dict": {"bowl": object()},
                "fixtures_dict": {},
                "object_states_dict": {},
                "_eval_predicate": lambda self, state: state[0] == "In",
            },
        )()
        wrapper = type("OffScreen", (), {"env": core})()
        lerobot_env = type("LeRobot", (), {"_env": wrapper})()
        vector_env = type("SyncVector", (), {"envs": [lerobot_env]})()
        state, error = capture_libero_semantics(vector_env)
        self.assertIsNone(error)
        self.assertEqual(state["task_mapping"]["objects_of_interest"], ["bowl"])
        self.assertEqual(state["objects"]["bowl"]["position"], [0.1, 0.2, 0.3])
        self.assertTrue(state["goal_predicates"][0]["value"])
        self.assertEqual(state["contacts"][0]["geom2"], "bowl_geom")

    def test_simulator_state_resolves_through_sync_lerobot_libero_layers(self):
        class State:
            def flatten(self):
                return np.array([1.0, 2.0, 3.0])

        class Sim:
            def get_state(self):
                return State()

        core = type("Core", (), {"sim": Sim()})()
        libero_wrapper = type("OffScreen", (), {"env": core})()
        lerobot_env = type("LeRobot", (), {"_env": libero_wrapper})()
        vector_env = type("SyncVector", (), {"envs": [lerobot_env]})()
        state, error = capture_simulator_state(vector_env)
        self.assertEqual(state, [1.0, 2.0, 3.0])
        self.assertIsNone(error)

    def test_simulator_state_fails_closed_for_multiple_vector_envs(self):
        state, error = capture_simulator_state(type("Vector", (), {"envs": [object(), object()]})())
        self.assertIsNone(state)
        self.assertIn("exactly one", error)

    def test_frame_preserves_unknown_semantic_state_as_null(self):
        frame = build_frame_record(
            episode_index=3,
            timestep=2,
            fps=20,
            raw_observation=raw_observation(),
            previous_gripper_position=[0.0, 0.1],
            action=np.arange(7),
            reward=1.0,
            task_success=False,
            done=False,
            simulator_state_vector=[1.0],
            next_simulator_state_vector=[2.0],
        )
        self.assertEqual(frame["schema_version"], SCHEMA_VERSION)
        self.assertEqual(frame["gripper_velocity"], [2.0, 2.0])
        self.assertIsNone(frame["target_object_attached"])
        self.assertIsNone(frame["goal_predicates"])
        self.assertNotIn("false_complete", frame)

    def test_writer_is_append_only_and_assigns_no_scientific_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "telemetry_v2"
            writer = TelemetryV2Writer(root, SCHEMA_PATH)
            writer.begin_episode(
                {
                    "episode_index": 0,
                    "suite": "libero_10",
                    "task_id": 0,
                    "task_description": "task",
                    "seed": 1000,
                }
            )
            writer.write_frame(
                build_frame_record(
                    episode_index=0,
                    timestep=0,
                    fps=20,
                    raw_observation=raw_observation(),
                    previous_gripper_position=None,
                    action=np.zeros(7),
                    reward=0,
                    task_success=False,
                    done=False,
                    simulator_state_vector=None,
                    next_simulator_state_vector=None,
                )
            )
            writer.finish_episode({"episode_index": 0, "num_frames": 1})
            manifest_path = writer.finalize()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(manifest["scientific_labels_assigned"])
            self.assertFalse(manifest["detectors_run"])
            self.assertIsNone(manifest["terminal_detector_version"])
            self.assertIsNone(manifest["rule_config_hash"])
            records = [
                json.loads(line)
                for line in (root / "episodes" / "episode_000000.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([record["record_type"] for record in records], ["episode_start", "frame", "episode_end"])
            self.assertIsNone(records[0]["pair_id"])
            self.assertIsNone(records[0]["perturbation_type"])
            self.assertNotIn("goal_predicates", records[0]["unresolved_task_specific_fields"])
            with self.assertRaises(FileExistsError):
                TelemetryV2Writer(root, SCHEMA_PATH)


if __name__ == "__main__":
    unittest.main()
