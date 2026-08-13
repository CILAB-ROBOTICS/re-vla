import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from validate_sham_determinism import TRAJECTORY_FIELDS, compare_pair


def write_episode(root: Path, condition: str, diverge: bool = False) -> None:
    episodes = root / "telemetry_v2" / "episodes"
    episodes.mkdir(parents=True)
    rng = {
        "python": "p",
        "numpy_global": "n",
        "torch_cpu": "t",
        "torch_cuda": None,
        "env": {"core": "e"},
        "env_unavailable_reason": None,
    }
    header = {
        "record_type": "episode_start",
        "condition": condition,
        "pair_id": "pair-1",
        "suite": "libero_10",
        "task_id": 0,
        "seed": 1000,
        "task_mapping_hash": "mapping",
        "determinism_audit_enabled": True,
        "initial_observation_hash": "observation",
        "initial_rng_state_hashes": rng,
    }
    frame = {
        "record_type": "frame",
        "timestep": 0,
        "sham_hook_rng_before": rng if condition == "sham" else None,
        "sham_hook_rng_after": rng if condition == "sham" else None,
        "perturbation_triggered": condition == "sham",
    }
    for key in TRAJECTORY_FIELDS:
        frame[key] = rng if key.endswith("rng_state_hashes") else [1.0]
    if diverge:
        frame["action"] = [2.0]
    records = [header, frame, {"record_type": "episode_end"}]
    (episodes / "episode_000000.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8"
    )


class ShamDeterminismValidatorTest(unittest.TestCase):
    def test_exact_pair_passes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_episode(root / "baseline", "baseline")
            write_episode(root / "sham", "sham")
            result = compare_pair(root / "baseline", root / "sham")
            self.assertEqual(result, {"validation": "PASS", "frames": 1, "trigger_count": 1, "bitwise": True})

    def test_trajectory_divergence_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_episode(root / "baseline", "baseline")
            write_episode(root / "sham", "sham", diverge=True)
            with self.assertRaisesRegex(ValueError, "Trajectory divergence"):
                compare_pair(root / "baseline", root / "sham")


if __name__ == "__main__":
    unittest.main()
