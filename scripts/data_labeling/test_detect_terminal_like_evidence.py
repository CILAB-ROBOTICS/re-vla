import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from detect_terminal_like_evidence import DEFAULT_CONFIG, detect_episode, run_collection


def event(kind, step):
    return {"event_type": kind, "timestep": step, "timestamp": step / 20, "evidence": {}}


def phase(frame_count=100, retry_after=False, omit_retract=False):
    binding_events = [event("place_attempt_entry", 20)]
    if not omit_retract:
        binding_events.append(event("retract_entry", 30))
    if retry_after:
        binding_events.extend([event("approach_entry", 50), event("grasp_attempt_entry", 55)])
    return {
        "episode_index": 0,
        "frame_count": frame_count,
        "gripper_events": [event("gripper_open_crossing", 20)],
        "global_phase_events": [event("settle_entry", 40)],
        "bindings": [{"binding_index": 0, "evidence_status": "available", "events": binding_events}],
    }


def failure(valid_recovery=False):
    return {
        "episode_index": 0,
        "failure_events": [{
            "failure_event_type": "missed_grasp",
            "failure_timestep": 10,
            "binding_index": 0,
            "next_phase_entry_timestep": 20,
            "valid_recovery_attempt": valid_recovery,
        }],
    }


class TerminalLikeEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))

    def test_ordered_release_retract_settle_without_retry_is_terminal_like(self):
        result = detect_episode(phase(), failure(False), self.config)
        sequence = result["sequences"][0]
        self.assertTrue(sequence["terminal_like"])
        self.assertEqual(sequence["terminal_like_score"], 1.0)

    def test_valid_recovery_or_later_retry_rejects_terminal_like(self):
        self.assertFalse(detect_episode(phase(), failure(True), self.config)["sequences"][0]["terminal_like"])
        self.assertFalse(detect_episode(phase(retry_after=True), failure(False), self.config)["sequences"][0]["terminal_like"])

    def test_insufficient_post_settle_window_stays_unknown(self):
        sequence = detect_episode(phase(frame_count=45), failure(False), self.config)["sequences"][0]
        self.assertIsNone(sequence["terminal_like"])
        self.assertIsNone(sequence["terminal_like_score"])

    def test_missing_required_component_is_false(self):
        sequence = detect_episode(phase(omit_retract=True), failure(False), self.config)["sequences"][0]
        self.assertFalse(sequence["terminal_like"])

    def test_collection_is_versioned_distinct_and_false_complete_free(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); phase_dir = root / "phase"; failure_dir = root / "failure"
            phase_dir.mkdir(); failure_dir.mkdir()
            (phase_dir / "episode_000000.json").write_text(json.dumps(phase()), encoding="utf-8")
            (failure_dir / "episode_000000.json").write_text(json.dumps(failure()), encoding="utf-8")
            output = root / "output"
            manifest = json.loads(run_collection(phase_dir, failure_dir, output).read_text(encoding="utf-8"))
            self.assertEqual(manifest["forbidden_proxies_used"], [])
            self.assertFalse(manifest["sealed_reference_used"])
            self.assertFalse(manifest["false_complete_assigned"])
            with self.assertRaises(FileExistsError):
                run_collection(phase_dir, failure_dir, output)


if __name__ == "__main__":
    unittest.main()
