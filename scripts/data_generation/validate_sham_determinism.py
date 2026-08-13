#!/usr/bin/env python
"""Fail-closed exact comparator for a one-pair baseline-vs-sham smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TRAJECTORY_FIELDS = (
    "action",
    "simulator_state_vector",
    "next_simulator_state_vector",
    "objects",
    "sites",
    "contacts",
    "next_objects",
    "next_sites",
    "next_contacts",
    "eef_position",
    "eef_orientation",
    "gripper_position",
    "gripper_velocity",
    "pre_action_rng_state_hashes",
    "post_action_rng_state_hashes",
)


def _load_episode(root: Path) -> list[dict[str, Any]]:
    files = sorted((root / "telemetry_v2" / "episodes").glob("episode_*.jsonl"))
    if len(files) != 1:
        raise ValueError(f"Expected exactly one episode under {root}, found {len(files)}")
    return [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]


def compare_pair(baseline_root: Path, sham_root: Path) -> dict[str, Any]:
    baseline = _load_episode(baseline_root)
    sham = _load_episode(sham_root)
    if baseline[0].get("condition") != "baseline" or sham[0].get("condition") != "sham":
        raise ValueError("Expected baseline and sham episode headers")
    if not baseline[0].get("pair_id") or not sham[0].get("pair_id"):
        raise ValueError("Both episodes must have a non-empty matched pair_id")
    for key in ("pair_id", "suite", "task_id", "seed", "task_mapping_hash"):
        if baseline[0].get(key) != sham[0].get(key):
            raise ValueError(f"Header mismatch: {key}")
    if not baseline[0].get("determinism_audit_enabled") or not sham[0].get("determinism_audit_enabled"):
        raise ValueError("Both episodes must enable determinism audit")
    for key in ("initial_observation_hash", "initial_rng_state_hashes"):
        if baseline[0].get(key) is None or baseline[0].get(key) != sham[0].get(key):
            raise ValueError(f"Initial-state mismatch or missing field: {key}")
    if baseline[0]["initial_rng_state_hashes"].get("env_unavailable_reason") is not None:
        raise ValueError("Environment RNG state is unavailable; determinism claim is invalid")

    baseline_frames = [record for record in baseline if record.get("record_type") == "frame"]
    sham_frames = [record for record in sham if record.get("record_type") == "frame"]
    if len(baseline_frames) != len(sham_frames):
        raise ValueError("Frame-count mismatch")
    trigger_count = 0
    for index, (base, candidate) in enumerate(zip(baseline_frames, sham_frames, strict=True)):
        if base.get("timestep") != candidate.get("timestep"):
            raise ValueError(f"Timestep mismatch at frame {index}")
        if candidate.get("sham_hook_rng_before") is None:
            raise ValueError(f"Missing sham hook RNG evidence at frame {index}")
        if candidate["sham_hook_rng_before"] != candidate.get("sham_hook_rng_after"):
            raise ValueError(f"Sham hook consumed RNG at frame {index}")
        trigger_count += int(candidate.get("perturbation_triggered") is True)
        for key in TRAJECTORY_FIELDS:
            if base.get(key) is None or base.get(key) != candidate.get(key):
                raise ValueError(f"Trajectory divergence or missing field at frame {index}: {key}")
    if trigger_count != 1:
        raise ValueError(f"Expected exactly one sham trigger, found {trigger_count}")
    return {"validation": "PASS", "frames": len(baseline_frames), "trigger_count": trigger_count, "bitwise": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--sham-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare_pair(args.baseline_root, args.sham_root), sort_keys=True))


if __name__ == "__main__":
    main()
