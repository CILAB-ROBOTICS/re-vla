#!/usr/bin/env python
"""Detect failure and valid-recovery evidence without outcome-like proxies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2] if len(SCRIPT_PATH.parents) > 2 else Path.cwd()
DEFAULT_CONFIG = REPO_ROOT / "research" / "false_complete" / "FAILURE_RECOVERY_DETECTOR_V0_1.json"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _binding(frame: dict[str, Any], binding_index: int) -> dict[str, Any] | None:
    return next((item for item in frame.get("bindings", []) if item.get("binding_index") == binding_index), None)


def _phase_events(binding: dict[str, Any], event_type: str) -> list[dict[str, Any]]:
    return sorted(
        (event for event in binding.get("events", []) if event.get("event_type") == event_type),
        key=lambda event: int(event["timestep"]),
    )


def _predicate_true_after(frames: list[dict[str, Any]], binding_index: int, start_step: int, end_step: int) -> int | None:
    for frame in frames:
        step = int(frame["timestep"])
        if step < start_step or step > end_step:
            continue
        item = _binding(frame, binding_index)
        if item is not None and item.get("predicate_value") is True:
            return step
    return None


def _recovery_evidence(
    failure_step: int,
    binding: dict[str, Any],
    frames: list[dict[str, Any]],
    binding_index: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    last_step = int(frames[-1]["timestep"])
    minimum_post = int(config["minimum_post_event_observation_frames"])
    approaches = [event for event in _phase_events(binding, "approach_entry") if int(event["timestep"]) > failure_step]
    approaches = [event for event in approaches if int(event["timestep"]) - failure_step <= int(config["maximum_reapproach_delay_frames"])]
    pair = None
    for approach in approaches:
        approach_step = int(approach["timestep"])
        closes = [
            event for event in _phase_events(binding, "grasp_attempt_entry")
            if approach_step <= int(event["timestep"]) <= approach_step + int(config["maximum_close_after_reapproach_frames"])
        ]
        if closes:
            pair = (approach, closes[0])
            break
    if pair is None:
        attempt = None if last_step - failure_step < minimum_post else False
        return {
            "valid_recovery_attempt": attempt,
            "recovery_succeeded": None,
            "recovery_evidence_status": "insufficient_post_event_window" if attempt is None else "no_reapproach_close_pair",
        }

    approach, close = pair
    close_step = int(close["timestep"])
    success_limit = min(last_step, close_step + int(config["maximum_success_delay_frames"]))
    transports = [
        event for event in _phase_events(binding, "transport_entry")
        if close_step < int(event["timestep"]) <= success_limit
    ]
    predicate_step = _predicate_true_after(frames, binding_index, close_step, success_limit)
    if transports or predicate_step is not None:
        succeeded: bool | None = True
        status = "target_transport_or_goal_predicate"
    elif last_step - close_step < minimum_post:
        succeeded = None
        status = "insufficient_post_recovery_window"
    else:
        succeeded = False
        status = "no_transport_or_goal_predicate"
    return {
        "valid_recovery_attempt": True,
        "recovery_succeeded": succeeded,
        "recovery_evidence_status": status,
        "reapproach_timestep": int(approach["timestep"]),
        "regrasp_timestep": close_step,
        "recovery_transport_timestep": None if not transports else int(transports[0]["timestep"]),
        "recovery_predicate_true_timestep": predicate_step,
    }


def detect_episode(
    frames: list[dict[str, Any]],
    phase_record: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    if not frames:
        raise ValueError("At least one feature frame is required")
    episode_index = int(frames[0]["episode_index"])
    if int(phase_record["episode_index"]) != episode_index:
        raise ValueError("Feature/phase episode mismatch")
    frame_by_step = {int(frame["timestep"]): frame for frame in frames}
    failures: list[dict[str, Any]] = []

    for phase_binding in phase_record.get("bindings", []):
        if phase_binding.get("evidence_status") != "available":
            continue
        binding_index = int(phase_binding["binding_index"])
        grasps = _phase_events(phase_binding, "grasp_attempt_entry")
        transports = _phase_events(phase_binding, "transport_entry")
        places = _phase_events(phase_binding, "place_attempt_entry")

        for place in places:
            place_step = int(place["timestep"])
            prior_grasps = [event for event in grasps if int(event["timestep"]) < place_step]
            if not prior_grasps:
                continue
            grasp = prior_grasps[-1]
            grasp_step = int(grasp["timestep"])
            transport_between = [event for event in transports if grasp_step < int(event["timestep"]) < place_step]
            recorded_transport = place.get("evidence", {}).get("target_transport_observed")
            if transport_between or recorded_transport is True:
                continue
            failure = {
                "failure_event_type": "missed_grasp",
                "failure_timestep": grasp_step,
                "failure_evidence_status": "detected",
                "next_phase_entry_timestep": place_step,
                "binding_index": binding_index,
                "evidence": {
                    "grasp_attempt_timestep": grasp_step,
                    "place_attempt_timestep": place_step,
                    "target_transport_observed": False,
                },
            }
            failure.update(_recovery_evidence(grasp_step, phase_binding, frames, binding_index, config["recovery"]))
            failures.append(failure)

        slip_cfg = config["slip"]
        for transport in transports:
            transport_step = int(transport["timestep"])
            start_frame = frame_by_step.get(transport_step)
            start_binding = None if start_frame is None else _binding(start_frame, binding_index)
            start_distance = None if start_binding is None else _finite(start_binding.get("eef_to_target_distance"))
            if start_distance is None:
                continue
            limit = transport_step + int(slip_cfg["window_frames"])
            slip_step = None
            separation = None
            target_speed = None
            for step in range(transport_step + 1, min(limit, int(frames[-1]["timestep"])) + 1):
                frame = frame_by_step.get(step)
                item = None if frame is None else _binding(frame, binding_index)
                width = None if frame is None else _finite(frame.get("gripper_width_proxy"))
                distance = None if item is None else _finite(item.get("eef_to_target_distance"))
                speed = None if item is None else _finite(item.get("target_speed"))
                if width is None or width > float(slip_cfg["closed_width_max"]):
                    break
                if (
                    distance is not None and speed is not None
                    and distance - start_distance >= float(slip_cfg["eef_target_separation_increase"])
                    and speed >= float(slip_cfg["minimum_target_speed"])
                ):
                    slip_step, separation, target_speed = step, distance - start_distance, speed
                    break
            if slip_step is None:
                continue
            failure = {
                "failure_event_type": "slip",
                "failure_timestep": slip_step,
                "failure_evidence_status": "detected",
                "next_phase_entry_timestep": None,
                "binding_index": binding_index,
                "evidence": {
                    "transport_timestep": transport_step,
                    "eef_target_separation_increase": separation,
                    "target_speed": target_speed,
                    "gripper_closed": True,
                },
            }
            failure.update(_recovery_evidence(slip_step, phase_binding, frames, binding_index, config["recovery"]))
            failures.append(failure)

    unique = {}
    for failure in failures:
        key = (failure["failure_event_type"], failure["failure_timestep"], failure["binding_index"])
        unique[key] = failure
    failures = sorted(unique.values(), key=lambda item: (item["failure_timestep"], item["binding_index"], item["failure_event_type"]))
    return {
        "record_type": "failure_recovery_evidence",
        "episode_index": episode_index,
        "frame_count": len(frames),
        "failure_events": failures,
        "failure_event_count": len(failures),
        "valid_recovery_attempt_count": sum(item["valid_recovery_attempt"] is True for item in failures),
        "recovery_succeeded_count": sum(item["recovery_succeeded"] is True for item in failures),
        "scientific_episode_labels_assigned": False,
        "terminal_like_assigned": False,
        "false_complete_assigned": False,
        "subtype_labels_assigned": False,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_collection(feature_dir: Path, phase_dir: Path, output_dir: Path, config_path: Path = DEFAULT_CONFIG) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite recovery output: {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    feature_paths = sorted(feature_dir.glob("episode_*.jsonl"))
    if not feature_paths:
        raise FileNotFoundError(f"No feature episodes under {feature_dir}")
    output_episodes = output_dir / "episodes"
    output_episodes.mkdir(parents=True, exist_ok=False)
    summaries = []
    for feature_path in feature_paths:
        phase_path = phase_dir / f"{feature_path.stem}.json"
        if not phase_path.is_file():
            raise FileNotFoundError(f"Missing phase evidence: {phase_path}")
        result = detect_episode(_load_jsonl(feature_path), json.loads(phase_path.read_text(encoding="utf-8")), config)
        output_path = output_episodes / phase_path.name
        output_path.write_text(_canonical_json(result) + "\n", encoding="utf-8")
        summaries.append({
            "episode_index": result["episode_index"],
            "frame_count": result["frame_count"],
            "failure_event_count": result["failure_event_count"],
            "valid_recovery_attempt_count": result["valid_recovery_attempt_count"],
            "recovery_succeeded_count": result["recovery_succeeded_count"],
            "feature_sha256": file_sha256(feature_path),
            "phase_sha256": file_sha256(phase_path),
            "output_sha256": file_sha256(output_path),
        })
    manifest = {
        "detector_version": config["detector_version"],
        "detector_schema_version": config["schema_version"],
        "detector_status": config["status"],
        "detector_config_hash": canonical_hash(config),
        "detector_code_sha256": file_sha256(SCRIPT_PATH),
        "episode_count": len(summaries),
        "frame_count": sum(item["frame_count"] for item in summaries),
        "failure_event_count": sum(item["failure_event_count"] for item in summaries),
        "valid_recovery_attempt_count": sum(item["valid_recovery_attempt_count"] for item in summaries),
        "recovery_succeeded_count": sum(item["recovery_succeeded_count"] for item in summaries),
        "forbidden_proxies_used": [],
        "sealed_reference_used": False,
        "scientific_episode_labels_assigned": False,
        "terminal_like_assigned": False,
        "false_complete_assigned": False,
        "subtype_labels_assigned": False,
        "episodes": summaries,
    }
    manifest_path = output_dir / "detector_manifest.json"
    manifest_path.write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--phase-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    print(run_collection(args.feature_dir, args.phase_dir, args.output_dir, args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
