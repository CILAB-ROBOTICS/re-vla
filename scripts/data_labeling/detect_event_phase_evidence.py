#!/usr/bin/env python
"""Detect versioned event/phase evidence from label-free geometry features.

This development stage consumes only explicit geometry and kinematics feature
fields. It does not read outcome, reward, done, success, episode length, or any
trajectory/representation-similarity signal, and it does not assign scientific
failure, recovery, terminal-like, subtype, or False Complete labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2] if len(SCRIPT_PATH.parents) > 2 else Path.cwd()
DEFAULT_CONFIG = REPO_ROOT / "research" / "false_complete" / "EVENT_PHASE_DETECTOR_V0_1.json"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


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


def _vector(value: Any, length: int = 3) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    result = [_finite(item) for item in value]
    return None if any(item is None for item in result) else [float(item) for item in result]


def _distance(left: Any, right: Any) -> float | None:
    first, second = _vector(left), _vector(right)
    if first is None or second is None:
        return None
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def _binding(frame: dict[str, Any], binding_index: int) -> dict[str, Any] | None:
    for item in frame.get("bindings", []):
        if isinstance(item, dict) and item.get("binding_index") == binding_index:
            return item
    return None


def _qualified_gripper_runs(frames: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    closed_max = float(config["closed_max"])
    open_min = float(config["open_min"])
    minimum = int(config["minimum_state_frames"])
    raw: list[str | None] = []
    for frame in frames:
        width = _finite(frame.get("gripper_width_proxy"))
        raw.append(None if width is None else "closed" if width <= closed_max else "open" if width >= open_min else None)

    runs: list[dict[str, Any]] = []
    index = 0
    while index < len(raw):
        state = raw[index]
        if state is None:
            index += 1
            continue
        end = index + 1
        while end < len(raw) and raw[end] == state:
            end += 1
        if end - index >= minimum:
            runs.append({"state": state, "start": index, "end": end - 1, "frames": end - index})
        index = end
    return runs


def _gripper_state_by_frame(frame_count: int, runs: list[dict[str, Any]]) -> list[str | None]:
    states: list[str | None] = [None] * frame_count
    current: str | None = None
    by_start = {int(run["start"]): str(run["state"]) for run in runs}
    for index in range(frame_count):
        if index in by_start:
            current = by_start[index]
        states[index] = current
    return states


def _debounced_gripper_events(
    frames: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    minimum_separation: int,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    stable_state: str | None = None
    last_event_start = -10**9
    for run in runs:
        state = str(run["state"])
        start = int(run["start"])
        if stable_state is None:
            stable_state = state
            continue
        if state == stable_state or start - last_event_start < minimum_separation:
            continue
        event_type = "gripper_close_crossing" if state == "closed" else "gripper_open_crossing"
        events.append(_event(event_type, frames[start], previous_state=stable_state, stable_frames=run["frames"]))
        stable_state = state
        last_event_start = start
    return events


def _event(event_type: str, frame: dict[str, Any], **evidence: Any) -> dict[str, Any]:
    return {
        "event_type": event_type,
        "timestep": int(frame["timestep"]),
        "timestamp": float(frame["timestamp"]),
        "evidence": evidence,
    }


def _rising_events(condition: list[bool], frames: list[dict[str, Any]], event_type: str, evidence_fn) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous = False
    for index, active in enumerate(condition):
        if active and not previous:
            events.append(_event(event_type, frames[index], **evidence_fn(index)))
        previous = active
    return events


def detect_episode(frames: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    if not frames:
        raise ValueError("At least one feature frame is required")
    episode_index = int(frames[0]["episode_index"])
    if any(int(frame["episode_index"]) != episode_index for frame in frames):
        raise ValueError("Mixed episode indices")

    runs = _qualified_gripper_runs(frames, config["gripper"])
    gripper_states = _gripper_state_by_frame(len(frames), runs)
    gripper_events = _debounced_gripper_events(
        frames,
        runs,
        int(config["gripper"].get("minimum_transition_separation_frames", 0)),
    )

    binding_indices = sorted({
        int(item["binding_index"])
        for frame in frames
        for item in frame.get("bindings", [])
        if isinstance(item, dict) and isinstance(item.get("binding_index"), int)
    })
    binding_results: list[dict[str, Any]] = []
    for binding_index in binding_indices:
        timeline = [_binding(frame, binding_index) for frame in frames]
        descriptor = next((item for item in timeline if item is not None), {})
        if descriptor.get("binding_kind") != "spatial":
            binding_results.append({
                "binding_index": binding_index,
                "predicate": descriptor.get("predicate"),
                "binding_kind": descriptor.get("binding_kind"),
                "evidence_status": "non_spatial_not_supported_v0.1",
                "events": [],
            })
            continue

        approach_cfg = config["approach"]
        approach_window = int(approach_cfg["window_frames"])
        approach_condition = [False] * len(frames)
        for index in range(approach_window - 1, len(frames)):
            first, last = timeline[index - approach_window + 1], timeline[index]
            start_distance = None if first is None else _finite(first.get("eef_to_target_distance"))
            end_distance = None if last is None else _finite(last.get("eef_to_target_distance"))
            approach_condition[index] = (
                start_distance is not None and end_distance is not None
                and start_distance - end_distance >= float(approach_cfg["minimum_distance_decrease"])
                and end_distance <= float(approach_cfg["near_target_max"])
            )
        events = _rising_events(
            approach_condition,
            frames,
            "approach_entry",
            lambda i: {"binding_index": binding_index, "eef_to_target_distance": timeline[i]["eef_to_target_distance"]},
        )

        close_steps = {event["timestep"] for event in gripper_events if event["event_type"] == "gripper_close_crossing"}
        grasp_steps: list[int] = []
        for index, frame in enumerate(frames):
            item = timeline[index]
            distance = None if item is None else _finite(item.get("eef_to_target_distance"))
            if int(frame["timestep"]) in close_steps and distance is not None and distance <= float(approach_cfg["near_target_max"]):
                grasp_steps.append(int(frame["timestep"]))
                events.append(_event("grasp_attempt_entry", frame, binding_index=binding_index, eef_to_target_distance=distance))

        transport_cfg = config["transport"]
        transport_window = int(transport_cfg["window_frames"])
        transport_condition = [False] * len(frames)
        for index in range(transport_window - 1, len(frames)):
            first, last = timeline[index - transport_window + 1], timeline[index]
            displacement = None if first is None or last is None else _distance(first.get("target_position"), last.get("target_position"))
            eef_distance = None if last is None else _finite(last.get("eef_to_target_distance"))
            transport_condition[index] = (
                displacement is not None and eef_distance is not None
                and displacement >= float(transport_cfg["minimum_target_displacement"])
                and eef_distance <= float(transport_cfg["eef_to_target_max"])
                and gripper_states[index] == "closed"
            )
        transport_events = _rising_events(
            transport_condition,
            frames,
            "transport_entry",
            lambda i: {"binding_index": binding_index, "eef_to_target_distance": timeline[i]["eef_to_target_distance"]},
        )
        events.extend(transport_events)

        goal_cfg = config["goal_approach"]
        goal_window = int(goal_cfg["window_frames"])
        goal_condition = [False] * len(frames)
        for index in range(goal_window - 1, len(frames)):
            first, last = timeline[index - goal_window + 1], timeline[index]
            start_distance = None if first is None else _finite(first.get("target_to_goal_distance"))
            end_distance = None if last is None else _finite(last.get("target_to_goal_distance"))
            goal_condition[index] = (
                start_distance is not None and end_distance is not None
                and start_distance - end_distance >= float(goal_cfg["minimum_distance_decrease"])
                and end_distance <= float(goal_cfg["near_goal_max"])
            )
        events.extend(_rising_events(
            goal_condition,
            frames,
            "goal_approach_entry",
            lambda i: {"binding_index": binding_index, "target_to_goal_distance": timeline[i]["target_to_goal_distance"]},
        ))

        open_events = [event for event in gripper_events if event["event_type"] == "gripper_open_crossing"]
        place_steps: list[int] = []
        prior_manipulation_steps = sorted(grasp_steps + [event["timestep"] for event in transport_events])
        eef_goal_cfg = config.get("eef_goal_approach")
        eef_goal_distances = [
            None if item is None else _distance(frame.get("eef_position"), item.get("goal_position"))
            for frame, item in zip(frames, timeline)
        ]
        if eef_goal_cfg is not None:
            eef_goal_window = int(eef_goal_cfg["window_frames"])
            eef_goal_condition = [False] * len(frames)
            for index in range(eef_goal_window - 1, len(frames)):
                start_distance = eef_goal_distances[index - eef_goal_window + 1]
                end_distance = eef_goal_distances[index]
                eef_goal_condition[index] = (
                    start_distance is not None and end_distance is not None
                    and start_distance - end_distance >= float(eef_goal_cfg["minimum_distance_decrease"])
                    and end_distance <= float(eef_goal_cfg["near_goal_max"])
                )
            events.extend(_rising_events(
                eef_goal_condition,
                frames,
                "eef_goal_approach_entry",
                lambda i: {"binding_index": binding_index, "eef_to_goal_distance": eef_goal_distances[i]},
            ))
        for open_event in open_events:
            step = int(open_event["timestep"])
            index = next((i for i, frame in enumerate(frames) if int(frame["timestep"]) == step), None)
            if index is None or not any(prior < step for prior in prior_manipulation_steps):
                continue
            item = timeline[index]
            goal_distance = None if item is None else _finite(item.get("target_to_goal_distance"))
            prior_grasps = [prior for prior in grasp_steps if prior < step]
            if eef_goal_cfg is None:
                place_evidence = goal_distance is not None and goal_distance <= float(goal_cfg["near_goal_max"])
                eef_goal_distance = None
            else:
                eef_goal_distance = eef_goal_distances[index]
                place_evidence = (
                    bool(prior_grasps)
                    and step - prior_grasps[-1] >= int(eef_goal_cfg["minimum_frames_after_grasp"])
                    and eef_goal_distance is not None
                    and eef_goal_distance <= float(eef_goal_cfg["near_goal_max"])
                )
            if place_evidence:
                place_steps.append(step)
                events.append(_event(
                    "place_attempt_entry",
                    frames[index],
                    binding_index=binding_index,
                    eef_to_goal_distance=eef_goal_distance,
                    target_to_goal_distance=goal_distance,
                    target_transport_observed=any(event["timestep"] < step for event in transport_events),
                ))

        retract_cfg = config["retract"]
        for place_step in place_steps:
            place_index = next(i for i, frame in enumerate(frames) if int(frame["timestep"]) == place_step)
            place_item = timeline[place_index]
            if eef_goal_cfg is None:
                start_distance = None if place_item is None else _finite(place_item.get("eef_to_target_distance"))
                distance_series = [None if item is None else _finite(item.get("eef_to_target_distance")) for item in timeline]
                increase_threshold = float(retract_cfg["minimum_eef_target_increase"])
                evidence_name = "eef_to_target_increase"
            else:
                start_distance = eef_goal_distances[place_index]
                distance_series = eef_goal_distances
                increase_threshold = float(retract_cfg["minimum_eef_goal_increase"])
                evidence_name = "eef_to_goal_increase"
            if start_distance is None:
                continue
            limit = min(len(frames), place_index + int(retract_cfg["window_frames"]) + 1)
            for index in range(place_index + 1, limit):
                distance = distance_series[index]
                if distance is not None and distance - start_distance >= increase_threshold:
                    events.append(_event("retract_entry", frames[index], binding_index=binding_index, **{evidence_name: distance - start_distance}))
                    break

        events.sort(key=lambda item: (item["timestep"], item["event_type"]))
        binding_results.append({
            "binding_index": binding_index,
            "predicate": descriptor.get("predicate"),
            "binding_kind": "spatial",
            "evidence_status": "available",
            "events": events,
        })

    settle_cfg = config["settle"]
    settle_window = int(settle_cfg["window_frames"])
    settle_condition = [False] * len(frames)
    for index in range(settle_window - 1, len(frames)):
        window = frames[index - settle_window + 1:index + 1]
        checks = []
        for frame in window:
            speed = _finite(frame.get("eef_speed"))
            translation = _finite(frame.get("action_translation_norm"))
            rotation = _finite(frame.get("action_rotation_norm"))
            checks.append(
                speed is not None and translation is not None and rotation is not None
                and speed <= float(settle_cfg["eef_speed_max"])
                and translation <= float(settle_cfg["translation_action_norm_max"])
                and rotation <= float(settle_cfg["rotation_action_norm_max"])
            )
        settle_condition[index] = all(checks)
    settle_events = _rising_events(settle_condition, frames, "settle_entry", lambda _i: {"window_frames": settle_window})

    return {
        "record_type": "event_phase_evidence",
        "episode_index": episode_index,
        "frame_count": len(frames),
        "gripper_runs": runs,
        "gripper_events": gripper_events,
        "global_phase_events": settle_events,
        "bindings": binding_results,
        "scientific_labels_assigned": False,
        "failure_events_assigned": False,
        "recovery_assigned": False,
        "terminal_like_assigned": False,
        "false_complete_assigned": False,
    }


def load_feature_episode(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        frames = [json.loads(line) for line in handle if line.strip()]
    if not frames or any(frame.get("record_type") != "geometry_kinematics_features" for frame in frames):
        raise ValueError(f"Invalid feature episode: {path}")
    return frames


def run_collection(input_dir: Path, output_dir: Path, config_path: Path = DEFAULT_CONFIG) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite detector output: {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paths = sorted(input_dir.glob("episode_*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No feature episodes under {input_dir}")
    episode_dir = output_dir / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=False)
    summaries = []
    for path in paths:
        result = detect_episode(load_feature_episode(path), config)
        output_path = episode_dir / f"{path.stem}.json"
        output_path.write_text(_canonical_json(result) + "\n", encoding="utf-8")
        summaries.append({
            "episode_index": result["episode_index"],
            "frame_count": result["frame_count"],
            "input_sha256": file_sha256(path),
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
        "forbidden_proxies_used": [],
        "sealed_reference_used": False,
        "scientific_labels_assigned": False,
        "failure_events_assigned": False,
        "recovery_assigned": False,
        "terminal_like_assigned": False,
        "false_complete_assigned": False,
        "episodes": summaries,
    }
    manifest_path = output_dir / "detector_manifest.json"
    manifest_path.write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    print(run_collection(args.input_dir, args.output_dir, args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
