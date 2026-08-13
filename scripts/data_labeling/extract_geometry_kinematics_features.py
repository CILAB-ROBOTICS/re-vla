#!/usr/bin/env python
"""Extract detector-development geometry/kinematics features from telemetry-v2.

This stage performs task goal binding and raw feature extraction only. It does not
read outcome/reward/success/done fields, and it does not assign events, phases,
terminal-like behavior, recovery, subtypes, or False Complete labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2] if len(SCRIPT_PATH.parents) > 2 else Path.cwd()
DEFAULT_CONFIG = REPO_ROOT / "research" / "false_complete" / "GEOMETRY_KINEMATICS_FEATURES_V0_1.json"


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


def _finite_vector(value: Any, length: int | None = None) -> list[float] | None:
    if not isinstance(value, (list, tuple)):
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if length is not None and len(result) != length:
        return None
    if not all(math.isfinite(item) for item in result):
        return None
    return result


def _distance(first: Any, second: Any) -> float | None:
    left = _finite_vector(first, 3)
    right = _finite_vector(second, 3)
    if left is None or right is None:
        return None
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _norm(value: Any) -> float | None:
    vector = _finite_vector(value)
    if vector is None:
        return None
    return math.sqrt(sum(item * item for item in vector))


def _speed(current: Any, previous: Any, fps: float) -> float | None:
    distance = _distance(current, previous)
    return None if distance is None else distance * fps


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _name_candidates(value: str) -> set[str]:
    normalized = _normalize_name(value)
    without_instance = re.sub(r"_\d+$", "", normalized)
    return {candidate for candidate in (normalized, without_instance) if candidate}


def _contact_mentions(contact: dict[str, Any], entity: str) -> bool:
    entity_names = _name_candidates(entity)
    for key in ("geom1", "geom2"):
        geom = contact.get(key)
        if not isinstance(geom, str):
            continue
        normalized_geom = _normalize_name(geom)
        if any(candidate in normalized_geom for candidate in entity_names):
            return True
    return False


@dataclass(frozen=True)
class GoalBinding:
    binding_index: int
    predicate: str
    arguments: list[str]
    binding_kind: str
    target_entity: str | None
    goal_entity: str | None


def build_goal_bindings(episode_start: dict[str, Any], config: dict[str, Any]) -> list[GoalBinding]:
    mapping = episode_start.get("task_mapping")
    if not isinstance(mapping, dict):
        raise ValueError("episode_start.task_mapping is required")
    goals = mapping.get("goal_state")
    if not isinstance(goals, list) or not goals:
        raise ValueError("episode_start.task_mapping.goal_state must be a nonempty list")

    spatial = {str(item).lower() for item in config["spatial_predicates"]}
    non_spatial = {str(item).lower() for item in config["non_spatial_predicates"]}
    bindings: list[GoalBinding] = []
    for index, raw_goal in enumerate(goals):
        if not isinstance(raw_goal, list) or not raw_goal:
            raise ValueError(f"Invalid goal at index {index}: {raw_goal!r}")
        predicate = str(raw_goal[0]).lower()
        arguments = [str(item) for item in raw_goal[1:]]
        if predicate in spatial and len(arguments) >= 2:
            kind = "spatial"
            target, goal = arguments[0], arguments[1]
        elif predicate in non_spatial:
            kind = "non_spatial"
            target = arguments[0] if arguments else None
            goal = None
        else:
            kind = "unsupported"
            target = arguments[0] if arguments else None
            goal = arguments[1] if len(arguments) > 1 else None
        bindings.append(GoalBinding(index, predicate, arguments, kind, target, goal))
    return bindings


def _entity_record(frame: dict[str, Any], entity: str | None) -> tuple[str | None, dict[str, Any] | None]:
    if entity is None:
        return None, None
    objects = frame.get("objects")
    if isinstance(objects, dict) and isinstance(objects.get(entity), dict):
        return "object", objects[entity]
    sites = frame.get("sites")
    if isinstance(sites, dict) and isinstance(sites.get(entity), dict):
        return "site", sites[entity]
    return None, None


def _predicate_value(frame: dict[str, Any], binding: GoalBinding) -> bool | None:
    predicates = frame.get("goal_predicates")
    if not isinstance(predicates, list) or binding.binding_index >= len(predicates):
        return None
    record = predicates[binding.binding_index]
    if not isinstance(record, dict) or record.get("error"):
        return None
    value = record.get("value")
    return value if isinstance(value, bool) else None


def extract_frame_features(
    frame: dict[str, Any],
    previous_frame: dict[str, Any] | None,
    bindings: list[GoalBinding],
    fps: float,
) -> dict[str, Any]:
    forbidden = {
        "reward_recorded_not_used_as_detector_proxy",
        "done_recorded_not_used_as_detector_proxy",
        "task_success",
        "false_complete",
    }
    # Enforce that this extractor cannot accidentally use these fields below.
    clean_frame = {key: value for key, value in frame.items() if key not in forbidden}
    action = _finite_vector(clean_frame.get("action"))
    eef_position = clean_frame.get("eef_position")
    previous_eef = None if previous_frame is None else previous_frame.get("eef_position")
    gripper_position = _finite_vector(clean_frame.get("gripper_position"))
    contacts = clean_frame.get("contacts") if isinstance(clean_frame.get("contacts"), list) else []

    binding_features: list[dict[str, Any]] = []
    for binding in bindings:
        target_kind, target_record = _entity_record(clean_frame, binding.target_entity)
        goal_kind, goal_record = _entity_record(clean_frame, binding.goal_entity)
        target_position = None if target_record is None else target_record.get("position")
        goal_position = None if goal_record is None else goal_record.get("position")
        previous_target_position = None
        if previous_frame is not None:
            _, previous_target = _entity_record(previous_frame, binding.target_entity)
            if previous_target is not None:
                previous_target_position = previous_target.get("position")
        recorded_target_velocity = None if target_record is None else target_record.get("linear_velocity")
        target_speed = _norm(recorded_target_velocity)
        target_speed_source = "simulator_linear_velocity" if target_speed is not None else None
        if target_speed is None:
            target_speed = _speed(target_position, previous_target_position, fps)
            if target_speed is not None:
                target_speed_source = "finite_difference"

        missing: list[str] = []
        if binding.binding_kind == "spatial":
            if target_record is None:
                missing.append("target_entity")
            if goal_record is None:
                missing.append("goal_entity")

        binding_features.append(
            {
                **asdict(binding),
                "target_entity_kind": target_kind,
                "goal_entity_kind": goal_kind,
                "target_position": _finite_vector(target_position, 3),
                "goal_position": _finite_vector(goal_position, 3),
                "eef_to_target_distance": _distance(eef_position, target_position),
                "target_to_goal_distance": _distance(target_position, goal_position),
                "target_speed": target_speed,
                "target_speed_source": target_speed_source,
                "target_contact_count": sum(
                    _contact_mentions(contact, binding.target_entity)
                    for contact in contacts
                    if binding.target_entity is not None and isinstance(contact, dict)
                ),
                "goal_contact_count": sum(
                    _contact_mentions(contact, binding.goal_entity)
                    for contact in contacts
                    if binding.goal_entity is not None and isinstance(contact, dict)
                ),
                "predicate_value": _predicate_value(clean_frame, binding),
                "geometry_available": binding.binding_kind == "spatial" and not missing,
                "missing_geometry": missing,
            }
        )

    return {
        "record_type": "geometry_kinematics_features",
        "episode_index": int(clean_frame["episode_index"]),
        "timestep": int(clean_frame["timestep"]),
        "timestamp": float(clean_frame["timestamp"]),
        "eef_position": _finite_vector(eef_position, 3),
        "eef_speed": _speed(eef_position, previous_eef, fps),
        "gripper_position": gripper_position,
        "gripper_width_proxy": None if gripper_position is None else sum(abs(item) for item in gripper_position),
        "gripper_velocity_norm": _norm(clean_frame.get("gripper_velocity")),
        "action_translation_norm": None if action is None else _norm(action[:3]),
        "action_rotation_norm": None if action is None else _norm(action[3:6]),
        "action_gripper_command": None if action is None or len(action) < 7 else action[6],
        "total_contact_count": len(contacts),
        "bindings": binding_features,
    }


def load_episode(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if len(records) < 3 or records[0].get("record_type") != "episode_start":
        raise ValueError(f"Invalid telemetry episode: {path}")
    frames = [record for record in records[1:] if record.get("record_type") == "frame"]
    if not frames:
        raise ValueError(f"No telemetry frames: {path}")
    return records[0], frames


def extract_episode(path: Path, output_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    start, frames = load_episode(path)
    bindings = build_goal_bindings(start, config)
    fps = float(config["fps"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        for index, frame in enumerate(frames):
            features = extract_frame_features(frame, None if index == 0 else frames[index - 1], bindings, fps)
            handle.write(_canonical_json(features) + "\n")
    return {
        "episode_index": int(start["episode_index"]),
        "task_id": int(start["task_id"]),
        "seed": int(start["seed"]),
        "num_frames": len(frames),
        "bindings": [asdict(binding) for binding in bindings],
        "input_sha256": file_sha256(path),
        "output_sha256": file_sha256(output_path),
    }


def run_collection(input_dir: Path, output_dir: Path, config_path: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite feature output: {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_hash = canonical_hash(config)
    episodes = sorted(input_dir.glob("episode_*.jsonl"))
    if not episodes:
        raise FileNotFoundError(f"No telemetry episodes under {input_dir}")
    output_episodes = output_dir / "episodes"
    output_episodes.mkdir(parents=True, exist_ok=False)
    summaries = [
        extract_episode(path, output_episodes / path.name, config)
        for path in episodes
    ]
    manifest = {
        "feature_extractor_version": config["feature_extractor_version"],
        "feature_schema_version": config["schema_version"],
        "feature_config_hash": config_hash,
        "input_directory": str(input_dir),
        "episode_count": len(summaries),
        "frame_count": sum(item["num_frames"] for item in summaries),
        "scientific_labels_assigned": False,
        "detector_events_assigned": False,
        "forbidden_fields_used": [],
        "episodes": summaries,
    }
    manifest_path = output_dir / "feature_manifest.json"
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
    manifest = run_collection(args.input_dir, args.output_dir, args.config)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
