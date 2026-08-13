#!/usr/bin/env python
"""Versioned, append-only telemetry-v2 sidecars for new LIBERO rollouts.

This module deliberately does not derive events, phases, terminal-like behavior, or
False Complete labels.  It stores detector inputs and explicit nulls for unavailable
signals so later rules can be rerun without changing the source rollout.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "2.0.0-draft"


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if hasattr(value, "flatten"):
        return _json_value(np.asarray(value).flatten())
    raise TypeError(f"Unsupported telemetry value type: {type(value)!r}")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def capture_rng_state_hashes(env: Any, torch_module: Any | None = None) -> dict[str, Any]:
    """Hash RNG states without sampling or mutating them.

    Environment RNG capture is fail-closed: a missing ``np_random`` state is
    represented explicitly and must make a determinism validator fail rather than
    silently claiming equivalence.
    """
    if torch_module is None:
        import torch as torch_module  # Local import keeps schema-only tooling light.

    hashes: dict[str, Any] = {
        "python": canonical_hash(random.getstate()),
        "numpy_global": canonical_hash(np.random.get_state()),
        "torch_cpu": canonical_hash(torch_module.get_rng_state().cpu().numpy()),
        "torch_cuda": None,
        "env": {},
        "env_unavailable_reason": None,
    }
    try:
        if torch_module.cuda.is_initialized():
            hashes["torch_cuda"] = [
                canonical_hash(state.cpu().numpy()) for state in torch_module.cuda.get_rng_state_all()
            ]
    except Exception as exc:  # CUDA state is diagnostic; preserve explicit unavailability.
        hashes["torch_cuda_unavailable_reason"] = f"{type(exc).__name__}: {exc}"

    try:
        vector_member = env.envs[0]
        core = get_libero_core_env(env)
        candidates = (("vector_member", vector_member), ("core", core))
        for name, owner in candidates:
            rng = getattr(owner, "np_random", None)
            if rng is None:
                continue
            if hasattr(rng, "bit_generator"):
                state = rng.bit_generator.state
            elif hasattr(rng, "get_state"):
                state = rng.get_state()
            else:
                continue
            hashes["env"][name] = canonical_hash(state)
        if not hashes["env"]:
            hashes["env_unavailable_reason"] = "No readable np_random state on vector member or LIBERO core"
    except Exception as exc:
        hashes["env_unavailable_reason"] = f"{type(exc).__name__}: {exc}"
    return hashes


def load_schema_contract(path: Path) -> tuple[dict[str, Any], str]:
    with path.open(encoding="utf-8") as handle:
        schema = json.load(handle)
    if schema.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Telemetry schema mismatch: module={SCHEMA_VERSION!r}, contract={schema.get('schema_version')!r}"
        )
    return schema, canonical_hash(schema)


def extract_robot_telemetry(
    raw_observation: dict[str, Any], previous_gripper_position: list[float] | None, fps: float
) -> dict[str, Any]:
    robot_state = raw_observation["robot_state"]
    eef = robot_state["eef"]
    gripper = robot_state["gripper"]
    gripper_position = np.asarray(gripper["qpos"][0], dtype=np.float64).reshape(-1)
    if previous_gripper_position is None:
        gripper_velocity = None
    else:
        previous = np.asarray(previous_gripper_position, dtype=np.float64).reshape(-1)
        gripper_velocity = ((gripper_position - previous) * float(fps)).tolist()
    return {
        "eef_position": np.asarray(eef["pos"][0], dtype=np.float64).reshape(-1).tolist(),
        "eef_orientation": np.asarray(eef["quat"][0], dtype=np.float64).reshape(-1).tolist(),
        "gripper_position": gripper_position.tolist(),
        "gripper_velocity": gripper_velocity,
    }


def get_libero_core_env(vector_env: Any) -> Any:
    """Resolve SyncVectorEnv -> LeRobot LiberoEnv -> LIBERO task env.

    Telemetry-v2 is intentionally restricted to synchronous single-environment
    collection.  Failing closed here avoids silently pairing state from the wrong
    vector member.
    """
    envs = getattr(vector_env, "envs", None)
    if envs is None or len(envs) != 1:
        raise ValueError("Telemetry-v2 requires a synchronous vector env containing exactly one environment")
    lerobot_env = getattr(envs[0], "unwrapped", envs[0])
    libero_wrapper = getattr(lerobot_env, "_env", None)
    if libero_wrapper is None:
        raise AttributeError("LeRobot LiberoEnv._env is unavailable")
    core_env = getattr(libero_wrapper, "env", libero_wrapper)
    if not hasattr(core_env, "sim"):
        raise AttributeError("LIBERO core simulator is unavailable")
    return core_env


def capture_simulator_state(env: Any) -> tuple[list[float] | None, str | None]:
    """Read simulator state without inventing values when an environment lacks the API."""
    try:
        state = get_libero_core_env(env).sim.get_state().flatten()
    except Exception as exc:  # Environment-version compatibility boundary.
        return None, f"{type(exc).__name__}: {exc}"
    try:
        return np.asarray(state, dtype=np.float64).reshape(-1).tolist(), None
    except (TypeError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _safe_name(model: Any, kind: str, identifier: int) -> str | None:
    resolver = getattr(model, f"{kind}_id2name", None)
    if resolver is None:
        return None
    try:
        return resolver(int(identifier))
    except Exception:
        return None


def capture_libero_semantics(env: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Capture raw task/object/contact/goal state without assigning scientific labels."""
    try:
        core = get_libero_core_env(env)
        sim = core.sim
        parsed_problem = getattr(core, "parsed_problem", {})
        goal_state = _json_value(parsed_problem.get("goal_state", []))
        objects_of_interest = _json_value(
            getattr(core, "obj_of_interest", parsed_problem.get("obj_of_interest", []))
        )

        objects: dict[str, Any] = {}
        object_models = getattr(core, "objects_dict", {})
        fixture_models = getattr(core, "fixtures_dict", {})
        for name, body_id in sorted(getattr(core, "obj_body_id", {}).items()):
            record: dict[str, Any] = {
                "kind": "object" if name in object_models else "fixture" if name in fixture_models else "body",
                "position": _json_value(sim.data.body_xpos[body_id]),
                "orientation_wxyz": _json_value(sim.data.body_xquat[body_id]),
                "linear_velocity": None,
                "angular_velocity": None,
            }
            try:
                record["linear_velocity"] = _json_value(sim.data.body_xvelp[body_id])
                record["angular_velocity"] = _json_value(sim.data.body_xvelr[body_id])
            except Exception:
                pass
            objects[str(name)] = record

        sites: dict[str, Any] = {}
        for name, state in sorted(getattr(core, "object_states_dict", {}).items()):
            if getattr(state, "object_state_type", None) != "site":
                continue
            try:
                geom = state.get_geom_state()
                sites[str(name)] = {
                    "position": _json_value(geom.get("pos")),
                    "orientation": _json_value(geom.get("quat")),
                }
            except Exception as exc:
                sites[str(name)] = {"position": None, "orientation": None, "error": type(exc).__name__}

        predicates: list[dict[str, Any]] = []
        evaluator = getattr(core, "_eval_predicate", None)
        for state in goal_state:
            value = None
            error = None
            if evaluator is not None:
                try:
                    value = bool(evaluator(state))
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            predicates.append(
                {
                    "predicate": state[0] if state else None,
                    "arguments": list(state[1:]) if state else [],
                    "value": value,
                    "error": error,
                }
            )

        contacts: list[dict[str, Any]] = []
        for index in range(int(getattr(sim.data, "ncon", 0))):
            contact = sim.data.contact[index]
            contacts.append(
                {
                    "geom1": _safe_name(sim.model, "geom", contact.geom1),
                    "geom2": _safe_name(sim.model, "geom", contact.geom2),
                    "distance": _json_value(getattr(contact, "dist", None)),
                }
            )

        return {
            "task_mapping": {
                "objects_of_interest": objects_of_interest,
                "goal_state": goal_state,
            },
            "objects": objects,
            "sites": sites,
            "goal_predicates": predicates,
            "contacts": contacts,
        }, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def capture_step_state(
    env: Any,
) -> tuple[list[float] | None, str | None, dict[str, Any] | None, str | None]:
    simulator_state, simulator_error = capture_simulator_state(env)
    semantic_state, semantic_error = capture_libero_semantics(env)
    return simulator_state, simulator_error, semantic_state, semantic_error


def build_frame_record(
    *,
    episode_index: int,
    timestep: int,
    fps: float,
    raw_observation: dict[str, Any],
    previous_gripper_position: list[float] | None,
    action: Any,
    reward: float,
    task_success: bool,
    done: bool,
    simulator_state_vector: list[float] | None,
    next_simulator_state_vector: list[float] | None,
    semantic_state: dict[str, Any] | None = None,
    next_semantic_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    robot = extract_robot_telemetry(raw_observation, previous_gripper_position, fps)
    return {
        "record_type": "frame",
        "schema_version": SCHEMA_VERSION,
        "episode_index": int(episode_index),
        "timestep": int(timestep),
        "timestamp": float(timestep) / float(fps),
        "action": _json_value(action),
        **robot,
        "simulator_state_vector": simulator_state_vector,
        "next_simulator_state_vector": next_simulator_state_vector,
        "objects": None if semantic_state is None else semantic_state.get("objects"),
        "sites": None if semantic_state is None else semantic_state.get("sites"),
        "contacts": None if semantic_state is None else semantic_state.get("contacts"),
        "next_objects": None if next_semantic_state is None else next_semantic_state.get("objects"),
        "next_sites": None if next_semantic_state is None else next_semantic_state.get("sites"),
        "next_contacts": None if next_semantic_state is None else next_semantic_state.get("contacts"),
        "target_object_position": None,
        "target_object_orientation": None,
        "target_object_linear_velocity": None,
        "target_object_angular_velocity": None,
        "target_object_attached": None,
        "target_object_contacts": None,
        "goal_region_position": None,
        "goal_predicates": None if semantic_state is None else semantic_state.get("goal_predicates"),
        "next_goal_predicates": None
        if next_semantic_state is None
        else next_semantic_state.get("goal_predicates"),
        "task_success": bool(task_success),
        "reward_recorded_not_used_as_detector_proxy": float(reward),
        "done_recorded_not_used_as_detector_proxy": bool(done),
        "perturbation_scheduled": False,
        "perturbation_triggered": False,
        "perturbation_parameters": None,
    }


class TelemetryV2Writer:
    """Write one JSONL per episode plus a versioned collection manifest."""

    def __init__(self, root: Path, schema_path: Path) -> None:
        self.root = root
        if root.exists():
            raise FileExistsError(f"Refusing to overwrite telemetry-v2 output: {root}")
        self.schema, self.schema_hash = load_schema_contract(schema_path)
        self.episodes_dir = root / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=False)
        self._handle = None
        self._current_episode: int | None = None
        self._episode_summaries: list[dict[str, Any]] = []

    def begin_episode(self, metadata: dict[str, Any]) -> None:
        if self._handle is not None:
            raise RuntimeError("Cannot begin an episode while another telemetry episode is open")
        episode_index = int(metadata["episode_index"])
        path = self.episodes_dir / f"episode_{episode_index:06d}.jsonl"
        self._handle = path.open("x", encoding="utf-8", newline="\n")
        self._current_episode = episode_index
        header = {
            "record_type": "episode_start",
            "schema_version": SCHEMA_VERSION,
            "schema_hash": self.schema_hash,
            "collector_version": "collect-libero-rollouts-telemetry-v2.0",
            "condition": "baseline",
            "event_origin": "natural",
            "pair_id": None,
            "perturbation_type": None,
            "perturbation_config_hash": None,
            "telemetry_completeness": "v2_raw_state_with_nullable_task_semantics",
            "event_detector_version": None,
            "event_config_hash": None,
            "phase_detector_version": None,
            "phase_config_hash": None,
            "terminal_detector_version": None,
            "terminal_config_hash": None,
            "rule_version": None,
            "rule_config_hash": None,
            "unresolved_task_specific_fields": [
                "target_object_position",
                "target_object_orientation",
                "target_object_linear_velocity",
                "target_object_angular_velocity",
                "target_object_attached",
                "target_object_contacts",
                "goal_region_position",
            ],
            **metadata,
        }
        self._write(header)

    def write_frame(self, frame: dict[str, Any]) -> None:
        if self._handle is None or self._current_episode != int(frame["episode_index"]):
            raise RuntimeError("Telemetry frame does not match the open episode")
        self._write(frame)

    def finish_episode(self, summary: dict[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("No telemetry episode is open")
        self._write({"record_type": "episode_end", "schema_version": SCHEMA_VERSION, **summary})
        self._handle.close()
        self._handle = None
        self._episode_summaries.append(_json_value(summary))
        self._current_episode = None

    def close_partial(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None
            self._current_episode = None

    def finalize(self) -> Path:
        if self._handle is not None:
            raise RuntimeError("Cannot finalize telemetry while an episode is open")
        path = self.root / "telemetry_manifest.json"
        payload = {
            "schema_name": self.schema["schema_name"],
            "schema_version": SCHEMA_VERSION,
            "schema_hash": self.schema_hash,
            "scientific_labels_assigned": False,
            "detectors_run": False,
            "event_detector_version": None,
            "event_config_hash": None,
            "phase_detector_version": None,
            "phase_config_hash": None,
            "terminal_detector_version": None,
            "terminal_config_hash": None,
            "rule_version": None,
            "rule_config_hash": None,
            "episodes": self._episode_summaries,
        }
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
        return path

    def _write(self, record: dict[str, Any]) -> None:
        assert self._handle is not None
        self._handle.write(json.dumps(_json_value(record), sort_keys=True, allow_nan=False) + "\n")
        self._handle.flush()
