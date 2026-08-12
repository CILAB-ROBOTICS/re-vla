#!/usr/bin/env python
"""Roll out a fine-tuned VLA policy on LIBERO and collect success / failure /
recoverable-failure episodes, including the recovery behavior itself.

Policy-agnostic: `--policy-path` is loaded via `PreTrainedConfig.from_pretrained`, which
reads the policy type from the checkpoint's own config.json, so this works unchanged for
any lerobot policy (SmolVLA, Pi0.5, ...). See the `collect_libero_rollouts_smolvla.sh`
and `collect_libero_rollouts_pi05.sh` wrappers for the two currently supported recipes.

Outcome definition ("extended-rollout" rule)
---------------------------------------------
Every LIBERO suite has a standard step budget (``standard_max_steps``, e.g. 280 steps
for libero_spatial). For each episode we keep stepping the policy past that budget, up
to ``extended_max_steps = round(standard_max_steps * --extension-factor)``:

  - success             : the task's success condition is met at or before
                           `standard_max_steps`.
  - recoverable_failure : NOT solved within `standard_max_steps`, but the policy keeps
                           going and solves it before `extended_max_steps`. The frames
                           captured after `standard_max_steps` are the recovery behavior
                           and are tagged `is_recovery_phase=True` in the dataset.
  - failure             : still not solved by `extended_max_steps` (unrecoverable within
                           the extended budget).

All episodes (whatever the outcome) are written as one LeRobotDataset (images, state,
action, reward, success/done flags, `is_recovery_phase`). A `rollout_manifest.json`
sidecar records the per-episode outcome/task/seed summary without needing to decode
video.

Usage
-----
    python scripts/collect_libero_rollouts.py \\
        --policy-path outputs/libero_smolvla/checkpoints/last/pretrained_model \\
        --task libero_10 \\
        --episodes-per-task 10 \\
        --output-dir ./outputs/libero_rollouts \\
        --repo-id local/libero_smolvla_rollouts

Notes
-----
This script imports lerobot's env/policy factories from their submodules
(``lerobot.envs.factory``, ``lerobot.envs.configs``, ...) rather than the top-level
``lerobot.envs``/``lerobot.policies`` packages, since some lerobot checkouts don't
re-export these names at the package level. The submodule paths are stable across
recent lerobot versions.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STR
from lerobot.utils.random_utils import set_seed

from telemetry_v2 import (
    TelemetryV2Writer,
    build_frame_record,
    canonical_hash,
    capture_libero_semantics,
    capture_step_state,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("collect_libero_rollouts")

OUTCOME_SUCCESS = "success"
OUTCOME_RECOVERABLE_FAILURE = "recoverable_failure"
OUTCOME_FAILURE = "failure"

# robosuite's underlying env has a *fixed* horizon (default 1000, hardcoded in LIBERO's
# env_wrapper.py / bddl_base_domain.py and not exposed through lerobot's LiberoEnv). If a
# step is taken once robosuite's internal `self.timestep >= self.horizon`, it raises
# ValueError("executing action in terminated episode") on the *next* call - and LIBERO's
# bddl_base_domain.step() overwrites the returned `done` with the task-success check, so
# horizon expiry is invisible to our termination check (`is_success`/`terminated`) above
# this layer. extended_max_steps must therefore never approach this hard limit; the
# safety margin covers LiberoEnv.reset()'s internal warmup steps (num_steps_wait=10),
# which also count against the horizon.
LIBERO_ROBOSUITE_HORIZON = 1000
LIBERO_HORIZON_SAFETY_MARGIN = 20

# Matches LiberoEnv's default camera_name_mapping (agentview -> image, wrist -> image2).
CAMERA_KEYS = ["image", "image2"]
# [eef_pos(3), eef_quat(4), gripper_qpos(2)]. NOTE: this is a raw concatenation for
# analysis purposes, not the axis-angle 8-dim `observation.state` layout used by the
# published lerobot/libero training dataset.
STATE_DIM = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect LIBERO rollouts labeled success/failure/recoverable_failure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--policy-path",
        required=True,
        help="Local dir or Hub id of the fine-tuned policy checkpoint (any lerobot policy type, "
        "e.g. SmolVLA or Pi0.5 — the type is read from the checkpoint's own config.json).",
    )
    parser.add_argument("--task", default="libero_10", help="Comma-separated LIBERO suite name(s).")
    parser.add_argument(
        "--task-ids", type=int, nargs="*", default=None, help="Restrict to these task ids (default: all in suite)."
    )
    parser.add_argument("--episodes-per-task", type=int, default=5)
    parser.add_argument("--control-mode", default="relative", choices=["relative", "absolute"])
    parser.add_argument(
        "--extension-factor",
        type=float,
        default=2.0,
        help="extended_max_steps = round(standard_max_steps * extension_factor).",
    )
    parser.add_argument("--seed", type=int, default=1000, help="Seed for episode 0; incremented per episode.")
    parser.add_argument("--device", default=None, help="Override policy device (e.g. cuda, cpu).")
    parser.add_argument(
        "--rename-map",
        type=json.loads,
        default=None,
        help="JSON mapping from environment feature names to policy feature names, matching lerobot_eval.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-id", default="local/libero_rollouts", help="Dataset repo id (used for local dir naming).")
    parser.add_argument(
        "--telemetry-v2",
        action="store_true",
        help="Write append-only telemetry-v2 JSONL sidecars. Requires a fresh output directory.",
    )
    return parser.parse_args()


def get_image_hw(env) -> tuple[int, int]:
    """Read the true rendered image size from the env's observation space.

    Some LiberoEnv config versions declare `observation_height`/`observation_width`
    fields that aren't actually threaded through to the underlying gym env (it falls
    back to its own default). Reading the live observation space is version-agnostic.
    """
    shape = env.single_observation_space["pixels"][CAMERA_KEYS[0]].shape
    return int(shape[0]), int(shape[1])


def build_dataset_features(observation_height: int, observation_width: int) -> dict:
    features: dict[str, Any] = {}
    for key in CAMERA_KEYS:
        features[f"{OBS_IMAGES}.{key}"] = {
            "dtype": "video",
            "shape": (observation_height, observation_width, 3),
            "names": ["height", "width", "channel"],
        }
    features[f"{OBS_STR}.state"] = {"dtype": "float32", "shape": (STATE_DIM,), "names": None}
    features[ACTION] = {"dtype": "float32", "shape": (7,), "names": None}
    features["next.reward"] = {"dtype": "float32", "shape": (1,), "names": None}
    features["next.success"] = {"dtype": "bool", "shape": (1,), "names": None}
    features["next.done"] = {"dtype": "bool", "shape": (1,), "names": None}
    features["is_recovery_phase"] = {"dtype": "bool", "shape": (1,), "names": None}
    return features


def build_raw_state(raw_obs: dict) -> np.ndarray:
    rs = raw_obs["robot_state"]
    eef_pos = np.asarray(rs["eef"]["pos"][0], dtype=np.float32)
    eef_quat = np.asarray(rs["eef"]["quat"][0], dtype=np.float32)
    gripper_qpos = np.asarray(rs["gripper"]["qpos"][0], dtype=np.float32)
    return np.concatenate([eef_pos, eef_quat, gripper_qpos], axis=0)


def build_frame(
    raw_obs: dict,
    action: np.ndarray,
    reward: float,
    success: bool,
    done: bool,
    is_recovery_phase: bool,
    task_desc: str,
) -> dict:
    frame: dict[str, Any] = {}
    for key in CAMERA_KEYS:
        # LIBERO's raw MuJoCo camera images are upside-down and mirrored; lerobot's
        # LiberoProcessorStep flips both H and W before feeding the policy, matching the
        # orientation convention of the published training dataset (see its docstring).
        # Apply the same flip here so recorded frames match what the policy sees / what a
        # human reviewing the video expects, instead of storing raw (upside-down) images.
        frame[f"{OBS_IMAGES}.{key}"] = np.ascontiguousarray(raw_obs["pixels"][key][0][::-1, ::-1])
    frame[f"{OBS_STR}.state"] = build_raw_state(raw_obs)
    frame[ACTION] = action
    frame["next.reward"] = np.atleast_1d(np.float32(reward))
    frame["next.success"] = np.atleast_1d(np.bool_(success))
    frame["next.done"] = np.atleast_1d(np.bool_(done))
    frame["is_recovery_phase"] = np.atleast_1d(np.bool_(is_recovery_phase))
    frame["task"] = task_desc
    return frame


def classify_outcome(first_success_step: int | None, standard_max_steps: int) -> str:
    if first_success_step is None:
        return OUTCOME_FAILURE
    if first_success_step < standard_max_steps:
        return OUTCOME_SUCCESS
    return OUTCOME_RECOVERABLE_FAILURE


def run_episode(
    env,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    dataset: LeRobotDataset,
    seed: int,
    standard_max_steps: int,
    extended_max_steps: int,
    episode_index: int,
    suite_name: str,
    task_id: int,
    fps: float,
    telemetry_writer: TelemetryV2Writer | None = None,
) -> dict:
    policy.reset()
    observation, _info = env.reset(seed=[seed])
    task_desc = str(env.call("task_description")[0])

    if telemetry_writer is not None:
        initial_semantics, initial_semantics_error = capture_libero_semantics(env)
        task_mapping = None if initial_semantics is None else initial_semantics.get("task_mapping")
        telemetry_writer.begin_episode(
            {
                "episode_index": episode_index,
                "suite": suite_name,
                "task_id": task_id,
                "task_description": task_desc,
                "seed": seed,
                "task_mapping": task_mapping,
                "task_mapping_hash": None if task_mapping is None else canonical_hash(task_mapping),
                "task_mapping_unavailable_reason": initial_semantics_error,
            }
        )

    first_success_step: int | None = None
    num_frames = 0
    previous_gripper_position: list[float] | None = None
    for step_idx in range(extended_max_steps):
        raw_observation = deepcopy(observation)
        simulator_state = simulator_state_error = semantic_state = semantic_state_error = None
        if telemetry_writer is not None:
            simulator_state, simulator_state_error, semantic_state, semantic_state_error = capture_step_state(env)

        obs = preprocess_observation(observation)
        obs["task"] = [task_desc]
        obs = env_preprocessor(obs)
        obs = preprocessor(obs)
        with torch.inference_mode():
            action = policy.select_action(obs)
        action = postprocessor(action)
        action = env_postprocessor({ACTION: action})[ACTION]
        action_np = action.to("cpu").numpy()

        observation, reward, terminated, truncated, info = env.step(action_np)
        next_simulator_state = next_simulator_state_error = next_semantic_state = next_semantic_state_error = None
        if telemetry_writer is not None:
            (
                next_simulator_state,
                next_simulator_state_error,
                next_semantic_state,
                next_semantic_state_error,
            ) = capture_step_state(env)

        success = bool(np.asarray(info["is_success"]).reshape(-1)[0])
        done = bool(terminated[0] or truncated[0])
        is_recovery_phase = step_idx >= standard_max_steps

        frame = build_frame(
            raw_observation, action_np[0], float(np.asarray(reward)[0]), success, done, is_recovery_phase, task_desc
        )
        dataset.add_frame(frame)
        if telemetry_writer is not None:
            telemetry_frame = build_frame_record(
                episode_index=episode_index,
                timestep=step_idx,
                fps=fps,
                raw_observation=raw_observation,
                previous_gripper_position=previous_gripper_position,
                action=action_np[0],
                reward=float(np.asarray(reward)[0]),
                task_success=success,
                done=done,
                simulator_state_vector=simulator_state,
                next_simulator_state_vector=next_simulator_state,
                semantic_state=semantic_state,
                next_semantic_state=next_semantic_state,
            )
            if simulator_state_error or next_simulator_state_error:
                telemetry_frame["simulator_state_unavailable_reason"] = {
                    "pre_action": simulator_state_error,
                    "post_action": next_simulator_state_error,
                }
            if semantic_state_error or next_semantic_state_error:
                telemetry_frame["semantic_state_unavailable_reason"] = {
                    "pre_action": semantic_state_error,
                    "post_action": next_semantic_state_error,
                }
            telemetry_writer.write_frame(telemetry_frame)
            previous_gripper_position = telemetry_frame["gripper_position"]
        num_frames += 1

        if success and first_success_step is None:
            first_success_step = step_idx
        if success or done:
            break

    dataset.save_episode()

    result = {
        "task_description": task_desc,
        "seed": seed,
        "num_frames": num_frames,
        "standard_max_steps": standard_max_steps,
        "extended_max_steps": extended_max_steps,
        "first_success_step": first_success_step,
        "outcome": classify_outcome(first_success_step, standard_max_steps),
    }
    if telemetry_writer is not None:
        telemetry_writer.finish_episode(
            {
                "episode_index": episode_index,
                "num_frames": num_frames,
                "first_success_step": first_success_step,
                "time_budget_outcome_recorded_not_used_as_detector_proxy": result["outcome"],
            }
        )
    return result


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    if args.telemetry_v2 and any((output_dir / name).exists() for name in ("dataset", "rollout_manifest.json", "telemetry_v2")):
        raise FileExistsError(f"--telemetry-v2 requires a fresh output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    telemetry_writer = None
    if args.telemetry_v2:
        schema_path = Path(__file__).resolve().parents[2] / "research" / "false_complete" / "TELEMETRY_SCHEMA_V2.json"
        telemetry_writer = TelemetryV2Writer(output_dir / "telemetry_v2", schema_path)

    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = Path(args.policy_path)
    if args.device:
        policy_cfg.device = args.device

    env_cfg = LiberoEnvConfig(
        task=args.task,
        fps=20,
        obs_type="pixels_agent_pos",
        control_mode=args.control_mode,
        init_states=True,
    )

    logger.info("Building LIBERO envs for suites: %s", args.task)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)

    logger.info("Loading policy from %s", args.policy_path)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=args.rename_map)
    policy.eval()

    preprocessor_overrides = {"device_processor": {"device": str(policy_cfg.device)}}
    if args.rename_map:
        preprocessor_overrides["rename_observations_processor"] = {"rename_map": args.rename_map}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(args.policy_path),
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    first_env = next(iter(next(iter(envs.values())).values()))
    observation_height, observation_width = get_image_hw(first_env)
    features = build_dataset_features(observation_height, observation_width)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=env_cfg.fps,
        features=features,
        root=str(output_dir / "dataset"),
        use_videos=True,
    )

    task_ids_filter = set(args.task_ids) if args.task_ids else None
    manifest: list[dict] = []
    episode_index = 0

    try:
        for suite_name, task_map in envs.items():
            for task_id in sorted(task_map):
                if task_ids_filter is not None and task_id not in task_ids_filter:
                    continue
                env = task_map[task_id]
                standard_max_steps = int(env.call("_max_episode_steps")[0])
                horizon_cap = LIBERO_ROBOSUITE_HORIZON - LIBERO_HORIZON_SAFETY_MARGIN
                extended_max_steps = max(standard_max_steps, int(round(standard_max_steps * args.extension_factor)))
                if extended_max_steps > horizon_cap:
                    logger.warning(
                        "[%s] task_id=%d: extended_max_steps=%d exceeds LIBERO's fixed robosuite "
                        "horizon; capping to %d (see LIBERO_ROBOSUITE_HORIZON comment).",
                        suite_name,
                        task_id,
                        extended_max_steps,
                        horizon_cap,
                    )
                    extended_max_steps = horizon_cap
                logger.info(
                    "[%s] task_id=%d standard_max_steps=%d extended_max_steps=%d",
                    suite_name,
                    task_id,
                    standard_max_steps,
                    extended_max_steps,
                )
                for _ in range(args.episodes_per_task):
                    seed = args.seed + episode_index
                    meta = run_episode(
                        env,
                        policy,
                        env_preprocessor,
                        env_postprocessor,
                        preprocessor,
                        postprocessor,
                        dataset,
                        seed,
                        standard_max_steps,
                        extended_max_steps,
                        episode_index,
                        suite_name,
                        task_id,
                        env_cfg.fps,
                        telemetry_writer,
                    )
                    meta.update({"suite": suite_name, "task_id": task_id, "episode_index": episode_index})
                    manifest.append(meta)
                    logger.info(
                        "episode %d [%s/%d]: outcome=%s frames=%d first_success_step=%s",
                        episode_index,
                        suite_name,
                        task_id,
                        meta["outcome"],
                        meta["num_frames"],
                        meta["first_success_step"],
                    )
                    episode_index += 1
    finally:
        if telemetry_writer is not None:
            telemetry_writer.close_partial()
        dataset.finalize()
        close_envs(envs)

    manifest_path = output_dir / "rollout_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    if telemetry_writer is not None:
        telemetry_manifest_path = telemetry_writer.finalize()
        logger.info("Telemetry v2: %s", telemetry_manifest_path)

    counts = Counter(m["outcome"] for m in manifest)
    logger.info("Done: %d episodes -> %s", len(manifest), dict(counts))
    logger.info("Dataset: %s", output_dir / "dataset")
    logger.info("Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
