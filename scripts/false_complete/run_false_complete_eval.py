#!/usr/bin/env python
"""False Complete evaluation: for each task, run paired (control, perturbed) SmolVLA
rollouts from the *same* initial simulator state, inject a perturbation into the
perturbed run at a configurable trigger point, and log everything needed to check
whether the policy's action output actually changes once the environment has diverged.

Research question (see module design note in metrics.py): a perturbed episode failing
is not itself evidence of False Complete — what matters is whether the *policy's action*
changed once the *environment* actually diverged from the successful/control case. So
every pair shares task, seed, and initial simulator state (see `_pin_init_state` below),
and the raw action *chunk* SmolVLA predicts is logged in full (not just the single
action executed per step — see `ChunkedActionRunner`), since that's the actual thing
being compared between control and perturbed.

Reused, not reimplemented (see scripts/data_generation/collect_libero_rollouts.py for
the same pattern applied to plain rollout collection):
  - env/policy construction: lerobot.envs.factory.make_env, lerobot.policies.factory.make_policy
  - obs/action pre/post-processing pipelines: make_pre_post_processors, make_env_pre_post_processors
  - grasp detection: robosuite's `env._check_grasp` (manipulation_env.py)
  - object pose / initial-state restore: LIBERO's own `ControlEnv` accessors
    (`get_sim_state`/`set_init_state`, env_wrapper.py) — see `_pin_init_state`.

Usage
-----
    python run_false_complete_eval.py \\
        --task libero_object --task-id 0 \\
        --checkpoint outputs/libero_smolvla/checkpoints/last/pretrained_model \\
        --perturbation object_drop --trigger timestep --trigger-step 40 \\
        --num-episodes 5 --output-dir outputs/false_complete

    python run_false_complete_eval.py \\
        --task libero_object --task-id 0 \\
        --checkpoint outputs/libero_smolvla/checkpoints/last/pretrained_model \\
        --perturbation object_relocation --translation 0.08 0.0 0.0 --trigger grasp \\
        --num-episodes 5 --output-dir outputs/false_complete
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import metrics  # noqa: E402
from perturbations import PERTURBATIONS, Perturbation  # noqa: E402
from rollout_logger import EpisodeLogger  # noqa: E402
from triggers import TRIGGERS, Trigger  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.random_utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_false_complete_eval")

STATE_DIM = 9  # [eef_pos(3), eef_quat(4), gripper_qpos(2)] — same raw layout as collect_libero_rollouts.py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired control/perturbed False Complete evaluation for SmolVLA on LIBERO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", default="libero_object", help="LIBERO suite name.")
    parser.add_argument("--task-id", type=int, default=0, help="Task index within the suite.")
    parser.add_argument("--checkpoint", required=True, help="Fine-tuned SmolVLA checkpoint (local dir or Hub id).")
    parser.add_argument("--control-mode", default="relative", choices=["relative", "absolute"])
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of control/perturbed *pairs*.")
    parser.add_argument("--seed", type=int, default=1000, help="Seed for pair 0; incremented per pair.")
    parser.add_argument("--device", default=None, help="Override policy device (e.g. cuda, cpu).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--agentview-key",
        default="image",
        help="observation.images.<key> the checkpoint expects for the main/agentview camera. "
        "Must match how the training dataset named it — check the "
        "'Feature mismatch' error's 'Missing features' list if unsure.",
    )
    parser.add_argument(
        "--wrist-key",
        default="image2",
        help="observation.images.<key> the checkpoint expects for the wrist camera "
        "(e.g. 'wrist_image' for datasets using that convention instead of lerobot's default 'image2').",
    )

    parser.add_argument("--perturbation", required=True, choices=sorted(PERTURBATIONS.keys()))
    parser.add_argument("--target-object", default=None, help="Defaults to the task's first obj_of_interest.")
    parser.add_argument(
        "--drop-offset",
        type=float,
        default=0.03,
        help="[object_drop] Downward nudge (meters) to break gripper contact; gravity does the rest.",
    )
    parser.add_argument(
        "--translation", type=float, nargs=3, default=(0.08, 0.0, 0.0), help="[object_relocation] dx dy dz."
    )

    parser.add_argument("--trigger", required=True, choices=sorted(TRIGGERS.keys()))
    parser.add_argument("--trigger-step", type=int, default=40, help="[timestep trigger]")
    parser.add_argument("--lift-threshold", type=float, default=0.04, help="[lift trigger] meters above initial z.")

    parser.add_argument(
        "--similarity-window", type=int, default=20, help="Steps after trigger to average action similarity over."
    )
    parser.add_argument("--continue-threshold", type=float, default=0.85)
    parser.add_argument("--recovery-threshold", type=float, default=0.6)

    return parser.parse_args()


def build_perturbation(args: argparse.Namespace) -> Perturbation:
    cls = PERTURBATIONS[args.perturbation]
    if args.perturbation == "object_drop":
        return cls(drop_offset=args.drop_offset, target_object=args.target_object)
    if args.perturbation == "object_relocation":
        return cls(translation=tuple(args.translation), target_object=args.target_object)
    return cls(target_object=args.target_object)


def build_trigger(args: argparse.Namespace) -> Trigger:
    cls = TRIGGERS[args.trigger]
    if args.trigger == "timestep":
        return cls(args.trigger_step)
    if args.trigger == "lift":
        return cls(target_object=args.target_object, height_threshold=args.lift_threshold)
    return cls(target_object=args.target_object)


def get_libero_env(vec_env):
    """The raw LIBERO ControlEnv (env_wrapper.py), for low-level sim access — perturbations,
    triggers, and object-pose logging all need this layer, not the lerobot gym wrapper."""
    return vec_env.envs[0]._env


def pin_init_state(vec_env, gym_env, seed: int, init_state_id: int):
    """Forces a specific `_init_states` entry so a (control, perturbed) pair starts from
    the *same* simulator state — reuses LiberoEnv's own reset machinery (num_steps_wait
    warmup, control_mode setup, obs formatting/batching) rather than reimplementing it;
    just pins which pre-sampled initial state that reset draws from.
    """
    gym_env.init_state_id = init_state_id
    return vec_env.reset(seed=[seed])


def _batch_wrap(obs):
    """Recursively adds a leading batch dim to a single-env observation dict, matching
    the shape gym.vector.SyncVectorEnv normally produces — needed after manually
    refreshing observations post-perturbation (see run_episode)."""
    if isinstance(obs, dict):
        return {k: _batch_wrap(v) for k, v in obs.items()}
    if isinstance(obs, np.ndarray):
        return np.expand_dims(obs, 0)
    return obs


def get_object_pose_pair(libero_env, obj_name: str) -> np.ndarray:
    from perturbations import get_object_pose

    return get_object_pose(libero_env, obj_name)


def get_ee_pose(raw_obs: dict) -> np.ndarray:
    rs = raw_obs["robot_state"]
    return np.concatenate(
        [np.asarray(rs["eef"]["pos"][0], dtype=np.float32), np.asarray(rs["eef"]["quat"][0], dtype=np.float32)]
    )


def get_raw_state(raw_obs: dict) -> np.ndarray:
    rs = raw_obs["robot_state"]
    return np.concatenate(
        [
            np.asarray(rs["eef"]["pos"][0], dtype=np.float32),
            np.asarray(rs["eef"]["quat"][0], dtype=np.float32),
            np.asarray(rs["gripper"]["qpos"][0], dtype=np.float32),
        ]
    )


def get_rgb_frames(raw_obs: dict, agentview_key: str, wrist_key: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Applies the same 180-degree flip LiberoProcessorStep applies before feeding the
    policy (LIBERO's raw MuJoCo camera images are upside-down/mirrored — see
    collect_libero_rollouts.py's build_frame for the same fix), so saved video matches
    what the policy actually sees / looks right to a human reviewer."""
    rgb = np.ascontiguousarray(raw_obs["pixels"][agentview_key][0][::-1, ::-1])
    wrist = None
    if wrist_key in raw_obs["pixels"]:
        wrist = np.ascontiguousarray(raw_obs["pixels"][wrist_key][0][::-1, ::-1])
    return rgb, wrist


class ChunkedActionRunner:
    """Drives the policy via `predict_action_chunk()` directly rather than
    `select_action()`, so the full raw chunk is available every time one is generated
    (select_action only exposes it internally, via a private deque — see
    SmolVLAPolicy.select_action in modeling_smolvla.py). Queue-management logic below
    mirrors select_action's own (transpose + deque) exactly, just made visible.
    """

    def __init__(self, policy, n_action_steps: int, torch_seed_base: int):
        self.policy = policy
        self.n_action_steps = n_action_steps
        self.torch_seed_base = torch_seed_base
        self._queue: list[torch.Tensor] = []
        self._chunk_counter = 0

    def reset(self) -> None:
        self.policy.reset()
        self._queue = []
        self._chunk_counter = 0

    def step(self, obs_batch: dict) -> tuple[torch.Tensor, np.ndarray | None]:
        """Returns (action (batch=1, action_dim), new_chunk or None if the queue wasn't empty)."""
        new_chunk = None
        if not self._queue:
            # Same seed at the same chunk-generation index on both control and perturbed
            # runs of a pair -> identical sampled noise -> identical action chunks until
            # the actual observations diverge (post-perturbation). Pure inference, no
            # other torch randomness happens between calls, so this holds cleanly.
            torch.manual_seed(self.torch_seed_base + self._chunk_counter)
            with torch.inference_mode():
                actions = self.policy.predict_action_chunk(obs_batch)  # (batch=1, n_action_steps, action_dim)
            new_chunk = actions[0, : self.n_action_steps].detach().cpu().numpy()
            self._queue = list(actions.transpose(0, 1)[: self.n_action_steps])  # matches select_action's own logic
            self._chunk_counter += 1
        action = self._queue.pop(0)
        return action, new_chunk


def run_episode(
    *,
    vec_env,
    gym_env,
    libero_env,
    action_runner: ChunkedActionRunner,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    seed: int,
    init_state_id: int,
    max_steps: int,
    target_object: str,
    task_desc: str,
    is_perturbed_run: bool,
    perturbation: Perturbation,
    trigger: Trigger,
    episode_logger: EpisodeLogger,
    agentview_key: str,
    wrist_key: str,
) -> dict[str, Any]:
    action_runner.reset()
    trigger.reset()
    observation, _info = pin_init_state(vec_env, gym_env, seed, init_state_id)

    trigger_step: int | None = None
    perturbation_details: dict | None = None
    success = False

    for step_idx in range(max_steps):
        raw_observation = deepcopy(observation)

        if is_perturbed_run and trigger_step is None and trigger.check(step_idx, libero_env, {"target_object": target_object}):
            trigger_step = step_idx
            perturbation_details = perturbation.apply(libero_env, {"target_object": target_object})
            print("=" * 40)
            print("PERTURBATION TRIGGERED")
            print(f"type: {perturbation.name}")
            print(f"timestep: {step_idx}")
            print("=" * 40)
            # The perturbation just wrote directly into sim.data.qpos (see perturbations.py) -
            # refresh cached observables/observations using LIBERO's own post-mutation
            # sequence (env_wrapper.py's regenerate_obs_from_state, minus the set_state
            # step since we didn't touch the full sim state, just one object's qpos).
            libero_env.check_success()
            libero_env._post_process()
            libero_env._update_observables(force=True)
            raw_libero_obs = libero_env.env._get_observations()
            raw_observation = _batch_wrap(gym_env._format_raw_obs(raw_libero_obs))

        raw_observation = perturbation.modify_observation(raw_observation, {"target_object": target_object})

        obs = preprocess_observation(raw_observation)
        obs["task"] = [task_desc]
        obs = env_preprocessor(obs)
        obs = preprocessor(obs)

        action, new_chunk = action_runner.step(obs)
        if new_chunk is not None:
            episode_logger.log_action_chunk(step_idx, new_chunk)

        action = postprocessor(action)
        action = env_postprocessor({ACTION: action})[ACTION]
        action_np = action.to("cpu").numpy()

        observation, reward, terminated, truncated, info = vec_env.step(action_np)
        success = bool(np.asarray(info["is_success"]).reshape(-1)[0])
        done = bool(terminated[0] or truncated[0])

        rgb, wrist_rgb = get_rgb_frames(raw_observation, agentview_key, wrist_key)
        episode_logger.log_step(
            rgb=rgb,
            wrist_rgb=wrist_rgb,
            state=get_raw_state(raw_observation),
            executed_action=action_np[0],
            object_pose=get_object_pose_pair(libero_env, target_object),
            ee_pose=get_ee_pose(raw_observation),
            reward=float(np.asarray(reward)[0]),
            done=done,
            success=success,
            is_perturbed=is_perturbed_run and trigger_step is not None,
        )

        if success or done:
            break

    return {
        "success": success,
        "trigger_step": trigger_step,
        "perturbation_details": perturbation_details,
        "num_steps": step_idx + 1,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    policy_cfg.pretrained_path = Path(args.checkpoint)
    if args.device:
        policy_cfg.device = args.device

    env_cfg = LiberoEnvConfig(
        task=args.task,
        fps=20,
        obs_type="pixels_agent_pos",
        control_mode=args.control_mode,
        init_states=True,
        camera_name_mapping={"agentview_image": args.agentview_key, "robot0_eye_in_hand_image": args.wrist_key},
    )

    logger.info("Building LIBERO env: suite=%s task_id=%d", args.task, args.task_id)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    vec_env = envs[args.task][args.task_id]
    gym_env = vec_env.envs[0]
    libero_env = get_libero_env(vec_env)

    logger.info("Loading policy from %s", args.checkpoint)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(policy_cfg.device)}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    standard_max_steps = int(vec_env.call("_max_episode_steps")[0])
    n_action_steps = getattr(policy.config, "n_action_steps", 1)
    task_desc = str(vec_env.call("task_description")[0])
    if args.target_object is None:
        if not libero_env.obj_of_interest:
            raise ValueError(
                f"Task {args.task}/{args.task_id} has no obj_of_interest in its BDDL definition; "
                "pass --target-object explicitly (see libero_env.env.objects_dict for available names)."
            )
        target_object = libero_env.obj_of_interest[0]
    else:
        target_object = args.target_object

    print("Task:", args.task)
    print("Episode: (pairs) x", args.num_episodes)
    print("Seed:", args.seed)
    print("Instruction:", task_desc)
    print("Checkpoint:", args.checkpoint)
    print("Camera keys:", [args.agentview_key, args.wrist_key])
    print("State shape:", (STATE_DIM,))
    print("Action shape:", vec_env.single_action_space.shape)
    print("Action chunk size:", n_action_steps)
    print("Perturbation:", args.perturbation)
    print("Trigger:", args.trigger)
    print("Trigger timestep (if timestep trigger):", args.trigger_step)
    print("Target object:", target_object)
    print()
    print("Running False Complete Evaluation\n")
    print(f"Task: {args.task} / task_id={args.task_id}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Perturbation: {args.perturbation}\n")

    pair_summaries: list[dict[str, Any]] = []

    try:
        for pair_idx in range(args.num_episodes):
            seed = args.seed + pair_idx
            init_state_id = pair_idx
            action_rng_base = seed * 1000

            pair_dir = output_dir / f"pair_{pair_idx:04d}"
            control_logger = EpisodeLogger(pair_dir / "control")
            perturbed_logger = EpisodeLogger(pair_dir / "perturbed")

            perturbation = build_perturbation(args)
            control_trigger = build_trigger(args)
            perturbed_trigger = build_trigger(args)
            action_runner = ChunkedActionRunner(policy, n_action_steps, action_rng_base)

            control_result = run_episode(
                vec_env=vec_env, gym_env=gym_env, libero_env=libero_env, action_runner=action_runner,
                env_preprocessor=env_preprocessor, env_postprocessor=env_postprocessor,
                preprocessor=preprocessor, postprocessor=postprocessor,
                seed=seed, init_state_id=init_state_id, max_steps=standard_max_steps,
                target_object=target_object, task_desc=task_desc, is_perturbed_run=False,
                perturbation=perturbation, trigger=control_trigger, episode_logger=control_logger,
                agentview_key=args.agentview_key, wrist_key=args.wrist_key,
            )
            control_logger.save(
                metadata={
                    "episode_id": f"pair_{pair_idx:04d}/control", "task_name": args.task, "task_id": args.task_id,
                    "seed": seed, "instruction": task_desc, "checkpoint": args.checkpoint,
                    "action_chunk_size": n_action_steps, "target_object": target_object,
                },
                events={
                    "perturbation_type": None, "perturbation_timestep": None, "is_perturbed": False,
                    "grasp_or_lift_trigger_timestep": control_result["trigger_step"], "success": control_result["success"],
                },
            )

            perturbed_result = run_episode(
                vec_env=vec_env, gym_env=gym_env, libero_env=libero_env, action_runner=action_runner,
                env_preprocessor=env_preprocessor, env_postprocessor=env_postprocessor,
                preprocessor=preprocessor, postprocessor=postprocessor,
                seed=seed, init_state_id=init_state_id, max_steps=standard_max_steps,
                target_object=target_object, task_desc=task_desc, is_perturbed_run=True,
                perturbation=perturbation, trigger=perturbed_trigger, episode_logger=perturbed_logger,
                agentview_key=args.agentview_key, wrist_key=args.wrist_key,
            )

            trigger_step = perturbed_result["trigger_step"]
            behavior_label = "unclassified"
            post_perturb_sim: dict[str, Any] = {"l2": None, "cosine": None, "n_steps": 0}
            if trigger_step is not None:
                post_perturb_sim = metrics.post_perturbation_window_similarity(
                    control_logger.executed_actions, perturbed_logger.executed_actions,
                    trigger_step, window=args.similarity_window,
                )
                behavior_label = metrics.classify_behavior(
                    post_perturb_sim["cosine"], perturbed_result["success"],
                    continue_threshold=args.continue_threshold, recovery_threshold=args.recovery_threshold,
                )

            perturbed_logger.save(
                metadata={
                    "episode_id": f"pair_{pair_idx:04d}/perturbed", "task_name": args.task, "task_id": args.task_id,
                    "seed": seed, "instruction": task_desc, "checkpoint": args.checkpoint,
                    "action_chunk_size": n_action_steps, "target_object": target_object,
                },
                events={
                    "perturbation_type": args.perturbation, "perturbation_timestep": trigger_step,
                    "perturbation_details": perturbed_result["perturbation_details"], "is_perturbed": trigger_step is not None,
                    "grasp_or_lift_trigger_timestep": trigger_step, "success": perturbed_result["success"],
                    "behavior_label": behavior_label, "post_perturbation_window_similarity": post_perturb_sim,
                },
            )

            pair_summaries.append(
                {
                    "pair_idx": pair_idx, "seed": seed, "control_success": control_result["success"],
                    "perturbed_success": perturbed_result["success"], "trigger_step": trigger_step,
                    "post_perturb_cosine_similarity": post_perturb_sim["cosine"],
                    "post_perturb_l2_distance": post_perturb_sim["l2"], "behavior_label": behavior_label,
                }
            )

            print(f"Episode {pair_idx + 1}")
            print(f"Control success: {control_result['success']}")
            print(f"Perturbation triggered at t={trigger_step}")
            print(f"Perturbed success: {perturbed_result['success']}")
            sim_str = f"{post_perturb_sim['cosine']:.2f}" if post_perturb_sim["cosine"] is not None else "n/a"
            print(f"Post-perturb action similarity: {sim_str}")
            print(f"Behavior: {behavior_label}\n")
    finally:
        close_envs(envs)

    total = len(pair_summaries)
    control_success_rate = float(np.mean([p["control_success"] for p in pair_summaries])) if total else 0.0
    perturbed_success_rate = float(np.mean([p["perturbed_success"] for p in pair_summaries])) if total else 0.0
    label_counts: dict[str, int] = {}
    label_sims: dict[str, list[float]] = {}
    for p in pair_summaries:
        label_counts[p["behavior_label"]] = label_counts.get(p["behavior_label"], 0) + 1
        if p["post_perturb_cosine_similarity"] is not None:
            label_sims.setdefault(p["behavior_label"], []).append(p["post_perturb_cosine_similarity"])

    summary = {
        "task": args.task, "task_id": args.task_id, "perturbation": args.perturbation, "trigger": args.trigger,
        "total_episodes": total, "control_success_rate": control_success_rate,
        "perturbed_success_rate": perturbed_success_rate, "behavior_label_counts": label_counts,
        "mean_post_perturb_similarity_by_label": {k: float(np.mean(v)) for k, v in label_sims.items()},
        "pairs": pair_summaries,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_summaries[0].keys()) if pair_summaries else [])
        writer.writeheader()
        writer.writerows(pair_summaries)

    print(f"Total episodes: {total}\n")
    print(f"Control success rate: {control_success_rate * 100:.0f}%")
    print(f"Perturbed success rate: {perturbed_success_rate * 100:.0f}%\n")
    for label, count in label_counts.items():
        print(f"{label}: {count}")
    print("\nMean post-perturb action similarity:")
    for label, sims in label_sims.items():
        print(f"{label}: {np.mean(sims):.2f}")
    print(f"\nSummary written to {output_dir / 'summary.json'} / {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
