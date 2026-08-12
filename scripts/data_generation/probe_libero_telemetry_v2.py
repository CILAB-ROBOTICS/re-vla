#!/usr/bin/env python
"""CPU/GPU-neutral runtime probe for LIBERO telemetry-v2 access boundaries."""

from __future__ import annotations

import argparse
import json

from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs

from telemetry_v2 import canonical_hash, capture_libero_semantics, capture_simulator_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    cfg = LiberoEnvConfig(
        task=args.task,
        fps=20,
        obs_type="pixels_agent_pos",
        control_mode="relative",
        init_states=True,
    )
    envs = make_env(cfg, n_envs=1, use_async_envs=False)
    try:
        env = envs[args.task][args.task_id]
        env.reset(seed=[args.seed])
        simulator_state, simulator_error = capture_simulator_state(env)
        semantic_state, semantic_error = capture_libero_semantics(env)
        mapping = None if semantic_state is None else semantic_state["task_mapping"]
        print(
            json.dumps(
                {
                    "task": args.task,
                    "task_id": args.task_id,
                    "seed": args.seed,
                    "simulator_state_length": None if simulator_state is None else len(simulator_state),
                    "simulator_error": simulator_error,
                    "task_mapping_hash": None if mapping is None else canonical_hash(mapping),
                    "objects_of_interest": None if mapping is None else mapping["objects_of_interest"],
                    "goal_state": None if mapping is None else mapping["goal_state"],
                    "object_count": None if semantic_state is None else len(semantic_state["objects"]),
                    "site_count": None if semantic_state is None else len(semantic_state["sites"]),
                    "contact_count": None if semantic_state is None else len(semantic_state["contacts"]),
                    "goal_predicate_values": None
                    if semantic_state is None
                    else [item["value"] for item in semantic_state["goal_predicates"]],
                    "semantic_error": semantic_error,
                    "scientific_labels_assigned": False,
                },
                indent=2,
            )
        )
    finally:
        close_envs(envs)


if __name__ == "__main__":
    main()
