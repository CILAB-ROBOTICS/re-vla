#!/usr/bin/env python
"""Fine-tune a VLA policy on LIBERO expert demonstrations.

Thin wrapper around ``lerobot-train`` (``lerobot.scripts.lerobot_train``): it builds the
documented LIBERO training command for the chosen policy type from a small set of CLI
flags and runs it as a subprocess, so all of lerobot's training-loop logic (optimizer
schedules, checkpointing, resuming, Hub push, ...) is reused as-is instead of being
reimplemented here. Recipes are taken from lerobot/docs/source/libero.mdx and
lerobot/docs/source/pi05.mdx.

Supported --policy-type values: smolvla, pi05. Each has a different recommended
fine-tuning recipe (see build_policy_flags below); prefer the per-policy shell wrappers
(finetune_smolvla_libero.sh, finetune_pi05_libero.sh) for day-to-day use, this script is
the shared engine behind both.

Examples
--------
    # SmolVLA, from scratch VLM init
    python finetune_libero.py --policy-type smolvla \\
        --dataset-repo-id lerobot/libero --output-dir ./outputs/libero_smolvla \\
        --steps 30000 --batch-size 8

    # Restrict training to specific LIBERO suites (resolved to --dataset.episodes)
    python finetune_libero.py --policy-type smolvla \\
        --dataset-repo-id lerobot/libero --output-dir ./outputs/libero_smolvla_object \\
        --task libero_object --steps 30000 --batch-size 8

    # Pi0.5, continuing from the LIBERO-pretrained base checkpoint
    python finetune_libero.py --policy-type pi05 \\
        --dataset-repo-id lerobot/libero --output-dir ./outputs/libero_pi05 \\
        --pretrained-path lerobot/pi05_libero_base --freeze-vision-encoder \\
        --steps 30000 --batch-size 2

    # Resume any run
    python finetune_libero.py --output-dir ./outputs/libero_smolvla --resume

Anything after ``--`` is passed through untouched to ``lerobot-train``, e.g.:
    python finetune_libero.py --policy-type smolvla --output-dir ./out -- --wandb.enable=true
"""

from __future__ import annotations

import argparse
import subprocess
import sys

POLICY_TYPES = ["smolvla", "pi05"]


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Fine-tune a VLA policy on a LIBERO dataset via lerobot-train.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy-type", default="smolvla", choices=POLICY_TYPES)
    parser.add_argument("--dataset-repo-id", default="lerobot/libero", help="HF dataset repo id or local path.")
    parser.add_argument("--dataset-revision", default=None, help="Pin a dataset revision (Hub commit SHA).")
    parser.add_argument(
        "--task",
        default=None,
        help="Restrict training to episodes for these LIBERO suite(s) (comma-separated, e.g. "
        "libero_object,libero_spatial — same suite names as collect_libero_rollouts.py). Resolved to "
        "--dataset.episodes by exact-matching each suite's task language against the dataset's task "
        "list, via the `libero` package and a (data-only, no video) read of the dataset. Omit to train "
        "on the whole dataset (default).",
    )
    parser.add_argument(
        "--video-backend",
        default="pyav",
        choices=["pyav", "torchcodec"],
        help="Video decoding backend for the dataset.",
    )
    parser.add_argument("--output-dir", required=True, help="Where checkpoints/logs are written.")
    parser.add_argument("--steps", type=int, default=30_000, help="Number of training steps.")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-step batch size.")
    parser.add_argument("--save-freq", type=int, default=5_000, help="Checkpoint frequency (steps).")
    parser.add_argument("--log-freq", type=int, default=200, help="Logging frequency (steps).")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default=None, help="Override policy device (e.g. cuda, cpu).")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision (SmolVLA only; enabled by default).")

    # SmolVLA-specific
    parser.add_argument(
        "--load-vlm-weights",
        action="store_true",
        default=True,
        help="[smolvla] Initialize the VLM backbone from pretrained weights (recommended).",
    )
    parser.add_argument(
        "--from-checkpoint",
        default=None,
        help="[smolvla] Continue fine-tuning from a full SmolVLA checkpoint (sets --policy.path; "
        "loads architecture config + weights). Mutually exclusive with --load-vlm-weights-from-scratch behavior.",
    )

    # Pi0.5-specific
    parser.add_argument(
        "--pretrained-path",
        default=None,
        help="[pi05] Base checkpoint to load weights from (sets --policy.pretrained_path; weights only, "
        "architecture flags below are re-specified explicitly). Defaults to lerobot/pi05_libero_base.",
    )
    parser.add_argument(
        "--freeze-vision-encoder",
        action="store_true",
        help="[pi05] Freeze the VLM and train only the action expert (train_expert_only=true). "
        "Much lower memory, some cost in success rate. Recommended on <24GB GPUs.",
    )
    parser.add_argument("--n-action-steps", type=int, default=10, help="[pi05] Action chunk steps to execute.")
    parser.add_argument("--empty-cameras", type=int, default=1, help="[pi05] See lerobot/docs/source/pi05.mdx.")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float32"], help="[pi05] Training compute dtype."
    )

    parser.add_argument("--push-to-hub", action="store_true", help="Push the trained policy to the HF Hub.")
    parser.add_argument("--policy-repo-id", default=None, help="Hub repo id to push to (requires --push-to-hub).")
    parser.add_argument("--resume", action="store_true", help="Resume training from --output-dir.")
    parser.add_argument("--job-name", default=None, help="Run name shown in logs/W&B (default: auto-generated).")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument(
        "--wandb-project", default="libero-vla", help="W&B project name (only used if --wandb is set)."
    )
    parser.add_argument(
        "--wandb-entity", default=None, help="W&B entity (team/user); defaults to your W&B account."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the resolved lerobot-train command without running it."
    )
    return parser.parse_known_args(argv)


def resolve_task_episodes(dataset_repo_id: str, dataset_revision: str | None, task: str) -> list[int]:
    """Resolve comma-separated LIBERO suite names to dataset episode indices.

    Matches each suite's task language (from the `libero` benchmark package) against the
    dataset's task list by exact string, then looks up which episode performs each
    matched task_index. Requires a (data-only, no video) read of the dataset since
    episode-to-task_index isn't stored in the lightweight metadata alone.
    """
    from libero.libero import benchmark

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    suites = [s.strip() for s in task.split(",") if s.strip()]
    if not suites:
        raise ValueError("--task must contain at least one LIBERO suite name.")

    bench = benchmark.get_benchmark_dict()
    meta = LeRobotDatasetMetadata(repo_id=dataset_repo_id, revision=dataset_revision)

    target_task_indices: set[int] = set()
    for suite_name in suites:
        if suite_name not in bench:
            raise ValueError(f"Unknown LIBERO suite '{suite_name}'. Available: {sorted(bench.keys())}")
        suite = bench[suite_name]()
        for libero_task in suite.tasks:
            task_index = meta.get_task_index(libero_task.language)
            if task_index is None:
                print(
                    f"WARNING: task '{libero_task.language}' (suite={suite_name}) not found in dataset "
                    f"'{dataset_repo_id}'; skipping."
                )
                continue
            target_task_indices.add(task_index)

    if not target_task_indices:
        raise ValueError(f"None of the requested suites {suites} match any task in dataset '{dataset_repo_id}'.")

    dataset = LeRobotDataset(repo_id=dataset_repo_id, revision=dataset_revision)
    episode_indices = []
    for episode in dataset.meta.episodes:
        first_frame_index = episode["dataset_from_index"]
        episode_task_index = int(dataset.hf_dataset[first_frame_index]["task_index"])
        if episode_task_index in target_task_indices:
            episode_indices.append(int(episode["episode_index"]))

    if not episode_indices:
        raise ValueError(f"No episodes in dataset '{dataset_repo_id}' matched suites {suites}.")

    return sorted(episode_indices)


def build_policy_flags(args: argparse.Namespace) -> list[str]:
    """Per-policy-type flags, following the recipes in libero.mdx / pi05.mdx."""
    if args.policy_type == "smolvla":
        if args.from_checkpoint:
            return [f"--policy.path={args.from_checkpoint}"]
        return ["--policy.type=smolvla", f"--policy.load_vlm_weights={str(args.load_vlm_weights).lower()}"]

    if args.policy_type == "pi05":
        pretrained_path = args.pretrained_path or "lerobot/pi05_libero_base"
        freeze = args.freeze_vision_encoder
        return [
            "--policy.type=pi05",
            f"--policy.pretrained_path={pretrained_path}",
            '--policy.normalization_mapping={"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}',
            f"--policy.n_action_steps={args.n_action_steps}",
            f"--policy.empty_cameras={args.empty_cameras}",
            f"--policy.freeze_vision_encoder={str(freeze).lower()}",
            f"--policy.train_expert_only={str(freeze).lower()}",
            "--policy.gradient_checkpointing=true",
            f"--policy.dtype={args.dtype}",
        ]

    raise ValueError(f"Unknown policy type: {args.policy_type}")


def build_command(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    if args.push_to_hub and not args.policy_repo_id:
        raise ValueError("--push-to-hub requires --policy-repo-id.")
    if args.resume and (args.from_checkpoint or args.pretrained_path):
        raise ValueError("--resume is mutually exclusive with --from-checkpoint/--pretrained-path.")
    if args.resume and args.task:
        raise ValueError("--resume is mutually exclusive with --task (episode selection is fixed by the prior run).")

    cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_train"]

    if args.resume:
        cmd += [f"--output_dir={args.output_dir}", "--resume=true"]
        cmd += passthrough
        return cmd

    cmd += build_policy_flags(args)
    cmd += [
        f"--policy.push_to_hub={str(args.push_to_hub).lower()}",
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--dataset.video_backend={args.video_backend}",
        f"--output_dir={args.output_dir}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        f"--seed={args.seed}",
        f"--wandb.enable={str(args.wandb).lower()}",
    ]
    if args.policy_type == "smolvla":
        cmd.append(f"--policy.use_amp={str(not args.no_amp).lower()}")
    if args.job_name:
        cmd.append(f"--job_name={args.job_name}")
    if args.wandb:
        cmd.append(f"--wandb.project={args.wandb_project}")
        if args.wandb_entity:
            cmd.append(f"--wandb.entity={args.wandb_entity}")
    if args.dataset_revision:
        cmd.append(f"--dataset.revision={args.dataset_revision}")
    if args.task:
        print(f"Resolving --task={args.task!r} to dataset episodes...")
        episode_indices = resolve_task_episodes(args.dataset_repo_id, args.dataset_revision, args.task)
        print(f"  -> {len(episode_indices)} episodes matched.")
        cmd.append(f"--dataset.episodes=[{', '.join(str(i) for i in episode_indices)}]")
    if args.device:
        cmd.append(f"--policy.device={args.device}")
    if args.policy_repo_id:
        cmd.append(f"--policy.repo_id={args.policy_repo_id}")

    cmd += passthrough
    return cmd


def main(argv: list[str] | None = None) -> int:
    args, passthrough = parse_args(argv)
    cmd = build_command(args, passthrough)

    print("Running:\n  " + " \\\n  ".join(cmd))
    if args.dry_run:
        return 0

    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
