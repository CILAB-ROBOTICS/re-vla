# LIBERO VLA scripts

1. `fine_tuning/finetune_libero.py` — fine-tune a VLA policy (SmolVLA or Pi0.5) on LIBERO
   expert demonstrations. Shared engine behind two per-policy shell wrappers:
   - `fine_tuning/finetune_smolvla_libero.sh`
   - `fine_tuning/finetune_pi05_libero.sh`
2. `data_generation/collect_libero_rollouts.py` — roll out a fine-tuned policy and collect
   success / failure / recoverable-failure episodes (including the recovery behavior).
   Policy-agnostic engine behind two per-policy shell wrappers:
   - `data_generation/collect_libero_rollouts_smolvla.sh`
   - `data_generation/collect_libero_rollouts_pi05.sh`

All target the `libero_smolvla` conda environment (Python 3.10, `lerobot` editable-installed
from `/home/eunju/research/lerobot`). They import lerobot's env/policy factories from their
submodules (`lerobot.envs.factory`, `lerobot.policies.factory`, ...) rather than the
top-level `lerobot.envs`/`lerobot.policies` packages, since that env's lerobot checkout
doesn't re-export these names at the package level. This keeps the scripts working without
needing to touch the environment.

```bash
conda activate libero_smolvla
export MUJOCO_GL=egl   # headless rendering
```

## 1. Fine-tuning

```bash
# SmolVLA (recipe: docs/source/libero.mdx)
./scripts/fine_tuning/finetune_smolvla_libero.sh

# Pi0.5, continuing from the LIBERO-pretrained base checkpoint (recipe: docs/source/pi05.mdx)
./scripts/fine_tuning/finetune_pi05_libero.sh
```

or call the shared engine directly with any supported `--policy-type`:

```bash
python scripts/fine_tuning/finetune_libero.py --policy-type smolvla \
    --dataset-repo-id lerobot/libero --output-dir ./outputs/libero_smolvla \
    --steps 30000 --batch-size 8
```

Both are thin wrappers around `lerobot-train` (invoked as `python -m lerobot.scripts.lerobot_train`)
so the actual training loop, checkpointing, and resuming logic is lerobot's own,
well-tested implementation — each `--policy-type` just assembles the flags from its
documented recipe (SmolVLA: VLM-init from scratch or `--policy.path=<ckpt>` to continue;
Pi0.5: `--policy.pretrained_path=lerobot/pi05_libero_base` plus the explicit architecture
flags pi05.mdx calls out, since `pretrained_path` loads weights only). Use `--dry-run` to
print the resolved command without running it, and `--resume` to continue an interrupted
run from `--output-dir`.

### Restricting to specific tasks

By default training uses the whole dataset (all suites/tasks). To fine-tune on only
some LIBERO suites, add `--task` (same suite-name convention as the rollout script):

```bash
./scripts/fine_tuning/finetune_smolvla_libero.sh \
    --output-dir ./outputs/libero_smolvla_object --task libero_object
```

This resolves each suite's task language (via the `libero` package) against the
dataset's task list by exact string match, then translates that to `--dataset.episodes`.
It requires a (data-only, no video) read of the dataset to map episodes to tasks, so
there's a short delay before training starts — the resolved episode list is also printed,
even with `--dry-run`, so you can sanity-check it before running.

### Offline validation

Every `--val-freq` steps (default: same as `--save-freq`), a held-out-episode validation
pass runs and logs `val_loss` (plus to W&B if enabled). `--val-ratio` (default `0.02`)
and `--max-val-episodes` (default `20`) bound how many episodes are held out, so a
validation pass — a full forward-pass sweep, at roughly training-step cost per batch —
stays a small fraction of wall-clock time instead of dominating it:

```bash
./scripts/fine_tuning/finetune_smolvla_libero.sh \
    --output-dir ./outputs/libero_smolvla \
    --val-freq 2000 --val-ratio 0.02 --max-val-episodes 20

# disable validation entirely
./scripts/fine_tuning/finetune_smolvla_libero.sh --output-dir ./outputs/libero_smolvla --val-freq 0
```

These map directly to `--val_freq`/`--val_ratio`/`--max_val_episodes` on the underlying
`lerobot-train` config (`TrainPipelineConfig`) — no `lerobot_train.py` edits needed to
retune them.

### Weights & Biases logging

Off by default. Enable with `--wandb`; `--wandb-project` (default `libero-vla`) and
`--wandb-entity` are optional:

```bash
./scripts/fine_tuning/finetune_smolvla_libero.sh \
    --output-dir ./outputs/libero_smolvla \
    --wandb --wandb-project libero-vla --wandb-entity your-team \
    --job-name libero-smolvla-run1
```

Assumes you're already logged in (`wandb login`, or an existing `~/.netrc` entry) —
this just sets `--wandb.enable=true`, `--wandb.project=`, `--wandb.entity=`, and
`--job_name=` on the underlying `lerobot-train` call.

**GPU note**: `--batch-size 8` for SmolVLA is a conservative default for an ~8GB GPU (the
published recipe uses 64 on 8x H100). Pi0.5 is a much larger (~3B param) VLA — its
reference recipe is "sized for a single 80GB GPU"; the shell wrapper defaults to
`--freeze-vision-encoder` (trains only the action expert) to cut memory, but it may still
not fit on an 8GB card. Use a bigger/cloud GPU for Pi0.5 if you hit OOM.

The resulting checkpoint used by the rollout script is
`<output-dir>/checkpoints/last/pretrained_model`.

## 2. Rollout collection

```bash
# SmolVLA
./scripts/data_generation/collect_libero_rollouts_smolvla.sh \
    --task libero_object --episodes-per-task 10

# Pi0.5
./scripts/data_generation/collect_libero_rollouts_pi05.sh \
    --task libero_object --episodes-per-task 10
```

or call the shared engine directly:

```bash
python scripts/data_generation/collect_libero_rollouts.py \
    --policy-path outputs/libero_smolvla/checkpoints/last/pretrained_model \
    --task libero_object \
    --episodes-per-task 10 \
    --output-dir ./outputs/libero_rollouts \
    --repo-id local/libero_smolvla_rollouts
```

`collect_libero_rollouts.py` is policy-agnostic: `--policy-path` is loaded via
`PreTrainedConfig.from_pretrained`, which reads the policy type from the checkpoint's own
`config.json`. The two `.sh` wrappers only differ in their default `--policy-path` (pointing
at each `finetune_*_libero.sh`'s `--output-dir`), `--output-dir`, and `--repo-id` — override
any of them via env vars (`POLICY_PATH=...`) or passthrough flags.

### Outcome definition (extended-rollout rule)

Every LIBERO suite has a standard step budget (e.g. 280 steps for `libero_spatial`).
For each episode, the policy keeps running past that budget, up to
`extended_max_steps = round(standard_max_steps * --extension-factor)` (default factor `2.0`):

| Outcome              | Meaning                                                                 |
| --------------------- | ------------------------------------------------------------------------ |
| `success`             | Solved at or before `standard_max_steps`.                               |
| `recoverable_failure` | Not solved by `standard_max_steps`, but solved before `extended_max_steps`. |
| `failure`             | Still not solved by `extended_max_steps`.                               |

Frames captured after `standard_max_steps` are tagged `is_recovery_phase=True` — for
`recoverable_failure` episodes, that segment *is* the recovery behavior.

`extended_max_steps` is capped at 980 (LIBERO's underlying robosuite env has a fixed
`horizon=1000`, hardcoded outside lerobot's control; stepping past it raises an error).
For `libero_10` (`standard_max_steps=520`), the default `--extension-factor 2.0` would
ask for 1040 and gets capped to 980 — lower `--extension-factor` if you want the actual
2x ratio to hold for that suite.

### Output

- `<output-dir>/dataset/` — one `LeRobotDataset` (images, `observation.state`, `action`,
  `next.reward`, `next.success`, `next.done`, `is_recovery_phase`) covering every episode,
  whatever its outcome.
- `<output-dir>/rollout_manifest.json` — per-episode summary (suite, task id, task
  description, seed, outcome, frame count, `first_success_step`) so you can filter
  episodes by outcome without decoding video.

Note: `observation.state` here is a raw `[eef_pos(3), eef_quat(4), gripper_qpos(2)]`
concatenation for analysis, not the axis-angle 8-dim layout used by the published
`lerobot/libero` training dataset.

### Useful flags

- `--task libero_spatial,libero_object` — comma-separated suites.
- `--task-ids 0 1 2` — restrict to specific task ids within the suite(s).
- `--control-mode relative|absolute` — must match how the policy was trained.
- `--extension-factor` — how far past the standard budget to keep trying.
