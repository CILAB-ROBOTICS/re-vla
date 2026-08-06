#!/usr/bin/env python
"""Label collected LIBERO rollouts as success / failure / recoverable_failure and index
them for comparison (e.g. the "False Complete" hypothesis: does SmolVLA keep emitting
action chunks that look like a successful run even after it has actually failed?).

Consumes the `rollout_manifest.json` written by `collect_libero_rollouts.py` (one or
more collection runs) — it does not re-derive labels from the dataset itself, since the
manifest already carries the ground-truth outcome computed during collection (see that
script's "extended-rollout rule" docstring).

Outputs (under --output-dir):
    labels.csv          - one row per episode: outcome, label_id, task, seed, frame
                           counts, source_dir. Easy to filter/join in pandas.
    labels_by_task.json - nested {suite: {task_id: {outcome: [episode indices]}}} index.
    comparison_pairs.json - for each (suite, task_id) with both a success and a
                           failure/recoverable_failure episode, every such
                           (success_episode, other_episode) pair - a ready-made list of
                           same-task success-vs-failure comparisons.

Usage
-----
    python labeling_libero.py \\
        --input-dir ../data_generation/outputs/libero_rollouts_smolvla \\
        --output-dir ../data_generation/outputs/libero_rollouts_smolvla/labels

    # combine multiple collection runs (e.g. different suites or policies) into one index
    python labeling_libero.py \\
        --input-dir outputs/libero_rollouts_smolvla --input-dir outputs/libero_rollouts_pi05 \\
        --output-dir outputs/labels_combined
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("labeling_libero")

OUTCOME_SUCCESS = "success"
OUTCOME_RECOVERABLE_FAILURE = "recoverable_failure"
OUTCOME_FAILURE = "failure"
VALID_OUTCOMES = (OUTCOME_SUCCESS, OUTCOME_RECOVERABLE_FAILURE, OUTCOME_FAILURE)
LABEL_IDS = {OUTCOME_SUCCESS: 0, OUTCOME_RECOVERABLE_FAILURE: 1, OUTCOME_FAILURE: 2}

# Which outcome pairs count as a same-task "success vs not-success" comparison, and
# under what name. failure and recoverable_failure are both compared against success;
# they're kept distinct so a "did it recover?" analysis can filter comparison_type.
COMPARISON_PAIRS = [
    (OUTCOME_SUCCESS, OUTCOME_FAILURE, "success_vs_failure"),
    (OUTCOME_SUCCESS, OUTCOME_RECOVERABLE_FAILURE, "success_vs_recoverable_failure"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label LIBERO rollouts (success/failure/recoverable_failure) and index them for comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        dest="input_dirs",
        required=True,
        help="A collect_libero_rollouts.py output dir (containing rollout_manifest.json). "
        "Repeat to combine multiple collection runs into one labeled index.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_manifest(input_dir: Path) -> list[dict[str, Any]]:
    manifest_path = input_dir / "rollout_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No rollout_manifest.json found under {input_dir}")
    with open(manifest_path) as f:
        episodes = json.load(f)
    for ep in episodes:
        if ep.get("outcome") not in VALID_OUTCOMES:
            raise ValueError(f"Unexpected outcome {ep.get('outcome')!r} in {manifest_path} (episode {ep})")
    logger.info("Loaded %d episodes from %s", len(episodes), manifest_path)
    return episodes


def build_labels(input_dirs: list[Path]) -> list[dict[str, Any]]:
    labels = []
    for input_dir in input_dirs:
        for ep in load_manifest(input_dir):
            labels.append(
                {
                    "source_dir": str(input_dir),
                    "dataset_dir": str(input_dir / "dataset"),
                    "episode_index": ep["episode_index"],
                    "suite": ep["suite"],
                    "task_id": ep["task_id"],
                    "task_description": ep["task_description"],
                    "seed": ep["seed"],
                    "outcome": ep["outcome"],
                    "label_id": LABEL_IDS[ep["outcome"]],
                    "num_frames": ep["num_frames"],
                    "standard_max_steps": ep["standard_max_steps"],
                    "extended_max_steps": ep["extended_max_steps"],
                    "first_success_step": ep["first_success_step"],
                }
            )
    return labels


def write_labels_csv(labels: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "source_dir",
        "dataset_dir",
        "episode_index",
        "suite",
        "task_id",
        "task_description",
        "seed",
        "outcome",
        "label_id",
        "num_frames",
        "standard_max_steps",
        "extended_max_steps",
        "first_success_step",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labels)


def build_labels_by_task(labels: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in labels:
        ref = {"source_dir": row["source_dir"], "episode_index": row["episode_index"], "seed": row["seed"]}
        by_task[row["suite"]][str(row["task_id"])][row["outcome"]].append(ref)
    return {suite: dict(tasks) for suite, tasks in by_task.items()}


def build_comparison_pairs(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Group episode refs by (source_dir, suite, task_id, outcome) so pairs only ever
    # compare episodes of the *same* task (and same collection run/policy).
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in labels:
        key = (row["source_dir"], row["suite"], row["task_id"], row["outcome"])
        groups[key].append(
            {
                "source_dir": row["source_dir"],
                "dataset_dir": row["dataset_dir"],
                "episode_index": row["episode_index"],
                "seed": row["seed"],
            }
        )

    pairs = []
    seen_task_keys = {(source_dir, suite, task_id) for source_dir, suite, task_id, _ in groups}
    for source_dir, suite, task_id in sorted(seen_task_keys):
        for outcome_a, outcome_b, comparison_type in COMPARISON_PAIRS:
            eps_a = groups.get((source_dir, suite, task_id, outcome_a), [])
            eps_b = groups.get((source_dir, suite, task_id, outcome_b), [])
            for ep_a, ep_b in product(eps_a, eps_b):
                pairs.append(
                    {
                        "comparison_type": comparison_type,
                        "source_dir": source_dir,
                        "suite": suite,
                        "task_id": task_id,
                        "success_episode": ep_a,
                        "other_episode": ep_b,
                    }
                )
    return pairs


def print_summary(labels: list[dict[str, Any]]) -> None:
    overall = Counter(row["outcome"] for row in labels)
    logger.info("Overall: %d episodes -> %s", len(labels), dict(overall))

    per_task_counts: dict[tuple[str, str, int], Counter] = defaultdict(Counter)
    for row in labels:
        per_task_counts[(row["source_dir"], row["suite"], row["task_id"])][row["outcome"]] += 1

    logger.info("Per task:")
    for (source_dir, suite, task_id), counts in sorted(per_task_counts.items()):
        logger.info(
            "  [%s] %s/task_id=%d: %s",
            source_dir,
            suite,
            task_id,
            {o: counts.get(o, 0) for o in VALID_OUTCOMES},
        )


def main() -> None:
    args = parse_args()
    input_dirs = [Path(d) for d in args.input_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = build_labels(input_dirs)
    if not labels:
        raise ValueError("No episodes found across the given --input-dir(s).")

    print_summary(labels)

    labels_csv_path = output_dir / "labels.csv"
    write_labels_csv(labels, labels_csv_path)

    labels_by_task_path = output_dir / "labels_by_task.json"
    with open(labels_by_task_path, "w") as f:
        json.dump(build_labels_by_task(labels), f, indent=2)

    pairs = build_comparison_pairs(labels)
    comparison_pairs_path = output_dir / "comparison_pairs.json"
    with open(comparison_pairs_path, "w") as f:
        json.dump(pairs, f, indent=2)

    logger.info("Wrote %s (%d rows)", labels_csv_path, len(labels))
    logger.info("Wrote %s", labels_by_task_path)
    logger.info("Wrote %s (%d comparison pairs)", comparison_pairs_path, len(pairs))


if __name__ == "__main__":
    main()
