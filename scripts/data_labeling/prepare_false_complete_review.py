#!/usr/bin/env python
"""Prepare a semantics-neutral review table for independent False Complete labels.

The rollout ``outcome`` is a time-budget result. ``false_complete`` is intentionally
kept as a separate, nullable human-review field. This script never infers or assigns
that label from outcome, reward, success, episode length, or any other proxy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from labeling_libero import LABEL_IDS, VALID_OUTCOMES, load_manifest


FALSE_COMPLETE_SUBTYPES = (
    "missed_grasp",
    "slip",
    "perturbation",
    "missed_orientation",
)

REVIEW_FIELDS = (
    "source_dir",
    "dataset_dir",
    "episode_index",
    "suite",
    "task_id",
    "task_description",
    "seed",
    "outcome",
    "outcome_label_id",
    "num_frames",
    "standard_max_steps",
    "extended_max_steps",
    "first_success_step",
    "false_complete",
    "false_complete_subtypes",
    "review_status",
    "evidence_start_frame",
    "evidence_end_frame",
    "reviewer",
    "review_notes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an unreviewed False Complete annotation table without assigning scientific labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        dest="input_dirs",
        required=True,
        help="A collection directory containing rollout_manifest.json. Repeat to combine suites.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only this script's review template/schema files if they already exist.",
    )
    return parser.parse_args()


def build_review_rows(input_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for input_dir in input_dirs:
        for episode in load_manifest(input_dir):
            outcome = episode["outcome"]
            if outcome not in VALID_OUTCOMES:
                raise ValueError(f"Unexpected outcome {outcome!r} in {input_dir}")
            rows.append(
                {
                    "source_dir": str(input_dir),
                    "dataset_dir": str(input_dir / "dataset"),
                    "episode_index": episode["episode_index"],
                    "suite": episode["suite"],
                    "task_id": episode["task_id"],
                    "task_description": episode["task_description"],
                    "seed": episode["seed"],
                    "outcome": outcome,
                    "outcome_label_id": LABEL_IDS[outcome],
                    "num_frames": episode["num_frames"],
                    "standard_max_steps": episode["standard_max_steps"],
                    "extended_max_steps": episode["extended_max_steps"],
                    "first_success_step": episode["first_success_step"],
                    # Empty means genuinely unreviewed. It must not be interpreted as false.
                    "false_complete": "",
                    "false_complete_subtypes": "",
                    "review_status": "unreviewed",
                    "evidence_start_frame": "",
                    "evidence_end_frame": "",
                    "reviewer": "",
                    "review_notes": "",
                }
            )
    return rows


def schema_document() -> dict[str, Any]:
    return {
        "schema_version": "0.1-draft",
        "scientific_status": "review_protocol_required_before_label_assignment",
        "invariants": [
            "false_complete is independent of time-budget outcome",
            "blank false_complete means unreviewed, never false",
            "subtypes may be assigned only when false_complete is true",
            "no label may be inferred from outcome, reward, success, or episode length alone",
        ],
        "fields": {
            "false_complete": {"type": "nullable_boolean", "allowed": [True, False, None]},
            "false_complete_subtypes": {
                "type": "nullable_multi_label",
                "allowed": list(FALSE_COMPLETE_SUBTYPES),
            },
            "review_status": {
                "type": "enum",
                "allowed": ["unreviewed", "reviewed", "adjudication_required"],
            },
            "evidence_start_frame": {"type": "nullable_nonnegative_integer"},
            "evidence_end_frame": {"type": "nullable_nonnegative_integer"},
            "reviewer": {"type": "nullable_string"},
            "review_notes": {"type": "nullable_string"},
        },
    }


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    review_path = output_dir / "false_complete_review.csv"
    schema_path = output_dir / "false_complete_schema.json"
    existing = [path for path in (review_path, schema_path) if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing review artifacts: {joined}")

    with review_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with schema_path.open("w", encoding="utf-8") as handle:
        json.dump(schema_document(), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return review_path, schema_path


def main() -> None:
    args = parse_args()
    rows = build_review_rows([Path(value) for value in args.input_dirs])
    if not rows:
        raise ValueError("No episodes found across the input directories")
    review_path, schema_path = write_outputs(rows, Path(args.output_dir), args.overwrite)
    print(f"Prepared {len(rows)} unreviewed episodes: {review_path}")
    print(f"Wrote draft schema: {schema_path}")


if __name__ == "__main__":
    main()
