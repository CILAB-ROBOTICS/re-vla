#!/usr/bin/env python
"""Build a blinded, video-addressable False Complete review queue.

This utility joins the unreviewed template to LeRobot episode metadata. The reviewer
queue deliberately omits rollout outcome and success/reward signals to reduce anchoring.
It assigns no scientific label; every annotation field remains blank.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


CAMERAS = {
    "agent": "observation.images.image",
    "wrist": "observation.images.image2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a blinded False Complete review queue.")
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stable_review_id(row: dict[str, str]) -> str:
    material = "|".join(
        [row["source_dir"], row["suite"], row["task_id"], row["episode_index"], row["seed"]]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def load_episode_metadata(dataset_dir: Path) -> tuple[dict[int, dict[str, Any]], int, str]:
    info = json.loads((dataset_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    episode_file = dataset_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(episode_file)
    rows = {int(row["episode_index"]): row for row in table.to_pylist()}
    return rows, int(info["fps"]), str(info["video_path"])


def build_queue(review_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in review_rows:
        by_dataset[row["dataset_dir"]].append(row)

    blinded: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for dataset_value, rows in by_dataset.items():
        dataset_dir = Path(dataset_value)
        metadata, fps, video_template = load_episode_metadata(dataset_dir)
        for row in rows:
            episode_index = int(row["episode_index"])
            episode = metadata[episode_index]
            if int(episode["length"]) != int(row["num_frames"]):
                raise ValueError(
                    f"Frame mismatch for {row['suite']} episode {episode_index}: "
                    f"manifest={row['num_frames']} metadata={episode['length']}"
                )
            review_id = stable_review_id(row)
            if review_id in seen_ids:
                raise ValueError(f"Duplicate review id: {review_id}")
            seen_ids.add(review_id)

            queue_row: dict[str, Any] = {
                "review_id": review_id,
                "suite": row["suite"],
                "task_id": row["task_id"],
                "task_description": row["task_description"],
                "fps": fps,
            }
            for view, video_key in CAMERAS.items():
                chunk = int(episode[f"videos/{video_key}/chunk_index"])
                file_index = int(episode[f"videos/{video_key}/file_index"])
                queue_row[f"{view}_video_path"] = str(
                    dataset_dir
                    / video_template.format(video_key=video_key, chunk_index=chunk, file_index=file_index)
                )
                queue_row[f"{view}_from_timestamp"] = episode[f"videos/{video_key}/from_timestamp"]
                queue_row[f"{view}_to_timestamp"] = episode[f"videos/{video_key}/to_timestamp"]

            queue_row.update(
                {
                    "task_complete_visual": "",
                    "completion_like_behavior": "",
                    "false_complete": "",
                    "false_complete_subtypes": "",
                    "confidence": "",
                    "evidence_start_timestamp": "",
                    "evidence_end_timestamp": "",
                    "reviewer": "",
                    "review_notes": "",
                }
            )
            blinded.append(queue_row)
            key.append(
                {
                    "review_id": review_id,
                    "source_dir": row["source_dir"],
                    "dataset_dir": row["dataset_dir"],
                    "episode_index": row["episode_index"],
                    "suite": row["suite"],
                    "task_id": row["task_id"],
                    "seed": row["seed"],
                    "outcome": row["outcome"],
                }
            )

    # Hash order is deterministic but hides collection/outcome ordering from reviewers.
    blinded.sort(key=lambda row: row["review_id"])
    key.sort(key=lambda row: row["review_id"])
    return blinded, key


def write_csv(rows: list[dict[str, Any]], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing review artifact: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    with Path(args.review_csv).open(encoding="utf-8", newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    if not review_rows:
        raise ValueError("Review CSV is empty")
    blinded, key = build_queue(review_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(blinded, output_dir / "false_complete_blinded_queue.csv", args.overwrite)
    write_csv(key, output_dir / "false_complete_review_key.csv", args.overwrite)
    print(f"Prepared {len(blinded)} blinded review rows and {len(key)} key rows")


if __name__ == "__main__":
    main()
