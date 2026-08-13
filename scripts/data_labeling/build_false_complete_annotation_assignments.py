#!/usr/bin/env python
"""Build blinded annotation assignments without opening or labeling review clips."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path


def deterministic_overlap(rows: list[dict[str, str]], overlap_count: int) -> set[str]:
    if overlap_count < 0 or overlap_count > len(rows):
        raise ValueError("overlap_count must be between zero and the number of rows")
    suites = sorted({row["suite"] for row in rows})
    base = overlap_count // len(suites)
    remainder = overlap_count % len(suites)
    selected: set[str] = set()
    for index, suite in enumerate(suites):
        count = base + (1 if index < remainder else 0)
        suite_rows = [row for row in rows if row["suite"] == suite]
        suite_rows.sort(
            key=lambda row: hashlib.sha256(
                f"overlap-v1|{row['review_id']}".encode("utf-8")
            ).hexdigest()
        )
        selected.update(row["review_id"] for row in suite_rows[:count])
    return selected


def build_assignments(
    rows: list[dict[str, str]], primary: str, secondary: str, overlap_count: int
) -> list[dict[str, str]]:
    if not primary.strip() or not secondary.strip() or primary == secondary:
        raise ValueError("primary and secondary annotators must be distinct nonblank IDs")
    overlap = deterministic_overlap(rows, overlap_count)
    assignments: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: item["review_id"]):
        assignments.append(
            {
                "assignment_id": f"{row['review_id']}:{primary}",
                "review_id": row["review_id"],
                "annotator_id": primary,
                "suite": row["suite"],
                "task_id": row["task_id"],
                "task_description": row["task_description"],
                "review_clip_path": row["review_clip_path"],
                "task_complete_visual": "",
                "terminal_like_human": "",
                "next_phase_entry_human": "",
                "confidence": "",
                "evidence_start_timestamp": "",
                "evidence_end_timestamp": "",
                "review_notes": "",
            }
        )
        if row["review_id"] in overlap:
            duplicate = dict(assignments[-1])
            duplicate["assignment_id"] = f"{row['review_id']}:{secondary}"
            duplicate["annotator_id"] = secondary
            assignments.append(duplicate)
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-packet-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--primary-annotator", required=True)
    parser.add_argument("--secondary-annotator", required=True)
    parser.add_argument("--overlap-count", type=int, default=50)
    args = parser.parse_args()
    with Path(args.review_packet_csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assignments = build_assignments(
        rows, args.primary_annotator, args.secondary_annotator, args.overlap_count
    )
    output = Path(args.output_csv)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing assignment file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(assignments[0]))
        writer.writeheader()
        writer.writerows(assignments)
    counts = Counter(row["annotator_id"] for row in assignments)
    print(f"Prepared {len(assignments)} assignments: {dict(counts)}")


if __name__ == "__main__":
    main()
