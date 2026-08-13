#!/usr/bin/env python
"""Validate False Complete annotation files without defining the scientific rule.

The validator checks syntax, evidence bounds, subtype consistency, and review/key
integrity. It deliberately does not infer or adjudicate ``false_complete``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


TRISTATE = {"", "true", "false", "uncertain"}
NULLABLE_BOOLEAN = {"", "true", "false"}
SUBTYPES = {"missed_grasp", "slip", "perturbation", "missed_orientation"}
ANNOTATION_FIELDS = {
    "task_complete_visual",
    "completion_like_behavior",
    "false_complete",
    "false_complete_subtypes",
    "confidence",
    "evidence_start_timestamp",
    "evidence_end_timestamp",
    "reviewer",
    "review_notes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a blinded False Complete annotation queue.")
    parser.add_argument("--annotations-csv", required=True)
    parser.add_argument("--review-key-csv")
    parser.add_argument("--summary-json")
    return parser.parse_args()


def parse_subtypes(value: str) -> list[str]:
    if not value.strip():
        return []
    return [item.strip() for item in value.replace(",", ";").split(";") if item.strip()]


def validate_rows(rows: list[dict[str, str]], key_rows: list[dict[str, str]] | None = None) -> dict[str, Any]:
    errors: list[str] = []
    review_ids = [row.get("review_id", "") for row in rows]
    duplicate_ids = sorted(review_id for review_id, count in Counter(review_ids).items() if count > 1)
    if "" in review_ids:
        errors.append("one or more rows have a blank review_id")
    if duplicate_ids:
        errors.append(f"duplicate review_id values: {duplicate_ids}")

    annotated_rows = 0
    for line_number, row in enumerate(rows, start=2):
        missing = sorted(ANNOTATION_FIELDS - set(row))
        if missing:
            errors.append(f"line {line_number}: missing annotation fields {missing}")
            continue
        if row["task_complete_visual"].strip().lower() not in TRISTATE:
            errors.append(f"line {line_number}: invalid task_complete_visual")
        if row["completion_like_behavior"].strip().lower() not in TRISTATE:
            errors.append(f"line {line_number}: invalid completion_like_behavior")
        false_complete = row["false_complete"].strip().lower()
        if false_complete not in NULLABLE_BOOLEAN:
            errors.append(f"line {line_number}: false_complete must be blank, true, or false")

        subtypes = parse_subtypes(row["false_complete_subtypes"])
        unknown_subtypes = sorted(set(subtypes) - SUBTYPES)
        if unknown_subtypes:
            errors.append(f"line {line_number}: unknown subtypes {unknown_subtypes}")
        if subtypes and false_complete != "true":
            errors.append(f"line {line_number}: subtypes require false_complete=true")

        confidence = row["confidence"].strip()
        if confidence:
            try:
                confidence_value = float(confidence)
                if not 0.0 <= confidence_value <= 1.0:
                    raise ValueError
            except ValueError:
                errors.append(f"line {line_number}: confidence must be between 0 and 1")

        evidence_start = row["evidence_start_timestamp"].strip()
        evidence_end = row["evidence_end_timestamp"].strip()
        if bool(evidence_start) != bool(evidence_end):
            errors.append(f"line {line_number}: evidence timestamps must be supplied as a pair")
        elif evidence_start:
            try:
                start = float(evidence_start)
                end = float(evidence_end)
                lower = min(float(row["agent_from_timestamp"]), float(row["wrist_from_timestamp"]))
                upper = max(float(row["agent_to_timestamp"]), float(row["wrist_to_timestamp"]))
                if not lower <= start <= end <= upper:
                    raise ValueError
            except (KeyError, ValueError):
                errors.append(f"line {line_number}: evidence timestamps fall outside the episode clip")

        has_annotation = any(row[field].strip() for field in ANNOTATION_FIELDS - {"review_notes", "reviewer"})
        if has_annotation:
            annotated_rows += 1
            if not row["reviewer"].strip():
                errors.append(f"line {line_number}: reviewer is required for an annotated row")

    if key_rows is not None:
        key_ids = [row.get("review_id", "") for row in key_rows]
        if len(key_ids) != len(set(key_ids)):
            errors.append("review key contains duplicate review_id values")
        missing_from_key = sorted(set(review_ids) - set(key_ids))
        missing_from_annotations = sorted(set(key_ids) - set(review_ids))
        if missing_from_key or missing_from_annotations:
            errors.append(
                "review/key ids are not one-to-one: "
                f"missing_from_key={missing_from_key}, missing_from_annotations={missing_from_annotations}"
            )

    return {
        "valid": not errors,
        "rows": len(rows),
        "annotated_rows": annotated_rows,
        "unreviewed_rows": len(rows) - annotated_rows,
        "errors": errors,
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    rows = read_csv(Path(args.annotations_csv))
    if not rows:
        raise ValueError("Annotations CSV is empty")
    key_rows = read_csv(Path(args.review_key_csv)) if args.review_key_csv else None
    summary = validate_rows(rows, key_rows)
    if args.summary_json:
        path = Path(args.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    if not summary["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
