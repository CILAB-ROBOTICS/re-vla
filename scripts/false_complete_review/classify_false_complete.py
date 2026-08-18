#!/usr/bin/env python
"""Separate likely False Complete from ordinary failure using rollout evidence.

The default robust policy uses the failure/next-phase/no-recovery core. Human annotations
are never accepted as input fields. The module never reads rollout outcome, reward,
success, done, episode length, or trajectory similarity.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


RULE_VERSIONS = {
    "robust": "false-complete-separator-v2-robust",
    "strict": "false-complete-separator-v1-strict",
}
DECISION_POLICIES = tuple(RULE_VERSIONS)
CLASSES = ("false_complete", "failure", "not_failure", "uncertain")
NORMALIZED_FIELDS = {
    "task_incomplete",
    "failure_event_detected",
    "next_phase_entry",
    "terminal_like",
    "valid_recovery_attempt",
}
DETECTOR_SUMMARY_FIELDS = {
    "task_incomplete",
    "detector_failure_event_count",
    "detector_next_phase_state",
    "detector_terminal_like_state",
    "detector_valid_recovery_attempt_count",
}
IDENTITY_FIELDS = ("review_id", "suite", "task_id", "episode_index", "seed")
FORBIDDEN_PROXY_COLUMNS = {
    "outcome",
    "collection_outcome_posthoc",
    "reward",
    "next.reward",
    "success",
    "next.success",
    "done",
    "next.done",
    "length",
    "frame_count",
    "num_frames",
    "similarity",
}


def parse_tristate(value: Any, field: str) -> bool | None:
    text = "" if value is None else str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", "not_applicable"}:
        return False
    if text in {"", "unknown", "uncertain", "null", "none", "na", "n/a"}:
        return None
    raise ValueError(f"{field}: expected true/false/unknown, got {value!r}")


def parse_nonnegative_int(value: Any, field: str) -> int:
    try:
        result = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}: expected a non-negative integer, got {value!r}") from exc
    if result < 0:
        raise ValueError(f"{field}: expected a non-negative integer, got {result}")
    return result


def parse_failure_types(value: Any) -> list[str]:
    text = "" if value is None else str(value).strip()
    if not text:
        return []
    for separator in (";", ","):
        text = text.replace(separator, "|")
    return sorted({part.strip() for part in text.split("|") if part.strip()})


def detect_profile(fieldnames: set[str], requested: str) -> str:
    if requested != "auto":
        required = NORMALIZED_FIELDS if requested == "normalized" else DETECTOR_SUMMARY_FIELDS
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"{requested} profile missing columns: {', '.join(missing)}")
        return requested
    if NORMALIZED_FIELDS <= fieldnames:
        return "normalized"
    if DETECTOR_SUMMARY_FIELDS <= fieldnames:
        return "detector-summary"
    raise ValueError(
        "Could not detect input profile. Provide normalized rollout evidence columns "
        "or automatic detector-summary columns described in README.md."
    )


def normalize_row(row: dict[str, str], profile: str) -> dict[str, Any]:
    if profile == "normalized":
        evidence = {
            "task_incomplete": parse_tristate(row["task_incomplete"], "task_incomplete"),
            "failure_event_detected": parse_tristate(
                row["failure_event_detected"], "failure_event_detected"
            ),
            "next_phase_entry": parse_tristate(row["next_phase_entry"], "next_phase_entry"),
            "terminal_like": parse_tristate(row["terminal_like"], "terminal_like"),
            "valid_recovery_attempt": parse_tristate(
                row["valid_recovery_attempt"], "valid_recovery_attempt"
            ),
            "failure_types": parse_failure_types(row.get("failure_types", "")),
        }
    else:
        failure_count = parse_nonnegative_int(
            row["detector_failure_event_count"], "detector_failure_event_count"
        )
        recovery_count = parse_nonnegative_int(
            row["detector_valid_recovery_attempt_count"],
            "detector_valid_recovery_attempt_count",
        )
        evidence = {
            "task_incomplete": parse_tristate(row["task_incomplete"], "task_incomplete"),
            "failure_event_detected": failure_count > 0,
            "next_phase_entry": parse_tristate(
                row["detector_next_phase_state"], "detector_next_phase_state"
            ),
            "terminal_like": parse_tristate(
                row["detector_terminal_like_state"], "detector_terminal_like_state"
            ),
            "valid_recovery_attempt": recovery_count > 0,
            "failure_types": parse_failure_types(row.get("detector_failure_types", "")),
        }
    evidence.update({field: row.get(field, "") for field in IDENTITY_FIELDS})
    return evidence


def load_analysis_root(analysis_root: Path) -> list[dict[str, str]]:
    """Build normalized rows from existing taxonomy/failure/terminal detector JSONs."""
    if (analysis_root / "taxonomy" / "episodes").is_dir():
        suite_dirs = [analysis_root]
    else:
        suite_dirs = sorted(
            path
            for path in analysis_root.iterdir()
            if path.is_dir() and (path / "taxonomy" / "episodes").is_dir()
        )
    if not suite_dirs:
        raise FileNotFoundError(f"No suite taxonomy/episodes directories under {analysis_root}")

    rows: list[dict[str, str]] = []
    for suite_dir in suite_dirs:
        taxonomy_paths = sorted((suite_dir / "taxonomy" / "episodes").glob("episode_*.json"))
        if not taxonomy_paths:
            raise FileNotFoundError(f"No taxonomy episode JSONs under {suite_dir}")
        for taxonomy_path in taxonomy_paths:
            failure_path = suite_dir / "failure_recovery" / "episodes" / taxonomy_path.name
            terminal_path = suite_dir / "terminal_like" / "episodes" / taxonomy_path.name
            if not failure_path.is_file() or not terminal_path.is_file():
                raise FileNotFoundError(f"Missing matched detector JSON for {taxonomy_path}")
            taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
            episode_index = int(taxonomy["episode_index"])
            if (
                int(failure["episode_index"]) != episode_index
                or int(terminal["episode_index"]) != episode_index
            ):
                raise ValueError(f"Episode identity mismatch for {taxonomy_path}")

            events = failure.get("failure_events", [])
            sequences = terminal.get("sequences", [])
            task_complete = taxonomy.get("task_complete_simulator")
            if task_complete not in (True, False, None):
                raise ValueError(f"Invalid task_complete_simulator for {taxonomy_path}")
            terminal_true = any(item.get("terminal_like") is True for item in sequences)
            terminal_unknown = any(item.get("terminal_like") is None for item in sequences)
            failure_types = {
                str(item["failure_event_type"])
                for item in events
                if item.get("failure_event_type")
            }
            rows.append(
                {
                    "review_id": str(taxonomy.get("review_id", "")),
                    "suite": str(taxonomy.get("suite", suite_dir.name)),
                    "task_id": str(taxonomy.get("task_id", "")),
                    "episode_index": str(episode_index),
                    "seed": str(taxonomy.get("seed", "")),
                    "task_incomplete": "unknown" if task_complete is None else str(not task_complete).lower(),
                    "failure_event_detected": str(bool(events)).lower(),
                    "next_phase_entry": str(
                        any(item.get("next_phase_entry_timestep") is not None for item in events)
                    ).lower(),
                    "terminal_like": (
                        "true" if terminal_true else "unknown" if terminal_unknown else "false"
                    ),
                    "valid_recovery_attempt": str(
                        any(item.get("valid_recovery_attempt") is True for item in events)
                    ).lower(),
                    "failure_types": "|".join(sorted(failure_types)),
                }
            )
    return rows


def classify(
    evidence: dict[str, Any], decision_policy: str = "robust"
) -> tuple[str, str, str, float | None]:
    if decision_policy not in DECISION_POLICIES:
        raise ValueError(f"Unknown decision policy: {decision_policy}")
    task_incomplete = evidence["task_incomplete"]
    if task_incomplete is False:
        return "not_failure", "task_complete", "high", 0.0
    if task_incomplete is None:
        return "uncertain", "task_completion_unknown", "low", None

    failure = evidence["failure_event_detected"]
    next_phase = evidence["next_phase_entry"]
    terminal = evidence["terminal_like"]
    recovery = evidence["valid_recovery_attempt"]
    core_pattern = failure is True and next_phase is True and recovery is False
    if core_pattern:
        if decision_policy == "strict":
            if terminal is True:
                return "false_complete", "strict_false_complete_pattern", "high", 1.0
            if terminal is False:
                return "failure", "strict_terminal_like_not_observed", "high", 0.75
            return "uncertain", "strict_terminal_like_unknown", "low", None
        if terminal is True:
            return "false_complete", f"{decision_policy}_core_pattern+terminal_support", "high", 1.0
        if terminal is False:
            return "false_complete", f"{decision_policy}_core_pattern+terminal_not_observed", "medium", 0.8
        return "false_complete", f"{decision_policy}_core_pattern+terminal_unknown", "medium", 0.8

    contradictions = []
    if failure is False:
        contradictions.append("no_failure_event")
    if next_phase is False:
        contradictions.append("no_next_phase_entry")
    if decision_policy == "strict" and terminal is False:
        contradictions.append("not_terminal_like")
    if recovery is True:
        contradictions.append("valid_recovery_attempt")
    if contradictions:
        return "failure", "+".join(contradictions), "high", 0.0
    return "uncertain", "required_core_evidence_unknown", "low", None


def classify_rows(
    rows: list[dict[str, str]], profile: str, decision_policy: str = "robust"
) -> list[dict[str, Any]]:
    output = []
    for line_number, row in enumerate(rows, start=2):
        try:
            evidence = normalize_row(row, profile)
            label, reason, confidence, score = classify(evidence, decision_policy)
        except ValueError as exc:
            raise ValueError(f"line {line_number}: {exc}") from exc
        review_recommended = label in {"false_complete", "uncertain"}
        if label == "uncertain" or (label == "false_complete" and confidence != "high"):
            review_priority = "high"
        elif label == "false_complete":
            review_priority = "normal"
        else:
            review_priority = "low"
        output.append(
            {
                **{field: evidence[field] for field in IDENTITY_FIELDS},
                "task_incomplete": evidence["task_incomplete"],
                "failure_event_detected": evidence["failure_event_detected"],
                "next_phase_entry": evidence["next_phase_entry"],
                "terminal_like": evidence["terminal_like"],
                "valid_recovery_attempt": evidence["valid_recovery_attempt"],
                "failure_types": "|".join(evidence["failure_types"]),
                "classification": label,
                "classification_reason": reason,
                "confidence": confidence,
                "false_complete_evidence_score": score,
                "review_recommended": review_recommended,
                "review_priority": review_priority,
                "decision_policy": decision_policy,
                "rule_version": RULE_VERSIONS[decision_policy],
                "human_label": "",
                "human_failure_type": "",
                "human_notes": "",
            }
        )
    return output


def summarize(
    rows: list[dict[str, Any]],
    profile: str,
    input_fields: set[str],
    decision_policy: str = "robust",
) -> dict[str, Any]:
    counts = Counter(row["classification"] for row in rows)
    confidence_counts = Counter(row["confidence"] for row in rows)
    subtype_counts: Counter[str] = Counter()
    for row in rows:
        if row["classification"] == "false_complete":
            subtype_counts.update(parse_failure_types(row["failure_types"]) or ["unspecified"])
    return {
        "rule_version": RULE_VERSIONS[decision_policy],
        "decision_policy": decision_policy,
        "input_profile": profile,
        "episodes": len(rows),
        "classification_counts": {name: counts.get(name, 0) for name in CLASSES},
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "review_recommended_count": sum(bool(row["review_recommended"]) for row in rows),
        "false_complete_failure_type_counts": dict(sorted(subtype_counts.items())),
        "forbidden_proxy_columns_ignored": sorted(input_fields & FORBIDDEN_PROXY_COLUMNS),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Episode evidence CSV")
    source.add_argument(
        "--analysis-root",
        type=Path,
        help="Local root containing suite/taxonomy, failure_recovery and terminal_like JSONs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output review CSV (default: <input>.false_complete_review.csv)",
    )
    parser.add_argument("--summary", type=Path, help="Optional summary JSON path")
    parser.add_argument(
        "--profile",
        choices=("auto", "normalized", "detector-summary"),
        default="auto",
    )
    parser.add_argument(
        "--decision-policy",
        choices=DECISION_POLICIES,
        default="robust",
        help="robust uses failure/next-phase/no-recovery; strict additionally requires terminal-like=true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.analysis_root is not None:
        rows = load_analysis_root(args.analysis_root)
        input_fields = set(rows[0]) if rows else set()
        profile = "normalized"
    else:
        with args.input.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            input_fields = set(reader.fieldnames or [])
            profile = detect_profile(input_fields, args.profile)
            rows = list(reader)
    if not rows:
        raise ValueError("Input CSV contains no episode rows")

    classified = classify_rows(rows, profile, args.decision_policy)
    if args.output is not None:
        output_path = args.output
    elif args.input is not None:
        output_path = args.input.with_name(f"{args.input.stem}.false_complete_review.csv")
    else:
        output_path = Path(f"{args.analysis_root.name}.false_complete_review.csv")
    summary_path = args.summary or output_path.with_suffix(".summary.json")
    if output_path.exists() or summary_path.exists():
        raise FileExistsError("Refusing to overwrite an existing prediction or summary output")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(classified[0]))
        writer.writeheader()
        writer.writerows(classified)

    summary = summarize(classified, profile, input_fields, args.decision_policy)
    summary["output_csv"] = str(output_path)
    summary["summary_json"] = str(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
