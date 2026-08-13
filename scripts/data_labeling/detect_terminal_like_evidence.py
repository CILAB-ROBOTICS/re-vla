#!/usr/bin/env python
"""Detect automatic terminal-like evidence from ordered, label-free events."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2] if len(SCRIPT_PATH.parents) > 2 else Path.cwd()
DEFAULT_CONFIG = REPO_ROOT / "research" / "false_complete" / "TERMINAL_LIKE_DETECTOR_V0_1.json"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _events(record: dict[str, Any], event_type: str) -> list[dict[str, Any]]:
    return sorted(
        (event for event in record.get("events", []) if event.get("event_type") == event_type),
        key=lambda event: int(event["timestep"]),
    )


def _binding(phase_record: dict[str, Any], binding_index: int) -> dict[str, Any] | None:
    return next((item for item in phase_record.get("bindings", []) if item.get("binding_index") == binding_index), None)


def _first_between(events: list[dict[str, Any]], start: int, end: int) -> dict[str, Any] | None:
    return next((event for event in events if start <= int(event["timestep"]) <= end), None)


def _terminal_value(components: dict[str, bool | None]) -> tuple[bool | None, float | None]:
    values = list(components.values())
    if any(value is False for value in values):
        return False, sum(value is True for value in values) / len(values)
    if any(value is None for value in values):
        return None, None
    return True, 1.0


def detect_episode(
    phase_record: dict[str, Any],
    failure_record: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    episode_index = int(phase_record["episode_index"])
    if int(failure_record["episode_index"]) != episode_index:
        raise ValueError("Phase/failure episode mismatch")
    last_step = int(phase_record["frame_count"]) - 1
    open_events = sorted(
        (event for event in phase_record.get("gripper_events", []) if event.get("event_type") == "gripper_open_crossing"),
        key=lambda event: int(event["timestep"]),
    )
    settle_events = sorted(
        (event for event in phase_record.get("global_phase_events", []) if event.get("event_type") == "settle_entry"),
        key=lambda event: int(event["timestep"]),
    )
    sequences = []
    for failure in failure_record.get("failure_events", []):
        failure_step = int(failure["failure_timestep"])
        binding_index = int(failure["binding_index"])
        phase_binding = _binding(phase_record, binding_index)
        next_phase_step = failure.get("next_phase_entry_timestep")
        next_phase = isinstance(next_phase_step, int)
        release = None
        retract = None
        settle = None
        no_retry_after: bool | None = None
        release_step = retract_step = settle_step = None

        if next_phase and phase_binding is not None:
            tolerance = int(config["release_match_tolerance_frames"])
            release_event = _first_between(open_events, int(next_phase_step) - tolerance, int(next_phase_step) + tolerance)
            release = release_event is not None
            if release_event is not None:
                release_step = int(release_event["timestep"])
                retract_event = _first_between(
                    _events(phase_binding, "retract_entry"),
                    release_step,
                    release_step + int(config["maximum_retract_delay_frames"]),
                )
                retract = retract_event is not None
                if retract_event is not None:
                    retract_step = int(retract_event["timestep"])
                    settle_event = _first_between(
                        settle_events,
                        retract_step,
                        retract_step + int(config["maximum_settle_delay_frames"]),
                    )
                    settle = settle_event is not None
                    if settle_event is not None:
                        settle_step = int(settle_event["timestep"])
                        later_retry = [
                            event for event_type in ("approach_entry", "grasp_attempt_entry")
                            for event in _events(phase_binding, event_type)
                            if int(event["timestep"]) > settle_step
                        ]
                        if later_retry:
                            no_retry_after = False
                        elif last_step - settle_step >= int(config["minimum_no_retry_observation_frames"]):
                            no_retry_after = True

        valid_recovery = failure.get("valid_recovery_attempt")
        no_valid_recovery = None if valid_recovery is None else not bool(valid_recovery)
        components = {
            "next_phase_entry": next_phase,
            "release": release,
            "retract": retract,
            "settle": settle,
            "no_valid_recovery": no_valid_recovery,
            "no_retry_after_terminal_sequence": no_retry_after,
        }
        terminal_like, score = _terminal_value(components)
        sequences.append({
            "failure_event_type": failure["failure_event_type"],
            "failure_timestep": failure_step,
            "binding_index": binding_index,
            "terminal_like": terminal_like,
            "terminal_like_score": score,
            "components": components,
            "evidence_start_timestep": int(next_phase_step) if next_phase else None,
            "release_timestep": release_step,
            "retract_timestep": retract_step,
            "settle_timestep": settle_step,
            "evidence_end_timestep": settle_step,
        })
    return {
        "record_type": "terminal_like_evidence",
        "episode_index": episode_index,
        "frame_count": int(phase_record["frame_count"]),
        "sequences": sequences,
        "terminal_like_true_count": sum(item["terminal_like"] is True for item in sequences),
        "terminal_like_false_count": sum(item["terminal_like"] is False for item in sequences),
        "terminal_like_unknown_count": sum(item["terminal_like"] is None for item in sequences),
        "automatic_terminal_like_assigned": True,
        "scientific_episode_labels_assigned": False,
        "false_complete_assigned": False,
    }


def run_collection(phase_dir: Path, failure_dir: Path, output_dir: Path, config_path: Path = DEFAULT_CONFIG) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite terminal-like output: {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    phase_paths = sorted(phase_dir.glob("episode_*.json"))
    if not phase_paths:
        raise FileNotFoundError(f"No phase episodes under {phase_dir}")
    episode_dir = output_dir / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=False)
    summaries = []
    for phase_path in phase_paths:
        failure_path = failure_dir / phase_path.name
        if not failure_path.is_file():
            raise FileNotFoundError(f"Missing failure evidence: {failure_path}")
        phase = json.loads(phase_path.read_text(encoding="utf-8"))
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        result = detect_episode(phase, failure, config)
        output_path = episode_dir / phase_path.name
        output_path.write_text(_canonical_json(result) + "\n", encoding="utf-8")
        summaries.append({
            "episode_index": result["episode_index"],
            "frame_count": result["frame_count"],
            "terminal_like_true_count": result["terminal_like_true_count"],
            "terminal_like_false_count": result["terminal_like_false_count"],
            "terminal_like_unknown_count": result["terminal_like_unknown_count"],
            "phase_sha256": file_sha256(phase_path),
            "failure_sha256": file_sha256(failure_path),
            "output_sha256": file_sha256(output_path),
        })
    manifest = {
        "detector_version": config["detector_version"],
        "detector_schema_version": config["schema_version"],
        "detector_status": config["status"],
        "detector_config_hash": canonical_hash(config),
        "detector_code_sha256": file_sha256(SCRIPT_PATH),
        "episode_count": len(summaries),
        "frame_count": sum(item["frame_count"] for item in summaries),
        "terminal_like_true_count": sum(item["terminal_like_true_count"] for item in summaries),
        "terminal_like_false_count": sum(item["terminal_like_false_count"] for item in summaries),
        "terminal_like_unknown_count": sum(item["terminal_like_unknown_count"] for item in summaries),
        "forbidden_proxies_used": [],
        "sealed_reference_used": False,
        "automatic_terminal_like_assigned": True,
        "scientific_episode_labels_assigned": False,
        "false_complete_assigned": False,
        "episodes": summaries,
    }
    manifest_path = output_dir / "detector_manifest.json"
    manifest_path.write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-dir", type=Path, required=True)
    parser.add_argument("--failure-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    print(run_collection(args.phase_dir, args.failure_dir, args.output_dir, args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
