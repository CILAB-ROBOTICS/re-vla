#!/usr/bin/env python
"""Create outcome-blinded side-by-side review clips from a review queue.

The source queue already contains exact episode timestamps for the agent and wrist
streams. This utility only materializes those intervals into compact MP4 files and
creates a reviewer-facing CSV. It never reads an outcome key and never assigns a
False Complete label or subtype.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Sequence


ANNOTATION_FIELDS = [
    "task_complete_visual",
    "completion_like_behavior",
    "false_complete",
    "false_complete_subtypes",
    "confidence",
    "evidence_start_timestamp",
    "evidence_end_timestamp",
    "reviewer",
    "review_notes",
]

PACKET_FIELDS = [
    "review_id",
    "suite",
    "task_id",
    "task_description",
    "fps",
    "review_clip_path",
    *ANNOTATION_FIELDS,
]

FORBIDDEN_REVIEW_FIELDS = {
    "outcome",
    "seed",
    "episode_index",
    "num_frames",
    "source_dir",
    "dataset_dir",
    "agent_video_path",
    "wrist_video_path",
    "agent_from_timestamp",
    "agent_to_timestamp",
    "wrist_from_timestamp",
    "wrist_to_timestamp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create blinded False Complete review clips.")
    parser.add_argument("--queue-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--packet-path-root")
    parser.add_argument("--review-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--crf", type=int, default=25)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_queue(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Review queue is empty")
    if set(rows[0]) & {"outcome", "reward", "success", "next.success"}:
        raise ValueError("Review queue contains a prohibited outcome/success field")
    return rows


def select_rows(
    rows: list[dict[str, str]], review_ids: Sequence[str], limit: int | None
) -> list[dict[str, str]]:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    if review_ids:
        by_id = {row["review_id"]: row for row in rows}
        missing = sorted(set(review_ids) - set(by_id))
        if missing:
            raise ValueError(f"Unknown review IDs: {missing}")
        selected = [by_id[review_id] for review_id in review_ids]
    else:
        selected = sorted(rows, key=lambda row: row["review_id"])
    return selected[:limit] if limit is not None else selected


def clip_bounds(row: dict[str, str]) -> tuple[float, float, float]:
    agent_start = float(row["agent_from_timestamp"])
    agent_end = float(row["agent_to_timestamp"])
    wrist_start = float(row["wrist_from_timestamp"])
    wrist_end = float(row["wrist_to_timestamp"])
    duration = min(agent_end - agent_start, wrist_end - wrist_start)
    if min(agent_start, wrist_start) < 0 or duration <= 0:
        raise ValueError(f"Invalid clip interval for review_id={row['review_id']}")
    return agent_start, wrist_start, duration


def ffmpeg_command(
    row: dict[str, str], output_path: Path, *, ffmpeg: str, width: int, crf: int
) -> list[str]:
    if width < 64 or width % 2:
        raise ValueError("--width must be an even integer of at least 64")
    if not 0 <= crf <= 51:
        raise ValueError("--crf must be between 0 and 51")
    agent_start, wrist_start, duration = clip_bounds(row)
    fps = float(row["fps"])
    if fps <= 0:
        raise ValueError(f"Invalid fps for review_id={row['review_id']}")
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{agent_start:.9f}",
        "-t",
        f"{duration:.9f}",
        "-i",
        row["agent_video_path"],
        "-ss",
        f"{wrist_start:.9f}",
        "-t",
        f"{duration:.9f}",
        "-i",
        row["wrist_video_path"],
        "-filter_complex",
        (
            f"[0:v]setpts=PTS-STARTPTS,scale={width}:-2[agent];"
            f"[1:v]setpts=PTS-STARTPTS,scale={width}:-2[wrist];"
            "[agent][wrist]hstack=inputs=2[review]"
        ),
        "-map",
        "[review]",
        "-an",
        "-r",
        f"{fps:g}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def packet_row(row: dict[str, str], clip_path: Path) -> dict[str, str]:
    packet = {
        "review_id": row["review_id"],
        "suite": row["suite"],
        "task_id": row["task_id"],
        "task_description": row["task_description"],
        "fps": row["fps"],
        "review_clip_path": str(clip_path),
    }
    packet.update({field: row.get(field, "") for field in ANNOTATION_FIELDS})
    if set(packet) & FORBIDDEN_REVIEW_FIELDS:
        raise AssertionError("Reviewer packet contains a forbidden field")
    return packet


def probe_clip(
    clip_path: Path, ffprobe: str, runner: Callable[..., Any] = subprocess.run
) -> dict[str, Any]:
    result = runner(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,duration,nb_read_frames",
            "-of",
            "json",
            str(clip_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"Expected exactly one video stream: {clip_path}")
    stream = streams[0]
    frames = int(stream.get("nb_read_frames", 0))
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    if frames <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Review clip failed decode validation: {clip_path}")
    return {
        "decoded_frames": frames,
        "width": width,
        "height": height,
        "duration_seconds_probe": float(stream.get("duration", 0.0)),
    }


def write_csv(rows: list[dict[str, str]], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing review artifact: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PACKET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_clips(
    rows: list[dict[str, str]],
    output_dir: Path,
    *,
    ffmpeg: str,
    ffprobe: str,
    width: int,
    crf: int,
    overwrite: bool,
    packet_path_root: str | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    packet: list[dict[str, str]] = []
    generated: list[dict[str, Any]] = []
    for row in rows:
        clip_path = clips_dir / f"{row['review_id']}.mp4"
        if clip_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing review clip: {clip_path}")
        command = ffmpeg_command(row, clip_path, ffmpeg=ffmpeg, width=width, crf=crf)
        runner(command, check=True)
        if not clip_path.is_file() or clip_path.stat().st_size <= 0:
            raise RuntimeError(f"ffmpeg did not create a nonempty clip: {clip_path}")
        probe = probe_clip(clip_path, ffprobe, runner)
        packet_clip_path = (
            PurePosixPath(packet_path_root) / "clips" / clip_path.name
            if packet_path_root
            else clip_path
        )
        packet.append(packet_row(row, packet_clip_path))
        _, _, duration = clip_bounds(row)
        generated.append(
            {
                "review_id": row["review_id"],
                "duration_seconds": duration,
                "size_bytes": clip_path.stat().st_size,
                "sha256": hashlib.sha256(clip_path.read_bytes()).hexdigest(),
                **probe,
            }
        )

    write_csv(packet, output_dir / "false_complete_review_packet.csv", overwrite)
    manifest = {
        "schema_version": 1,
        "scientific_labels_assigned": False,
        "outcome_key_used": False,
        "clips": generated,
    }
    manifest_path = output_dir / "clip_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return packet, manifest


def main() -> None:
    args = parse_args()
    rows = select_rows(read_queue(Path(args.queue_csv)), args.review_id, args.limit)
    packet, _ = build_clips(
        rows,
        Path(args.output_dir),
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        width=args.width,
        crf=args.crf,
        overwrite=args.overwrite,
        packet_path_root=args.packet_path_root,
    )
    print(f"Prepared {len(packet)} blinded side-by-side review clips")


if __name__ == "__main__":
    main()
