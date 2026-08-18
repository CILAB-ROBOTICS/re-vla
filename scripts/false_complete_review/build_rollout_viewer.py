#!/usr/bin/env python
"""Build and serve a local, review-ready viewer for rollout videos.

The viewer streams media directly from a local/NAS mount. It does not copy videos and
binds to localhost by default. Human annotations remain in browser localStorage until
the reviewer exports a CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import re
import shutil
import webbrowser
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, unquote, urlsplit


ASSET_TARGETS = {
    "viewer.html": "index.html",
    "viewer.js": "app.js",
    "viewer.css": "styles.css",
}
VIDEO_COLUMNS = ("video_paths", "video_path", "review_clip_path")
HUMAN_COLUMNS = ("human_label", "human_failure_type", "human_notes")
FORBIDDEN_BLIND_COLUMNS = {
    "outcome",
    "collection_outcome_posthoc",
    "reward",
    "next.reward",
    "success",
    "next.success",
    "done",
    "next.done",
    "length",
    "episode_length",
    "frame_count",
    "num_frames",
    "similarity",
    "trajectory_similarity",
}
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".m4v"}
EPISODE_PATTERN = re.compile(r"(?:^|[/_\-])episode[_\-]?0*(\d+)(?=\D|$)", re.IGNORECASE)
RANGE_PATTERN = re.compile(r"bytes=(\d*)-(\d*)$")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not fields or not rows:
        raise ValueError("Review CSV must contain a header and at least one episode")
    if len(fields) != len(set(fields)):
        raise ValueError("Review CSV contains duplicate columns")
    forbidden = sorted({field.strip().lower() for field in fields} & FORBIDDEN_BLIND_COLUMNS)
    if forbidden:
        raise ValueError(
            "Refusing unblinded review metadata columns: " + ", ".join(forbidden)
        )
    return fields, rows


def episode_key(row: dict[str, str], row_number: int) -> str:
    review_id = row.get("review_id", "").strip()
    if review_id:
        return review_id
    suite = row.get("suite", "").strip()
    episode_index = row.get("episode_index", "").strip()
    if suite and episode_index:
        return f"{suite}:{episode_index}"
    assignment_id = row.get("assignment_id", "").strip()
    if assignment_id:
        return assignment_id
    raise ValueError(
        f"CSV row {row_number} needs review_id, suite+episode_index, or assignment_id"
    )


def split_video_value(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"[|;]", value) if part.strip()]


def relative_media_path(path: Path, root: Path) -> str:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(root):
        raise ValueError(f"Video escapes --video-root or is not a file: {path}")
    if resolved.stat().st_size <= 0:
        raise ValueError(f"Video is empty: {path}")
    return resolved.relative_to(root).as_posix()


def resolve_explicit_videos(value: str, root: Path) -> list[str]:
    resolved: list[str] = []
    for raw in split_video_value(value):
        supplied = Path(raw)
        options = [supplied] if supplied.is_absolute() else [root / supplied]
        options.append(root / supplied.name)
        selected = next(
            (
                option
                for option in options
                if option.exists()
                and option.is_file()
                and option.resolve().is_relative_to(root)
            ),
            None,
        )
        if selected is None:
            raise FileNotFoundError(f"Video not found below --video-root: {raw}")
        relative = relative_media_path(selected, root)
        if relative not in resolved:
            resolved.append(relative)
    return resolved


def normalized_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def index_videos(root: Path) -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        relative = path.relative_to(root).as_posix()
        match = EPISODE_PATTERN.search(relative)
        if match:
            result.setdefault(int(match.group(1)), []).append(relative)
    return result


def discover_videos(
    row: dict[str, str], video_index: dict[int, list[str]], row_number: int
) -> list[str]:
    raw_index = row.get("episode_index", "").strip()
    try:
        episode_index = int(raw_index)
    except ValueError as exc:
        raise ValueError(
            f"CSV row {row_number} needs an integer episode_index for video discovery"
        ) from exc
    candidates = video_index.get(episode_index, [])
    if not candidates:
        raise FileNotFoundError(f"No video matched episode_index={episode_index}")
    suite = normalized_token(row.get("suite", ""))
    if suite:
        suite_matches = [path for path in candidates if suite in normalized_token(path)]
        if suite_matches:
            return suite_matches
        if len(candidates) > 1:
            raise ValueError(
                f"Ambiguous videos for suite={row.get('suite')} episode_index={episode_index}; "
                "add a video_path/review_clip_path column"
            )
    return candidates


def video_label(relative: str, index: int) -> str:
    path = PurePosixPath(relative)
    parent = path.parent.name
    if parent and not parent.lower().startswith("chunk-"):
        return parent
    return f"camera {index + 1}"


def build_viewer(
    review_csv: Path,
    video_root: Path,
    output_dir: Path,
    video_column: str | None = None,
) -> dict[str, Any]:
    review_csv = review_csv.resolve(strict=True)
    video_root = video_root.resolve(strict=True)
    if not video_root.is_dir():
        raise NotADirectoryError(video_root)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite viewer directory: {output_dir}")

    fields, rows = read_csv(review_csv)
    if video_column and video_column not in fields:
        raise ValueError(f"--video-column is missing from CSV: {video_column}")
    detected_column = video_column or next((name for name in VIDEO_COLUMNS if name in fields), None)
    needs_discovery = detected_column is None or any(
        not row.get(detected_column, "").strip() for row in rows
    )
    videos_by_index = index_videos(video_root) if needs_discovery else {}

    output_fields = list(fields)
    for field in HUMAN_COLUMNS:
        if field not in output_fields:
            output_fields.append(field)

    episodes = []
    keys: set[str] = set()
    for row_number, source_row in enumerate(rows, start=2):
        key = episode_key(source_row, row_number)
        if key in keys:
            raise ValueError(f"Duplicate episode identity: {key}")
        keys.add(key)
        row = {field: source_row.get(field, "") for field in output_fields}
        explicit = row.get(detected_column, "").strip() if detected_column else ""
        relative_videos = (
            resolve_explicit_videos(explicit, video_root)
            if explicit
            else discover_videos(row, videos_by_index, row_number)
        )
        if detected_column:
            row[detected_column] = "|".join(relative_videos)
        episodes.append(
            {
                "key": key,
                "row": row,
                "videos": [
                    {
                        "label": video_label(path, index),
                        "url": "/media/" + quote(path, safe="/"),
                    }
                    for index, path in enumerate(relative_videos)
                ],
            }
        )

    dataset_id = hashlib.sha256(review_csv.read_bytes()).hexdigest()[:16]
    payload = {
        "schema_version": "re-vla-local-rollout-viewer-v1",
        "dataset_id": dataset_id,
        "source_csv_name": review_csv.name,
        "fields": output_fields,
        "episodes": episodes,
    }
    output_dir.mkdir(parents=True)
    assets = Path(__file__).resolve().parent
    for asset, target in ASSET_TARGETS.items():
        shutil.copy2(assets / asset, output_dir / target)
    (output_dir / "episodes.json").write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return {
        "viewer_dir": str(output_dir),
        "video_root": str(video_root),
        "episodes": len(episodes),
        "videos": sum(len(item["videos"]) for item in episodes),
        "dataset_id": dataset_id,
        "explicit_video_column": detected_column,
        "videos_copied": 0,
        "forbidden_columns": [],
    }


def parse_byte_range(header: str | None, size: int) -> tuple[int, int] | None:
    if not header:
        return None
    match = RANGE_PATTERN.fullmatch(header.strip())
    if not match:
        raise ValueError("Invalid Range header")
    start_text, end_text = match.groups()
    if not start_text:
        length = int(end_text)
        if length <= 0:
            raise ValueError("Invalid suffix range")
        return max(0, size - length), size - 1
    start = int(start_text)
    end = int(end_text) if end_text else size - 1
    if start >= size or end < start:
        raise ValueError("Unsatisfiable range")
    return start, min(end, size - 1)


class ViewerRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, viewer_dir: Path, video_root: Path, **kwargs: Any):
        self.video_root = video_root
        super().__init__(*args, directory=str(viewer_dir), **kwargs)

    def do_HEAD(self) -> None:
        if urlsplit(self.path).path.startswith("/media/"):
            self.serve_media(send_body=False)
        else:
            super().do_HEAD()

    def do_GET(self) -> None:
        if urlsplit(self.path).path.startswith("/media/"):
            self.serve_media(send_body=True)
        else:
            super().do_GET()

    def serve_media(self, send_body: bool) -> None:
        encoded = urlsplit(self.path).path.removeprefix("/media/")
        relative = PurePosixPath(unquote(encoded))
        if relative.is_absolute() or ".." in relative.parts:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        media = (self.video_root / Path(*relative.parts)).resolve()
        if not media.is_relative_to(self.video_root) or not media.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        size = media.stat().st_size
        if size <= 0:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            selected = parse_byte_range(self.headers.get("Range"), size)
        except (ValueError, OverflowError):
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        start, end = selected or (0, size - 1)
        self.send_response(HTTPStatus.PARTIAL_CONTENT if selected else HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(media.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-store")
        if selected:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not send_body:
            return
        remaining = end - start + 1
        with media.open("rb") as handle:
            handle.seek(start)
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def serve(viewer_dir: Path, video_root: Path, host: str, port: int, open_browser: bool) -> None:
    viewer_dir = viewer_dir.resolve(strict=True)
    video_root = video_root.resolve(strict=True)
    handler = partial(
        ViewerRequestHandler,
        viewer_dir=viewer_dir,
        video_root=video_root,
    )
    with ThreadingHTTPServer((host, port), handler) as server:
        url = f"http://{host}:{server.server_port}/"
        print(json.dumps({"viewer_url": url, "videos_copied": 0}, ensure_ascii=False))
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="Build viewer metadata and static assets")
    build.add_argument("--review-csv", type=Path, required=True)
    build.add_argument("--video-root", type=Path, required=True)
    build.add_argument("--output-dir", type=Path)
    build.add_argument("--video-column")
    build.add_argument("--serve", action="store_true")
    build.add_argument("--host", default="127.0.0.1")
    build.add_argument("--port", type=int, default=8765)
    build.add_argument("--open", action="store_true")

    run = commands.add_parser("serve", help="Serve an already-built viewer")
    run.add_argument("--viewer-dir", type=Path, required=True)
    run.add_argument("--video-root", type=Path, required=True)
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8765)
    run.add_argument("--open", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "build":
        output = args.output_dir or args.review_csv.with_name(f"{args.review_csv.stem}_viewer")
        result = build_viewer(args.review_csv, args.video_root, output, args.video_column)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.serve:
            serve(output, args.video_root, args.host, args.port, args.open)
    else:
        serve(args.viewer_dir, args.video_root, args.host, args.port, args.open)


if __name__ == "__main__":
    main()
