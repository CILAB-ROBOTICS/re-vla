"""Per-episode data logging for the False Complete experiment.

Writes the directory structure requested for this experiment:
    episode_dir/
        metadata.json
        observations/{rgb.mp4, wrist_rgb.mp4, state.npy}
        actions/{policy_action_chunks.npz, executed_actions.npy}
        environment/{object_pose.npy, ee_pose.npy}
        events.json

RGB is written as mp4 via lerobot's own `write_video` (imageio-backed) rather than a new
video-writing path. `policy_action_chunks.npz` (not .npy) because chunks are only
generated every `n_action_steps`, not every step — it stores the chunk-generation step
indices alongside the stacked chunks so they can be re-aligned to the per-step arrays.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.utils.io_utils import write_video


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class EpisodeLogger:
    def __init__(self, episode_dir: str | Path, fps: int = 20):
        self.episode_dir = Path(episode_dir)
        self.fps = fps
        (self.episode_dir / "observations").mkdir(parents=True, exist_ok=True)
        (self.episode_dir / "actions").mkdir(parents=True, exist_ok=True)
        (self.episode_dir / "environment").mkdir(parents=True, exist_ok=True)

        self._rgb_frames: list[np.ndarray] = []
        self._wrist_frames: list[np.ndarray] = []
        self._states: list[np.ndarray] = []
        self._executed_actions: list[np.ndarray] = []
        self._object_poses: list[np.ndarray] = []
        self._ee_poses: list[np.ndarray] = []
        self._rewards: list[float] = []
        self._dones: list[bool] = []
        self._successes: list[bool] = []
        self._is_perturbed_flags: list[bool] = []
        self._chunk_step_indices: list[int] = []
        self._chunks: list[np.ndarray] = []

    def log_step(
        self,
        *,
        rgb: np.ndarray,
        wrist_rgb: np.ndarray | None,
        state: np.ndarray,
        executed_action: np.ndarray,
        object_pose: np.ndarray,
        ee_pose: np.ndarray,
        reward: float,
        done: bool,
        success: bool,
        is_perturbed: bool,
    ) -> None:
        self._rgb_frames.append(rgb)
        if wrist_rgb is not None:
            self._wrist_frames.append(wrist_rgb)
        self._states.append(np.asarray(state))
        self._executed_actions.append(np.asarray(executed_action))
        self._object_poses.append(np.asarray(object_pose))
        self._ee_poses.append(np.asarray(ee_pose))
        self._rewards.append(float(reward))
        self._dones.append(bool(done))
        self._successes.append(bool(success))
        self._is_perturbed_flags.append(bool(is_perturbed))

    def log_action_chunk(self, step_idx: int, chunk: np.ndarray) -> None:
        """chunk: raw policy output, shape (n_action_steps, action_dim) — logged every
        time a new chunk is generated (i.e. not every step)."""
        self._chunk_step_indices.append(step_idx)
        self._chunks.append(np.asarray(chunk))

    @property
    def executed_actions(self) -> np.ndarray:
        return np.stack(self._executed_actions) if self._executed_actions else np.empty((0,))

    @property
    def chunks(self) -> np.ndarray:
        return np.stack(self._chunks) if self._chunks else np.empty((0,))

    @property
    def chunk_step_indices(self) -> np.ndarray:
        return np.array(self._chunk_step_indices, dtype=np.int64)

    def save(self, metadata: dict[str, Any], events: dict[str, Any]) -> None:
        with open(self.episode_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=_json_default)

        if self._rgb_frames:
            write_video(str(self.episode_dir / "observations" / "rgb.mp4"), np.stack(self._rgb_frames), self.fps)
        if self._wrist_frames:
            write_video(
                str(self.episode_dir / "observations" / "wrist_rgb.mp4"), np.stack(self._wrist_frames), self.fps
            )
        np.save(self.episode_dir / "observations" / "state.npy", np.stack(self._states))

        np.save(self.episode_dir / "actions" / "executed_actions.npy", self.executed_actions)
        if self._chunks:
            np.savez(
                self.episode_dir / "actions" / "policy_action_chunks.npz",
                step_indices=self.chunk_step_indices,
                chunks=self.chunks,
            )

        np.save(self.episode_dir / "environment" / "object_pose.npy", np.stack(self._object_poses))
        np.save(self.episode_dir / "environment" / "ee_pose.npy", np.stack(self._ee_poses))

        events = dict(events)
        events.setdefault(
            "steps",
            {
                "reward": self._rewards,
                "done": self._dones,
                "success": self._successes,
                "is_perturbed": self._is_perturbed_flags,
            },
        )
        with open(self.episode_dir / "events.json", "w") as f:
            json.dump(events, f, indent=2, default=_json_default)
