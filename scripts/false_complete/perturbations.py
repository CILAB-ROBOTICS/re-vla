"""Modular perturbations for the False Complete experiment.

Perturbation.apply(libero_env, context) mutates the *simulator* directly and returns a
dict describing what happened (for events.json). `libero_env` is the raw LIBERO
`ControlEnv`/`OffScreenRenderEnv` (`vec_env.envs[0]._env`), not the lerobot gym wrapper —
low-level sim access lives at that layer:
  - `libero_env.env.objects_dict[name]` -> MujocoObject (has `.joints[0]`, the free-joint name)
  - `libero_env.env.obj_body_id[name]`  -> MuJoCo body id, for `sim.data.body_xpos`
  - `libero_env.env.sim`                -> MjSim (`.model.get_joint_qpos_addr`, `.data.qpos`, `.forward()`)
(see env_wrapper.py / bddl_base_domain.py in the LIBERO repo — these are the same
accessors LIBERO's own object-pose sensors and robosuite's `_check_grasp` use, reused
here rather than re-derived.)

Object perturbations write directly into `sim.data.qpos` at the target object's free
joint. robosuite doesn't expose a "teleport object" / "release from gripper" action, so
this is the standard way (also how robosuite's own placement_initializer positions
objects at reset) — `sim.forward()` afterward propagates the write into `body_xpos`,
contacts, and rendered observations before the next `env.step()`.
"""

from __future__ import annotations

import numpy as np


class Perturbation:
    name = "base"

    def apply(self, libero_env, context: dict) -> dict:
        """Mutate the simulator. Returns a dict merged into events.json."""
        raise NotImplementedError

    def modify_observation(self, raw_obs: dict, context: dict) -> dict:
        """Optional hook for perturbations that alter what the *policy* perceives
        rather than the simulator state (e.g. a proprioception override). Called every
        step after this perturbation has fired. Default: no-op passthrough."""
        return raw_obs


def _get_object_qpos_addr(libero_env, obj_name: str) -> tuple[int, int]:
    joint_name = libero_env.env.objects_dict[obj_name].joints[0]
    return libero_env.env.sim.model.get_joint_qpos_addr(joint_name)


def get_object_pose(libero_env, obj_name: str) -> np.ndarray:
    """[x, y, z, qw, qx, qy, qz] — free-joint qpos layout (MuJoCo quat convention)."""
    start, end = _get_object_qpos_addr(libero_env, obj_name)
    return np.array(libero_env.env.sim.data.qpos[start:end], dtype=np.float64)


def set_object_pose(libero_env, obj_name: str, pose: np.ndarray) -> None:
    start, end = _get_object_qpos_addr(libero_env, obj_name)
    libero_env.env.sim.data.qpos[start:end] = pose
    libero_env.env.sim.forward()


class ObjectDropPerturbation(Perturbation):
    """Simulates the target object slipping out of the gripper mid-grasp.

    Nudges the object down by a small offset — just enough to break contact with the
    gripper fingers — rather than teleporting it straight to table height. Grasping in
    robosuite/MuJoCo is friction/contact-based, not a hard constraint, so once contact
    breaks, gravity pulls the object the rest of the way down naturally over the
    following env.step() calls, giving a physically continuous fall instead of a
    one-frame teleport (which looked like the object instantly vanishing/reappearing).
    """

    name = "object_drop"

    def __init__(self, drop_offset: float = 0.03, target_object: str | None = None):
        # 3cm is comfortably more than typical fingertip/object interpenetration in
        # LIBERO's grasps, so contact reliably breaks; tune with --drop-offset if a
        # specific object/gripper combination needs more.
        self.drop_offset = drop_offset
        self.target_object = target_object

    def apply(self, libero_env, context: dict) -> dict:
        obj_name = self.target_object or context["target_object"]
        pose_before = get_object_pose(libero_env, obj_name)
        pose_after = pose_before.copy()
        pose_after[2] -= self.drop_offset
        set_object_pose(libero_env, obj_name, pose_after)
        return {
            "perturbation_type": self.name,
            "target_object": obj_name,
            "drop_offset": self.drop_offset,
            "pose_before": pose_before.tolist(),
            "pose_after": pose_after.tolist(),
        }


class ObjectRelocationPerturbation(Perturbation):
    """Translates the target object by a fixed offset (config: --translation dx dy dz)."""

    name = "object_relocation"

    def __init__(self, translation=(0.08, 0.0, 0.0), target_object: str | None = None):
        self.translation = np.asarray(translation, dtype=np.float64)
        self.target_object = target_object

    def apply(self, libero_env, context: dict) -> dict:
        obj_name = self.target_object or context["target_object"]
        pose_before = get_object_pose(libero_env, obj_name)
        pose_after = pose_before.copy()
        pose_after[:3] += self.translation
        set_object_pose(libero_env, obj_name, pose_after)
        return {
            "perturbation_type": self.name,
            "target_object": obj_name,
            "translation": self.translation.tolist(),
            "pose_before": pose_before.tolist(),
            "pose_after": pose_after.tolist(),
        }


PERTURBATIONS = {
    "object_drop": ObjectDropPerturbation,
    "object_relocation": ObjectRelocationPerturbation,
}
