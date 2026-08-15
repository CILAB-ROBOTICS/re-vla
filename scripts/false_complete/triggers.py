"""Perturbation triggers: decide *when* (which timestep) to fire a perturbation.

Each Trigger is checked once per step against the raw LIBERO env (the low-level
`ControlEnv`/`OffScreenRenderEnv` accessible as `vec_env.envs[0]._env`, not the lerobot
gym wrapper) — grasp/lift detection needs `_check_grasp`, `objects_dict`, and
`sim.data.body_xpos`, which only exist at that layer (see env_wrapper.py /
bddl_base_domain.py in the LIBERO repo).

Triggers are evaluated on *both* control and perturbed rollouts (harmless/cheap on
control — nothing gets applied there) so events.json can record e.g. grasp_timestep on
either regardless of whether a perturbation actually fired.
"""

from __future__ import annotations

import numpy as np


class Trigger:
    name = "base"

    def reset(self) -> None:
        """Called once at the start of each episode."""

    def check(self, step_idx: int, libero_env, context: dict) -> bool:
        """Return True the step this trigger should fire (only ever once per episode —
        callers are expected to stop calling check() after the first True)."""
        raise NotImplementedError


class TimestepTrigger(Trigger):
    name = "timestep"

    def __init__(self, trigger_step: int):
        self.trigger_step = trigger_step

    def check(self, step_idx: int, libero_env, context: dict) -> bool:
        return step_idx == self.trigger_step


class GraspTrigger(Trigger):
    """Fires the step the target object is first detected as grasped, via robosuite's
    own `_check_grasp` (contact between both gripper fingerpads and the object) —
    reused as-is rather than reimplemented (manipulation_env.py:_check_grasp).
    """

    name = "grasp"

    def __init__(self, target_object: str | None = None):
        self.target_object = target_object
        self._fired = False

    def reset(self) -> None:
        self._fired = False

    def check(self, step_idx: int, libero_env, context: dict) -> bool:
        if self._fired:
            return False
        obj_name = self.target_object or context["target_object"]
        grasped = libero_env.env._check_grasp(
            gripper=libero_env.robots[0].gripper,
            object_geoms=libero_env.env.objects_dict[obj_name],
        )
        if grasped:
            self._fired = True
            return True
        return False


class LiftTrigger(Trigger):
    """Fires the first step the target object's world-frame z rises more than
    `height_threshold` above where it was at the start of the episode."""

    name = "lift"

    def __init__(self, target_object: str | None = None, height_threshold: float = 0.04):
        self.target_object = target_object
        self.height_threshold = height_threshold
        self._initial_z: float | None = None
        self._fired = False

    def reset(self) -> None:
        self._initial_z = None
        self._fired = False

    def check(self, step_idx: int, libero_env, context: dict) -> bool:
        if self._fired:
            return False
        obj_name = self.target_object or context["target_object"]
        z = float(np.asarray(libero_env.env.sim.data.body_xpos[libero_env.env.obj_body_id[obj_name]])[2])
        if self._initial_z is None:
            self._initial_z = z
            return False
        if z > self._initial_z + self.height_threshold:
            self._fired = True
            return True
        return False


TRIGGERS = {
    "timestep": TimestepTrigger,
    "grasp": GraspTrigger,
    "lift": LiftTrigger,
}
