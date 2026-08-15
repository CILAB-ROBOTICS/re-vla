"""Control-vs-perturbed action similarity metrics and a first-pass behavior classifier.

Important framing (per the experiment's actual research question): a perturbed episode
failing is *not* itself evidence of False Complete. The thing we're after is whether the
*policy's output* changed once the *environment* actually diverged — so every function
here operates on paired (control, perturbed) action sequences from the same task/seed/
initial state, not on either rollout in isolation.
"""

from __future__ import annotations

import numpy as np


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def per_step_action_similarity(control_actions: np.ndarray, perturbed_actions: np.ndarray) -> dict:
    """control_actions/perturbed_actions: (T, action_dim) *executed* actions, aligned by
    timestep index. Compares up to min(len(control), len(perturbed))."""
    t = min(len(control_actions), len(perturbed_actions))
    l2 = np.array([l2_distance(control_actions[i], perturbed_actions[i]) for i in range(t)])
    cos = np.array([cosine_similarity(control_actions[i], perturbed_actions[i]) for i in range(t)])
    return {"l2": l2, "cosine": cos}


def chunk_similarity(control_chunk: np.ndarray, perturbed_chunk: np.ndarray) -> dict:
    """Compares two full raw action chunks (n_action_steps, action_dim): whole-chunk
    L2/cosine (flattened) plus a per-within-chunk-step cosine breakdown."""
    t = min(len(control_chunk), len(perturbed_chunk))
    return {
        "l2": l2_distance(control_chunk.ravel(), perturbed_chunk.ravel()),
        "cosine": cosine_similarity(control_chunk.ravel(), perturbed_chunk.ravel()),
        "per_step_cosine": np.array(
            [cosine_similarity(control_chunk[i], perturbed_chunk[i]) for i in range(t)]
        ),
    }


def post_perturbation_window_similarity(
    control_actions: np.ndarray,
    perturbed_actions: np.ndarray,
    trigger_step: int,
    window: int = 20,
) -> dict:
    """Mean similarity over executed actions in [trigger_step, trigger_step + window) —
    the window this experiment cares about most: does the *action* change once the
    environment has actually diverged from the control run?"""
    end = min(trigger_step + window, len(control_actions), len(perturbed_actions))
    if end <= trigger_step:
        return {"l2": None, "cosine": None, "n_steps": 0}
    sims = per_step_action_similarity(control_actions[trigger_step:end], perturbed_actions[trigger_step:end])
    return {
        "l2": float(np.mean(sims["l2"])),
        "cosine": float(np.mean(sims["cosine"])),
        "n_steps": end - trigger_step,
    }


def classify_behavior(
    post_perturb_cosine_similarity: float | None,
    perturbed_success: bool,
    continue_threshold: float = 0.85,
    recovery_threshold: float = 0.6,
) -> str:
    """Coarse, threshold-based first pass at CONTINUE / RECOVERY /
    SUCCESS_DESPITE_PERTURBATION / unclassified, from post-perturbation-window cosine
    similarity between control and perturbed executed actions.

    This is explicitly a starting point for human review, not a validated detector — the
    thresholds are unvalidated defaults, meant to be tuned once real data + human labels
    exist (see run_false_complete_eval.py's summary output and README.md's analysis
    notes). High similarity despite environmental failure is the False Complete
    candidate signal; low similarity suggests the policy noticed and reacted (RECOVERY),
    though "reacted" here isn't verified to mean "recovered *correctly*" — that still
    needs human/task-level judgment.
    """
    if perturbed_success:
        return "SUCCESS_DESPITE_PERTURBATION"
    if post_perturb_cosine_similarity is None:
        return "unclassified"
    if post_perturb_cosine_similarity >= continue_threshold:
        return "CONTINUE"
    if post_perturb_cosine_similarity < recovery_threshold:
        return "RECOVERY"
    return "unclassified"
