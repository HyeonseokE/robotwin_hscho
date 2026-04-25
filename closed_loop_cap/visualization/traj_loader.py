"""Load and parse traj.pkl files from closed-loop CaP rollouts."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def load_traj(path: str | Path) -> dict:
    """Load a traj.pkl file.

    Returns dict with keys ``left_joint_path``, ``right_joint_path`` —
    each a list of plan-result dicts ``{status, position, velocity}``.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def concat_positions(segments: list) -> np.ndarray:
    """Concatenate ``position`` arrays from plan-result dicts into (N, 6).

    Only includes segments whose ``status`` is ``"Success"``.
    Returns an empty ``(0, 6)`` array when no successful segments exist.
    """
    arrays = []
    for seg in segments:
        if isinstance(seg, dict) and seg.get("status") == "Success":
            arrays.append(np.asarray(seg["position"]))
    if not arrays:
        return np.empty((0, 6))
    return np.concatenate(arrays, axis=0)
