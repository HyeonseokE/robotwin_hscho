"""Native (SAPIEN) ↔ ADC-normalized unit conversion for dataset logging.

RoboTwin native units:
    - arm joints:  radians, URDF-defined limits per joint
    - gripper:     0~1 normalized (0 closed, 1 open)

ADC normalized units (LeRobot dataset schema):
    - arm joints:  -100 ~ +100 (linear map of [lower, upper] URDF limits)
    - gripper:     0 ~ 100 (linear scale of 0~1 native)

See docs/replay_dataset_logging.md §3.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ArmLimits:
    """Per-joint URDF limits and precomputed mid/half for fast conversion."""

    lower: np.ndarray  # shape (n_arm_joints,)
    upper: np.ndarray
    mid: np.ndarray
    half: np.ndarray

    @classmethod
    def from_sapien(cls, qlimits: np.ndarray) -> "ArmLimits":
        """`qlimits` is the raw (n_joints, 2) array from SAPIEN's get_qlimits."""
        qlimits = np.asarray(qlimits, dtype=np.float64)
        lower = qlimits[:, 0].copy()
        upper = qlimits[:, 1].copy()
        # Guard against degenerate joints (fixed joints or identical bounds).
        half = (upper - lower) / 2.0
        half[half == 0.0] = 1.0  # avoid div-by-zero; converted value will be 0.
        mid = (upper + lower) / 2.0
        return cls(lower=lower, upper=upper, mid=mid, half=half)


class LimitsCache:
    """Lazily captures URDF limits from a live task_env for both arms.

    RoboTwin aloha-agilex: 6 arm joints + 1 gripper per side. We capture only
    arm joint limits (the gripper is already 0~1 normalized and needs no
    URDF lookup).
    """

    def __init__(self) -> None:
        self.left: ArmLimits | None = None
        self.right: ArmLimits | None = None
        self.left_arm_dim: int = 0
        self.right_arm_dim: int = 0

    def capture(self, task_env: Any) -> None:
        left_entity = task_env.robot.left_entity
        right_entity = task_env.robot.right_entity
        left_all = np.asarray(left_entity.get_qlimits(), dtype=np.float64)
        right_all = np.asarray(right_entity.get_qlimits(), dtype=np.float64)
        # Only the active-joint indices that back get_left_arm_jointState() are
        # relevant. RoboTwin's robot.left_arm_joints list gives us the active
        # joint objects; their order into get_active_joints() is the qpos order.
        def _active_arm_slice(entity: Any, arm_joints: list) -> np.ndarray:
            active = entity.get_active_joints()
            idxs = [active.index(j) for j in arm_joints]
            full = np.asarray(entity.get_qlimits(), dtype=np.float64)
            return full[idxs]

        left_arm_limits = _active_arm_slice(left_entity, task_env.robot.left_arm_joints)
        right_arm_limits = _active_arm_slice(right_entity, task_env.robot.right_arm_joints)
        self.left = ArmLimits.from_sapien(left_arm_limits)
        self.right = ArmLimits.from_sapien(right_arm_limits)
        self.left_arm_dim = len(self.left.mid)
        self.right_arm_dim = len(self.right.mid)

    def to_dict(self) -> dict:
        """Serializable representation for meta/info.json custom field."""
        assert self.left is not None and self.right is not None, "capture() first"
        return {
            "left_arm_lower_rad": self.left.lower.tolist(),
            "left_arm_upper_rad": self.left.upper.tolist(),
            "right_arm_lower_rad": self.right.lower.tolist(),
            "right_arm_upper_rad": self.right.upper.tolist(),
            "gripper_range_native": [0.0, 1.0],
            "arm_normalized_range": [-100.0, 100.0],
            "gripper_normalized_range": [0.0, 100.0],
        }


def to_normalized(
    left_arm_rad: np.ndarray,
    left_gripper_01: float,
    right_arm_rad: np.ndarray,
    right_gripper_01: float,
    cache: LimitsCache,
) -> np.ndarray:
    """Pack (6L_arm + 1L_gripper + 6R_arm + 1R_gripper) into a single
    length-(left_arm_dim + 1 + right_arm_dim + 1) float32 vector in ADC units.
    """
    assert cache.left is not None and cache.right is not None
    la = np.clip(
        (np.asarray(left_arm_rad, dtype=np.float64) - cache.left.mid)
        / cache.left.half * 100.0,
        -100.0, 100.0,
    )
    ra = np.clip(
        (np.asarray(right_arm_rad, dtype=np.float64) - cache.right.mid)
        / cache.right.half * 100.0,
        -100.0, 100.0,
    )
    lg = np.clip(float(left_gripper_01) * 100.0, 0.0, 100.0)
    rg = np.clip(float(right_gripper_01) * 100.0, 0.0, 100.0)
    return np.concatenate([la, [lg], ra, [rg]]).astype(np.float32)


def to_native(
    normalized: np.ndarray,
    cache: LimitsCache,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Inverse of to_normalized. Returns (left_arm_rad, left_grip_01, right_arm_rad, right_grip_01)."""
    assert cache.left is not None and cache.right is not None
    normalized = np.asarray(normalized, dtype=np.float64)
    la_dim = cache.left_arm_dim
    ra_dim = cache.right_arm_dim
    la = normalized[:la_dim]
    lg = normalized[la_dim]
    ra = normalized[la_dim + 1 : la_dim + 1 + ra_dim]
    rg = normalized[la_dim + 1 + ra_dim]
    la_rad = la / 100.0 * cache.left.half + cache.left.mid
    ra_rad = ra / 100.0 * cache.right.half + cache.right.mid
    return la_rad, float(lg) / 100.0, ra_rad, float(rg) / 100.0
