"""Privileged GT feature collection (sim-only).

For a given task_env, pulls per-actor world pose as [x, y, z, qx, qy, qz, qw].
Enabled via yaml `logging.privileged_features: true`; otherwise the recorder
never emits these keys.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def collect_actor_poses(task_env: Any, actor_names: list[str]) -> dict[str, np.ndarray]:
    """Return {actor_name: pose7} for actors that resolve on the live env.

    Silently skips actors that can't be looked up (task may register different
    attribute names across embodiments)."""
    out: dict[str, np.ndarray] = {}
    for name in actor_names:
        obj = getattr(task_env, name, None)
        if obj is None:
            continue
        try:
            pose = obj.get_pose()
            p = pose.p
            q = pose.q  # SAPIEN quaternion is [w, x, y, z]
            pose7 = np.array(
                [p[0], p[1], p[2], q[1], q[2], q[3], q[0]],
                dtype=np.float32,
            )
            out[name] = pose7
        except Exception:  # noqa: BLE001 — don't break logging for one missing actor
            continue
    return out
