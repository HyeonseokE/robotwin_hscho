"""Subgoal-level stochastic perturbation for trajectory diversity.

Perturbs the target EE position of transit (non-interaction) subgoals by
sampling from a truncated isotropic 3D Gaussian.

**Transit vs. interaction is decided at the Action level, not by
subtask.skill_type.** Each skill function (grasp_actor / place_actor /
move_by_displacement / move_to_pose / back_to_origin) tags the Actions it
returns with ``transit=True|False``. ``Base_Task.move()`` reads that flag
and applies the sampled offset only to transit Actions. This module just
produces the offset; it does not decide *which* Action receives it.

See docs/stochastic_perturbation.md for motivation and parameter choices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SubgoalPerturbationConfig:
    """Runtime-immutable perturbation parameters."""

    enabled: bool = False
    sigma: float = 0.05        # metres; isotropic std-dev
    clip_factor: float = 2.0   # reject samples beyond clip_factor * sigma

    @property
    def clip_radius(self) -> float:
        return self.clip_factor * self.sigma


class SubgoalPerturbation:
    """Samples a single 3D offset per transit subgoal.

    Usage::

        pert = SubgoalPerturbation(cfg)

        # Before executing a subtask snippet:
        offset = pert.sample(skill_type, rng)   # None for interaction skills
        task_env._subgoal_offset = offset        # move() reads this

        # After execution:
        task_env._subgoal_offset = None
    """

    def __init__(self, config: SubgoalPerturbationConfig) -> None:
        self.cfg = config

    def sample(
        self,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray | None:
        """Sample a 3-element xyz offset, or ``None`` when perturbation is off.

        The caller stores the offset on the task env; ``Base_Task.move()``
        applies it only to Actions whose ``args["transit"]`` is ``True``. So
        offering an offset for every subtask is safe: interaction Actions
        (grasp / place contact) are protected by their own tag.
        """
        if not self.cfg.enabled:
            return None
        rng = rng or np.random.default_rng()
        return _sample_truncated_gaussian_3d(
            sigma=self.cfg.sigma,
            clip_radius=self.cfg.clip_radius,
            rng=rng,
        )


def _sample_truncated_gaussian_3d(
    sigma: float,
    clip_radius: float,
    rng: np.random.Generator,
    max_attempts: int = 100,
) -> np.ndarray:
    """Sample from N(0, sigma^2 I) with rejection beyond *clip_radius*."""
    for _ in range(max_attempts):
        sample = rng.normal(0.0, sigma, size=3)
        if np.linalg.norm(sample) <= clip_radius:
            return sample
    # Fallback: project onto the clip sphere (should virtually never happen
    # because P(||x|| > 2*sigma) ≈ 1.2% for 3D Gaussian — 100 attempts is
    # more than sufficient).
    sample = rng.normal(0.0, sigma, size=3)
    norm = np.linalg.norm(sample)
    if norm > clip_radius:
        sample = sample / norm * clip_radius
    return sample
