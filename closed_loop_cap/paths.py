"""Session-aware output path helpers.

Every collection run writes under
``<output_dir>/datasets/<session_id>/<task>/...`` so that repeated runs for
the same ``(task, seed)`` never overwrite each other. Session ids default
to a wall-clock timestamp; callers may override via ``--session <name>``
to group trials under a named experiment (e.g. ``perturb_on``).

This module is the single source of truth for the on-disk layout. Any
runner or visualization tool that touches dataset artefacts MUST resolve
its paths through the helpers here.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path


def default_session_id() -> str:
    """Generate a timestamp-based session id (``YYYYMMDD_HHMMSS``)."""
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def task_root(output_dir: Path, session: str, task: str) -> Path:
    """``<output_dir>/datasets/<session>/<task>`` — per-task root."""
    return Path(output_dir) / "datasets" / session / task


def logs_root(output_dir: Path, session: str, task: str) -> Path:
    """Task logs root (per-trial rollout artefacts)."""
    return task_root(output_dir, session, task) / "logs"


def seed_logs_dir(
    output_dir: Path, session: str, task: str, seed: int,
) -> Path:
    """Per-seed directory under logs."""
    return logs_root(output_dir, session, task) / f"seed_{seed}"


def trial_dir(
    output_dir: Path, session: str, task: str, seed: int, trial: int,
) -> Path:
    """Single-trial directory (``trial_KKK`` zero-padded to 3 digits)."""
    return seed_logs_dir(output_dir, session, task, seed) / f"trial_{trial:03d}"


def recorded_data_dir(
    output_dir: Path, session: str, task: str,
) -> Path:
    """LeRobot v3.0 dataset root (shared across seeds/trials in a session)."""
    return task_root(output_dir, session, task) / "recorded_data"


def viz_dir(
    output_dir: Path, session: str, task: str,
) -> Path:
    """Visualization output root."""
    return task_root(output_dir, session, task) / "viz"


def save_subdir(session: str, task: str, seed: int, trial: int) -> str:
    """Relative save-subdir string used by RoboTwin's ``save_traj_data``.

    This mirrors :func:`trial_dir` but expressed as a forward-slash string
    that RoboTwin accepts as a config value.
    """
    return f"datasets/{session}/{task}/logs/seed_{seed}/trial_{trial:03d}"
