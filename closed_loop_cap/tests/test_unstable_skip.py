"""Unit tests for unstable-seed skip path in run_closed_loop.run_episode."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from closed_loop_cap.run_closed_loop import _is_unstable_seed_error  # noqa: E402


class _UnStableError(Exception):
    """Synthetic stand-in for envs.utils.create_actor.UnStableError."""

    __name__ = "UnStableError"


@pytest.mark.unit
def test_is_unstable_matches_class_name() -> None:
    ExcClass = type("UnStableError", (Exception,), {})
    exc = ExcClass("benign message")
    assert _is_unstable_seed_error(exc)


@pytest.mark.unit
def test_is_unstable_matches_message() -> None:
    exc = RuntimeError("Objects is unstable in seed(3)")
    assert _is_unstable_seed_error(exc)


@pytest.mark.unit
def test_is_unstable_rejects_unrelated_error() -> None:
    assert not _is_unstable_seed_error(ValueError("bad input"))
    assert not _is_unstable_seed_error(RuntimeError("motion planner failed"))


@pytest.mark.unit
def test_is_unstable_needs_both_words() -> None:
    # 'unstable' alone (e.g., 'unstable api') is not enough — we require seed context.
    assert not _is_unstable_seed_error(RuntimeError("unstable connection"))


@pytest.mark.unit
def test_run_episode_returns_unstable_report(tmp_path) -> None:
    """make_env raising UnStableError → report marks unstable_seed, no crash."""
    from closed_loop_cap import run_closed_loop as rcl

    def _raise_unstable(*_args, **_kwargs):
        raise type("UnStableError", (Exception,), {})("Objects is unstable in seed(0)")

    # Minimal config — run_episode only looks up keys that gate the skip path.
    config = {"output_dir": str(tmp_path)}

    with patch.object(rcl, "make_env", side_effect=_raise_unstable), \
         patch.object(rcl, "load_task_meta", return_value=_FakeMeta()):
        report = rcl.run_episode("fake_task", seed=0, config=config, client=None)

    assert report.success is False
    assert report.abort_reason == "unstable_seed"
    assert report.num_subtasks == 0
    # report.json must be written even on skip
    assert (tmp_path / "fake_task" / "seed_0" / "report.json").exists()


class _FakeMeta:
    task_name = "fake_task"
    description = "fake"
    actor_names: tuple[str, ...] = ()


@pytest.mark.unit
def test_run_episode_reraises_other_exceptions(tmp_path) -> None:
    """Non-unstable exceptions should still propagate — no silent swallow."""
    from closed_loop_cap import run_closed_loop as rcl

    def _raise_other(*_args, **_kwargs):
        raise ValueError("unrelated bug")

    config = {"output_dir": str(tmp_path)}
    with patch.object(rcl, "make_env", side_effect=_raise_other), \
         patch.object(rcl, "load_task_meta", return_value=_FakeMeta()):
        with pytest.raises(ValueError, match="unrelated bug"):
            rcl.run_episode("fake_task", seed=0, config=config, client=None)
