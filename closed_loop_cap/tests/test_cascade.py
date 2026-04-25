"""Unit tests for cascade detection (L1-S7) in run_episode.

We drive the subtask loop by mocking each moving piece at the seam:
    - make_env / capture_rgb / snapshot_robot_state / close_env / is_task_success
    - plan_subtasks (returns a scripted plan)
    - generate_subtask_code (returns a fixed snippet)
    - execute_snippet (returns an ExecResult whose outcome is None for 'static-blocked')
    - judge_after_exec (returns whatever the test wants)

This isolates cascade bookkeeping without touching SAPIEN or the VLM.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from closed_loop_cap.executor.executor import ExecResult  # noqa: E402
from closed_loop_cap.executor.judger import JudgeResult  # noqa: E402
from closed_loop_cap.executor.sandbox import StaticReport  # noqa: E402
from closed_loop_cap.vlm.schema import PlannerResponse, SubtaskSpec  # noqa: E402


def _mk_subtask(i: int) -> SubtaskSpec:
    return SubtaskSpec(
        id=i,
        instruction=f"step {i}",
        skill_type="grasp",
        target_actor="self.hammer",
        arm_tag="left",
        success_hint="",
    )


def _mk_handle() -> SimpleNamespace:
    te = SimpleNamespace(save_traj_data=lambda *_a, **_k: None)
    return SimpleNamespace(
        task_env=te, task_name="fake", seed=0, config={}, recorder=None,
    )


class _FakeMeta:
    task_name = "fake"
    description = "fake"
    actor_names: tuple[str, ...] = ("hammer",)
    actor_details: dict = {}


def _patch_common(plan_subtasks_n: int, judge_results):
    """Context-managers list — caller uses contextlib.ExitStack."""
    from closed_loop_cap import run_closed_loop as rcl
    import numpy as np

    dummy_rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    dummy_state = MagicMock()
    plan = PlannerResponse(subtasks=tuple(_mk_subtask(i + 1) for i in range(plan_subtasks_n)))
    exec_result_ok = ExecResult(
        code="self.move(...)",
        static_report=StaticReport(True, "", ""),
        outcome=MagicMock(ok=True, exception=None, duration_s=0.1, timed_out=False),
    )

    judge_iter = iter(judge_results)

    return [
        patch.object(rcl, "make_env", return_value=_mk_handle()),
        patch.object(rcl, "capture_rgb", return_value=dummy_rgb),
        patch.object(rcl, "snapshot_robot_state", return_value=dummy_state),
        patch.object(rcl, "close_env"),
        patch.object(rcl, "is_task_success", return_value=False),
        patch.object(rcl, "plan_subtasks", return_value=plan),
        patch.object(rcl, "generate_subtask_code", return_value="self.move(...)"),
        patch.object(rcl, "execute_snippet", return_value=exec_result_ok),
        patch.object(rcl, "judge_after_exec", side_effect=lambda *a, **k: next(judge_iter)),
        patch.object(rcl, "load_task_meta", return_value=_FakeMeta()),
        patch.object(rcl, "imageio"),
    ]


def _enter(patches):
    import contextlib

    stack = contextlib.ExitStack()
    for p in patches:
        stack.enter_context(p)
    return stack


FAIL_JUDGE = JudgeResult(
    ok=False, layer=2, signal_id="L2-S5",
    detail="fake runtime error", hint_for_codegen="retry",
)
OK_JUDGE = JudgeResult(ok=True, layer=0, signal_id="", detail="", hint_for_codegen=None)


@pytest.mark.unit
def test_cascade_triggers_after_2_consecutive_fails(tmp_path) -> None:
    """3 subtasks, first two fail → cascade abort before subtask 3 runs."""
    from closed_loop_cap import run_closed_loop as rcl

    # 3 subtasks × 3 retries each = up to 9 judge calls. Feed enough failures.
    judges = [FAIL_JUDGE] * 9
    config = {
        "output_dir": str(tmp_path),
        "max_subtask_retries": 3,
        "max_consecutive_subtask_fails": 2,
    }
    with _enter(_patch_common(3, judges)):
        report = rcl.run_episode("fake", seed=0, config=config, client=None)

    assert report.success is False
    assert report.abort_reason is not None
    assert "L1-S7_cascade" in report.abort_reason
    # We should have attempted subtask 1 and 2, NOT 3 (cascade aborted early).
    assert len(report.subtasks) == 2


@pytest.mark.unit
def test_single_failure_continues_to_next_subtask(tmp_path) -> None:
    """Subtask 1 fails, subtask 2 succeeds → no cascade, not an abort."""
    from closed_loop_cap import run_closed_loop as rcl

    # Subtask 1 fails 3× (exhausted), subtask 2 succeeds on first attempt.
    judges = [FAIL_JUDGE, FAIL_JUDGE, FAIL_JUDGE, OK_JUDGE]
    config = {
        "output_dir": str(tmp_path),
        "max_subtask_retries": 3,
        "max_consecutive_subtask_fails": 2,
    }
    with _enter(_patch_common(2, judges)):
        report = rcl.run_episode("fake", seed=0, config=config, client=None)

    # Not aborted on cascade — both subtasks attempted.
    assert report.abort_reason is None or "cascade" not in report.abort_reason
    assert len(report.subtasks) == 2
    assert report.subtasks[0].final_ok is False
    assert report.subtasks[1].final_ok is True


@pytest.mark.unit
def test_all_success_reaches_plan_end(tmp_path) -> None:
    from closed_loop_cap import run_closed_loop as rcl

    judges = [OK_JUDGE, OK_JUDGE]
    config = {
        "output_dir": str(tmp_path),
        "max_subtask_retries": 3,
        "max_consecutive_subtask_fails": 2,
    }
    with _enter(_patch_common(2, judges)):
        report = rcl.run_episode("fake", seed=0, config=config, client=None)

    assert len(report.subtasks) == 2
    assert all(st.final_ok for st in report.subtasks)
    assert report.abort_reason is None or "cascade" not in (report.abort_reason or "")


@pytest.mark.unit
def test_single_fail_then_ok_resets_consecutive_counter(tmp_path) -> None:
    """Fail 1 → OK 2 → Fail 3 → should NOT trigger cascade (counter reset after OK)."""
    from closed_loop_cap import run_closed_loop as rcl

    # Subtask 1: 3 fails (exhausted). Subtask 2: ok. Subtask 3: 3 fails (exhausted).
    # After subtask 2 OK, counter resets. After subtask 3, counter=1. No cascade.
    judges = [FAIL_JUDGE] * 3 + [OK_JUDGE] + [FAIL_JUDGE] * 3
    config = {
        "output_dir": str(tmp_path),
        "max_subtask_retries": 3,
        "max_consecutive_subtask_fails": 2,
    }
    with _enter(_patch_common(3, judges)):
        report = rcl.run_episode("fake", seed=0, config=config, client=None)

    assert len(report.subtasks) == 3, [s.final_ok for s in report.subtasks]
    # Last subtask failed but cascade should NOT trigger (counter was reset).
    assert report.abort_reason is None or "cascade" not in (report.abort_reason or "")
