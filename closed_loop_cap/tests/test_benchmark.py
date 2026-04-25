"""Unit tests for benchmark aggregation (build_summary, format_markdown).

These are pure functions over dicts — no SAPIEN, no VLM, no disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from closed_loop_cap.run_benchmark import build_summary, format_markdown  # noqa: E402


def _ep(seed: int, success: bool, abort: str | None = None, signals: list[str] | None = None,
        num_subtasks: int = 2, duration: float = 30.0) -> dict:
    return {
        "task": "t",
        "seed": seed,
        "success": success,
        "abort_reason": abort,
        "num_subtasks": num_subtasks,
        "duration_s": duration,
        "subtasks": [
            {
                "attempts": [
                    {"ok": False, "signal_id": sid, "detail": "", "code": ""}
                    for sid in (signals or [])
                ],
                "final_ok": success,
            }
        ] if signals else [],
    }


@pytest.mark.unit
def test_build_summary_success_rate_excludes_unstable() -> None:
    reports = {
        "beat_block_hammer": [
            _ep(0, success=True),
            _ep(1, success=False, abort="L2_max_subtask_retries: subtask_id=2"),
            _ep(2, success=False, abort="unstable_seed"),   # skip
            _ep(3, success=True),
        ]
    }
    summary = build_summary(reports)
    row = summary["rows"][0]
    assert row["total"] == 4
    assert row["unstable_skipped"] == 1
    assert row["valid"] == 3
    assert row["success"] == 2
    assert row["success_rate"] == pytest.approx(2 / 3)


@pytest.mark.unit
def test_build_summary_counts_failure_signals() -> None:
    reports = {
        "t": [
            _ep(0, success=False, signals=["L3-P1", "L3-SKL-GRASP", "L3-SKL-GRASP"]),
            _ep(1, success=False, signals=["L2-S6"]),
        ]
    }
    summary = build_summary(reports)
    sigmap = dict(summary["rows"][0]["top_failure_signals"])
    assert sigmap["L3-SKL-GRASP"] == 2
    assert sigmap["L3-P1"] == 1
    assert sigmap["L2-S6"] == 1


@pytest.mark.unit
def test_build_summary_abort_reasons_normalized() -> None:
    reports = {
        "t": [
            _ep(0, success=False, abort="L1-S7_cascade: 2 consecutive (last subtask_id=1)"),
            _ep(1, success=False, abort="L1-S7_cascade: 2 consecutive (last subtask_id=2)"),
            _ep(2, success=False, abort="unstable_seed"),
        ]
    }
    summary = build_summary(reports)
    aborts = dict(summary["rows"][0]["top_abort_reasons"])
    # Different subtask_ids should be grouped under the prefix before ':'.
    assert aborts.get("L1-S7_cascade") == 2
    assert aborts.get("unstable_seed") == 1


@pytest.mark.unit
def test_build_summary_grand_total() -> None:
    reports = {
        "task_a": [_ep(0, success=True), _ep(1, success=False)],
        "task_b": [_ep(0, success=True), _ep(1, success=False, abort="unstable_seed")],
    }
    summary = build_summary(reports)
    g = summary["grand_total"]
    assert g["total"] == 4
    assert g["unstable_skipped"] == 1
    assert g["valid"] == 3
    assert g["success"] == 2
    assert g["success_rate"] == pytest.approx(2 / 3)


@pytest.mark.unit
def test_format_markdown_has_required_sections() -> None:
    reports = {"t": [_ep(0, success=True)]}
    md = format_markdown(build_summary(reports))
    assert "# Benchmark Summary" in md
    assert "## Per task" in md
    assert "## Grand total" in md
    assert "| Task |" in md


@pytest.mark.unit
def test_build_summary_handles_empty_reports() -> None:
    summary = build_summary({"t": []})
    row = summary["rows"][0]
    assert row["total"] == 0
    assert row["valid"] == 0
    assert row["success_rate"] == 0.0
    # No division-by-zero
    assert row["avg_duration_s"] == 0.0
    assert row["avg_subtasks"] == 0.0
