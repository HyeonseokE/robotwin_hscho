"""Unit tests for closed_loop_cap.tools.find_gt_seeds.

These cover cache save/load, rebuild flag, num_seeds subsetting, and the
"no auto-collect" path. The actual `collect_gt_seeds()` call is stubbed
because it needs SAPIEN.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from closed_loop_cap.tools.find_gt_seeds import (  # noqa: E402
    GTSeedResult,
    gt_seeds_path,
    load_gt_seeds,
    load_or_find_gt_seeds,
    save_gt_seeds,
)


def _cfg(tmp_path: Path) -> dict:
    return {"output_dir": str(tmp_path)}


# -------------------- Path + save/load --------------------


@pytest.mark.unit
def test_gt_seeds_path(tmp_path) -> None:
    p = gt_seeds_path("my_task", _cfg(tmp_path))
    assert p == tmp_path / "_gt_seeds" / "my_task.json"


@pytest.mark.unit
def test_save_and_load_roundtrip(tmp_path) -> None:
    result = GTSeedResult(
        task="t", target_count=5, max_attempts=20,
        successful_seeds=[0, 3, 7, 12, 19],
        failed_seeds=[1, 2], unstable_seeds=[4],
        total_attempts=20, duration_s=12.3,
    )
    p = save_gt_seeds(result, _cfg(tmp_path))
    assert p.is_file()

    loaded = load_gt_seeds("t", _cfg(tmp_path))
    assert loaded is not None
    assert loaded.successful_seeds == [0, 3, 7, 12, 19]
    assert loaded.total_attempts == 20
    assert loaded.task == "t"


@pytest.mark.unit
def test_load_missing_returns_none(tmp_path) -> None:
    assert load_gt_seeds("nope", _cfg(tmp_path)) is None


@pytest.mark.unit
def test_load_corrupt_returns_none(tmp_path) -> None:
    p = gt_seeds_path("t", _cfg(tmp_path))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("not valid json {{{")
    assert load_gt_seeds("t", _cfg(tmp_path)) is None


# -------------------- load_or_find_gt_seeds --------------------


@pytest.mark.unit
def test_load_or_find_uses_cache_when_sufficient(tmp_path) -> None:
    result = GTSeedResult(
        task="t", target_count=5, max_attempts=20,
        successful_seeds=[0, 3, 7, 12, 19],
    )
    save_gt_seeds(result, _cfg(tmp_path))

    # collect_gt_seeds should NOT be called since cache has enough.
    with patch("closed_loop_cap.tools.find_gt_seeds.collect_gt_seeds") as mock_collect:
        seeds = load_or_find_gt_seeds(
            "t", _cfg(tmp_path), target_count=5,
        )
    assert seeds == [0, 3, 7, 12, 19]
    mock_collect.assert_not_called()


@pytest.mark.unit
def test_load_or_find_collects_when_cache_missing(tmp_path) -> None:
    mock_result = GTSeedResult(
        task="t", target_count=3, max_attempts=10,
        successful_seeds=[0, 1, 5],
    )
    with patch(
        "closed_loop_cap.tools.find_gt_seeds.collect_gt_seeds",
        return_value=mock_result,
    ) as mock_collect:
        seeds = load_or_find_gt_seeds(
            "t", _cfg(tmp_path), target_count=3, max_attempts=10,
        )
    assert seeds == [0, 1, 5]
    mock_collect.assert_called_once()

    # Cache was persisted.
    cached = load_gt_seeds("t", _cfg(tmp_path))
    assert cached is not None
    assert cached.successful_seeds == [0, 1, 5]


@pytest.mark.unit
def test_load_or_find_collects_when_cache_insufficient(tmp_path) -> None:
    # Cache has only 2 but we need 5 → re-collect.
    save_gt_seeds(
        GTSeedResult(task="t", target_count=5, max_attempts=20,
                     successful_seeds=[0, 1]),
        _cfg(tmp_path),
    )
    mock_result = GTSeedResult(
        task="t", target_count=5, max_attempts=20,
        successful_seeds=[0, 1, 7, 9, 11],
    )
    with patch(
        "closed_loop_cap.tools.find_gt_seeds.collect_gt_seeds",
        return_value=mock_result,
    ) as mock_collect:
        seeds = load_or_find_gt_seeds("t", _cfg(tmp_path), target_count=5)
    assert seeds == [0, 1, 7, 9, 11]
    mock_collect.assert_called_once()


@pytest.mark.unit
def test_load_or_find_rebuild_flag(tmp_path) -> None:
    # Cache has plenty, but --rebuild forces re-collect.
    save_gt_seeds(
        GTSeedResult(task="t", target_count=5, max_attempts=20,
                     successful_seeds=[0, 1, 2, 3, 4]),
        _cfg(tmp_path),
    )
    mock_result = GTSeedResult(
        task="t", target_count=5, max_attempts=20,
        successful_seeds=[99, 100, 101, 102, 103],
    )
    with patch(
        "closed_loop_cap.tools.find_gt_seeds.collect_gt_seeds",
        return_value=mock_result,
    ) as mock_collect:
        seeds = load_or_find_gt_seeds(
            "t", _cfg(tmp_path), target_count=5, rebuild=True,
        )
    mock_collect.assert_called_once()
    assert seeds == [99, 100, 101, 102, 103]


@pytest.mark.unit
def test_load_or_find_collect_if_missing_false(tmp_path) -> None:
    """When collect_if_missing=False and no cache, return empty list."""
    with patch("closed_loop_cap.tools.find_gt_seeds.collect_gt_seeds") as mock_collect:
        seeds = load_or_find_gt_seeds(
            "t", _cfg(tmp_path), target_count=5, collect_if_missing=False,
        )
    assert seeds == []
    mock_collect.assert_not_called()


@pytest.mark.unit
def test_load_or_find_returns_partial_cache_when_collect_disabled(tmp_path) -> None:
    save_gt_seeds(
        GTSeedResult(task="t", target_count=5, max_attempts=20,
                     successful_seeds=[0, 1]),
        _cfg(tmp_path),
    )
    with patch("closed_loop_cap.tools.find_gt_seeds.collect_gt_seeds") as mock_collect:
        seeds = load_or_find_gt_seeds(
            "t", _cfg(tmp_path), target_count=5, collect_if_missing=False,
        )
    assert seeds == [0, 1]
    mock_collect.assert_not_called()
