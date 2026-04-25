"""Unit tests for grasp/place/open/close postcondition helpers (no SAPIEN).

Covers:
    - Helpers: _resolve_target_actor_name, _gripper_finger_link_names,
      _gripper_touches_target, _get_actor_ref, _actor_pose_xy
    - _postcondition_signal:
        * grasp: success records held_by_arm, failure returns L3-SKL-GRASP
        * place with prior grasp: release + destination proximity
        * place without prior grasp: fallback to old "not holding target" check
        * open_gripper clears held_by_arm
    - judge_after_exec:
        * task_done shortcut when check_success() returns True
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from closed_loop_cap.env.task_env import EnvHandle, RobotState  # noqa: E402
from closed_loop_cap.executor.episode_state import EpisodeState  # noqa: E402
from closed_loop_cap.executor.judger import (  # noqa: E402
    _actor_pose_xy,
    _get_actor_ref,
    _gripper_finger_link_names,
    _gripper_touches_target,
    _postcondition_signal,
    _resolve_target_actor_name,
    judge_after_exec,
)
from closed_loop_cap.executor.sandbox import ExecOutcome  # noqa: E402
from closed_loop_cap.vlm.schema import SubtaskSpec  # noqa: E402


# -------------------- lightweight fakes --------------------


class _FakePose:
    def __init__(self, xyz):
        self.p = list(xyz)


class _FakeActor:
    def __init__(self, name: str, xyz=(0.0, 0.0, 0.0)):
        self._name = name
        self._pose = _FakePose(xyz)

    def get_name(self) -> str:
        return self._name

    def get_pose(self):
        return self._pose

    def get_linear_velocity(self):
        return [0.0, 0.0, 0.0]


class _FakeLink:
    def __init__(self, name: str) -> None:
        self._name = name

    def get_name(self) -> str:
        return self._name


class _FakeEntity:
    def __init__(self, link_names: list[str]) -> None:
        self._links = [_FakeLink(n) for n in link_names]

    def get_links(self) -> list[_FakeLink]:
        return list(self._links)


class _FakeRobot:
    def __init__(self, left_links: list[str], right_links: list[str]) -> None:
        self.left_entity = _FakeEntity(left_links)
        self.right_entity = _FakeEntity(right_links)


class _FakeBody:
    def __init__(self, name: str) -> None:
        self.entity = type("E", (), {"name": name})()


class _FakeContact:
    def __init__(self, a: str, b: str) -> None:
        self.bodies = (_FakeBody(a), _FakeBody(b))


class _FakeScene:
    def __init__(self, contacts: list[tuple[str, str]]) -> None:
        self._contacts = [_FakeContact(a, b) for a, b in contacts]

    def get_contacts(self) -> list[_FakeContact]:
        return list(self._contacts)


@dataclass
class _FakeTaskEnv:
    robot: _FakeRobot
    scene: _FakeScene
    hammer: Any = None
    block: Any = None
    actor_name_dic: dict | None = None

    def check_success(self) -> bool:  # overridden per-test
        return False


def _handle(task_env: Any) -> EnvHandle:
    return EnvHandle(task_env=task_env, task_name="fake", seed=0, config={})


def _state() -> RobotState:
    import numpy as np

    return RobotState(
        left_ee_pose=np.zeros(7),
        right_ee_pose=np.zeros(7),
        left_gripper=0.0,
        right_gripper=0.0,
    )


def _subtask(skill: str, target: str = "self.hammer", arm: str = "left", st_id: int = 1) -> SubtaskSpec:
    return SubtaskSpec(
        id=st_id,
        instruction="test",
        skill_type=skill,
        target_actor=target,
        arm_tag=arm,
        success_hint="",
    )


# -------------------- resolvers & helpers --------------------


@pytest.mark.unit
def test_resolve_self_prefix_ok() -> None:
    env = _FakeTaskEnv(robot=_FakeRobot([], []), scene=_FakeScene([]))
    env.hammer = _FakeActor("020_hammer")
    assert _resolve_target_actor_name(env, "self.hammer") == "020_hammer"


@pytest.mark.unit
def test_resolve_missing_returns_none() -> None:
    env = _FakeTaskEnv(robot=_FakeRobot([], []), scene=_FakeScene([]))
    assert _resolve_target_actor_name(env, "self.unknown") is None


@pytest.mark.unit
def test_get_actor_ref_returns_actor_object() -> None:
    env = _FakeTaskEnv(robot=_FakeRobot([], []), scene=_FakeScene([]))
    env.hammer = _FakeActor("h")
    assert _get_actor_ref(env, "self.hammer") is env.hammer
    assert _get_actor_ref(env, "self.nope") is None
    assert _get_actor_ref(env, "hammer") is None


@pytest.mark.unit
def test_actor_pose_xy_extracts_first_two() -> None:
    a = _FakeActor("a", xyz=(1.0, 2.0, 3.0))
    import numpy as np
    xy = _actor_pose_xy(a)
    assert xy is not None
    np.testing.assert_allclose(xy, [1.0, 2.0])


@pytest.mark.unit
def test_gripper_link_names_token_match() -> None:
    robot = _FakeRobot(
        left_links=["fl_link1", "fl_link7", "fl_link8", "fl_cam"],
        right_links=["fr_link7", "fr_link8"],
    )
    assert _gripper_finger_link_names(robot, "left") == {"fl_link7", "fl_link8"}
    assert _gripper_finger_link_names(robot, "right") == {"fr_link7", "fr_link8"}


@pytest.mark.unit
def test_touches_true_when_contact_present() -> None:
    env = _FakeTaskEnv(
        robot=_FakeRobot(left_links=["fl_link7"], right_links=[]),
        scene=_FakeScene([("020_hammer", "fl_link7")]),
    )
    assert _gripper_touches_target(env, "020_hammer", "left") is True


@pytest.mark.unit
def test_touches_false_when_no_contact() -> None:
    env = _FakeTaskEnv(
        robot=_FakeRobot(left_links=["fl_link7"], right_links=[]),
        scene=_FakeScene([("table", "fl_link1")]),
    )
    assert _gripper_touches_target(env, "020_hammer", "left") is False


# -------------------- grasp postcondition --------------------


@pytest.mark.unit
def test_grasp_success_records_held_actor() -> None:
    env = _FakeTaskEnv(
        robot=_FakeRobot(left_links=["fl_link7"], right_links=[]),
        scene=_FakeScene([("020_hammer", "fl_link7")]),
    )
    env.hammer = _FakeActor("020_hammer")
    st = EpisodeState()
    res = _postcondition_signal(_handle(env), _subtask("grasp"), _state(), _state(), st)
    assert res is None
    assert st.held_by_arm == {"left": "self.hammer"}


@pytest.mark.unit
def test_grasp_failure_no_contact() -> None:
    env = _FakeTaskEnv(
        robot=_FakeRobot(left_links=["fl_link7"], right_links=[]),
        scene=_FakeScene([]),   # no contact
    )
    env.hammer = _FakeActor("020_hammer")
    st = EpisodeState()
    res = _postcondition_signal(_handle(env), _subtask("grasp"), _state(), _state(), st)
    assert res is not None
    assert res.signal_id == "L3-SKL-GRASP"
    assert st.held_by_arm == {}   # nothing was recorded on failure


@pytest.mark.unit
def test_grasp_skipped_when_target_unresolvable() -> None:
    env = _FakeTaskEnv(
        robot=_FakeRobot(left_links=["fl_link7"], right_links=[]),
        scene=_FakeScene([]),
    )
    # No env.hammer defined at all → target resolves to None → skip.
    st = EpisodeState()
    res = _postcondition_signal(
        _handle(env), _subtask("grasp", target="self.unknown"), _state(), _state(), st,
    )
    assert res is None
    assert st.held_by_arm == {}


# -------------------- place postcondition (stateful path) --------------------


@pytest.mark.unit
def test_place_success_releases_and_lands_near_destination() -> None:
    env = _FakeTaskEnv(
        robot=_FakeRobot(left_links=["fl_link7"], right_links=[]),
        scene=_FakeScene([]),   # gripper no longer in contact with hammer
    )
    env.hammer = _FakeActor("020_hammer", xyz=(0.30, 0.20, 0.80))
    env.block = _FakeActor("001_block", xyz=(0.32, 0.21, 0.80))   # within 10cm of hammer
    st = EpisodeState(held_by_arm={"left": "self.hammer"})
    res = _postcondition_signal(
        _handle(env), _subtask("place", target="self.block"), _state(), _state(), st,
    )
    assert res is None, res
    # Held slot cleared
    assert st.held_by_arm == {}


@pytest.mark.unit
def test_place_fails_when_still_holding() -> None:
    env = _FakeTaskEnv(
        robot=_FakeRobot(left_links=["fl_link7"], right_links=[]),
        scene=_FakeScene([("020_hammer", "fl_link7")]),   # still held
    )
    env.hammer = _FakeActor("020_hammer", xyz=(0.0, 0.0, 0.8))
    env.block = _FakeActor("001_block", xyz=(0.0, 0.0, 0.78))
    st = EpisodeState(held_by_arm={"left": "self.hammer"})
    res = _postcondition_signal(
        _handle(env), _subtask("place", target="self.block"), _state(), _state(), st,
    )
    assert res is not None
    assert res.signal_id == "L3-SKL-PLACE"
    assert "still holding" in res.detail


@pytest.mark.unit
def test_place_fails_when_held_lands_far_from_destination() -> None:
    env = _FakeTaskEnv(
        robot=_FakeRobot(left_links=["fl_link7"], right_links=[]),
        scene=_FakeScene([]),   # gripper released
    )
    env.hammer = _FakeActor("020_hammer", xyz=(0.10, 0.10, 0.80))
    env.block = _FakeActor("001_block", xyz=(0.50, 0.50, 0.80))   # ~0.57m away
    st = EpisodeState(held_by_arm={"left": "self.hammer"})
    res = _postcondition_signal(
        _handle(env), _subtask("place", target="self.block"), _state(), _state(), st,
        destination_tol_m=0.10,
    )
    assert res is not None
    assert res.signal_id == "L3-SKL-PLACE-DIST"
    assert "0.5" in res.detail
    # Held slot NOT cleared on failure — downstream subtasks still know
    # something is (probably) in the gripper.
    assert st.held_by_arm == {"left": "self.hammer"}


@pytest.mark.unit
def test_place_distance_tolerance_honored() -> None:
    """A larger tolerance should accept an otherwise-failed place."""
    env = _FakeTaskEnv(
        robot=_FakeRobot(left_links=["fl_link7"], right_links=[]),
        scene=_FakeScene([]),
    )
    env.hammer = _FakeActor("020_hammer", xyz=(0.10, 0.10, 0.80))
    env.block = _FakeActor("001_block", xyz=(0.15, 0.12, 0.80))   # ~0.054m
    st = EpisodeState(held_by_arm={"left": "self.hammer"})
    res = _postcondition_signal(
        _handle(env), _subtask("place", target="self.block"), _state(), _state(), st,
        destination_tol_m=0.10,
    )
    assert res is None
    assert st.held_by_arm == {}


# -------------------- place fallback path (no prior grasp) --------------------


@pytest.mark.unit
def test_place_fallback_when_no_grasp_state() -> None:
    env = _FakeTaskEnv(
        robot=_FakeRobot(left_links=["fl_link7"], right_links=[]),
        scene=_FakeScene([("001_block", "fl_link7")]),   # gripper touches destination
    )
    env.block = _FakeActor("001_block")
    res = _postcondition_signal(
        _handle(env), _subtask("place", target="self.block"), _state(), _state(), EpisodeState(),
    )
    assert res is not None
    assert res.signal_id == "L3-SKL-PLACE"


# -------------------- open/close gripper --------------------


@pytest.mark.unit
def test_open_gripper_clears_held() -> None:
    env = _FakeTaskEnv(robot=_FakeRobot([], []), scene=_FakeScene([]))
    st = EpisodeState(held_by_arm={"left": "self.hammer"})
    after = RobotState(
        left_ee_pose=__import__("numpy").zeros(7),
        right_ee_pose=__import__("numpy").zeros(7),
        left_gripper=1.0,
        right_gripper=0.0,
    )
    res = _postcondition_signal(
        _handle(env), _subtask("open_gripper", target="self.hammer"),
        _state(), after, st,
    )
    assert res is None
    assert st.held_by_arm == {}


# -------------------- judge_after_exec task_done shortcut --------------------


@pytest.mark.unit
def test_judge_after_exec_shortcuts_when_task_already_succeeded() -> None:
    import numpy as np
    env = _FakeTaskEnv(robot=_FakeRobot(left_links=["fl_link7"], right_links=[]),
                      scene=_FakeScene([]))
    env.check_success = lambda: True  # type: ignore[method-assign]
    env.hammer = _FakeActor("020_hammer")

    before = RobotState(
        left_ee_pose=np.zeros(7), right_ee_pose=np.zeros(7),
        left_gripper=0.0, right_gripper=0.0,
    )
    outcome = ExecOutcome(ok=True, exception=None, duration_s=0.1, timed_out=False)

    # EE delta big enough to pass the no-op gate (1 cm).
    import types
    import numpy as np
    # Patch snapshot_robot_state to return non-stale "after"
    from closed_loop_cap.executor import judger as jm

    after_state = RobotState(
        left_ee_pose=np.array([0.5] + [0.0] * 6),
        right_ee_pose=np.zeros(7),
        left_gripper=0.0, right_gripper=0.0,
    )
    orig = jm.snapshot_robot_state
    jm.snapshot_robot_state = lambda h: after_state  # type: ignore[assignment]
    try:
        result = judge_after_exec(
            _handle(env),
            _subtask("grasp", target="self.hammer"),
            outcome, before,
            noop_ee_delta_m=0.01, noop_gripper_delta=0.01,
            instability_velocity_threshold=2.0,
            instability_table_drop_margin=0.1,
            episode_state=EpisodeState(),
        )
    finally:
        jm.snapshot_robot_state = orig

    assert result.ok is True
    assert result.signal_id == "task_done"
