"""Unit tests for sandbox AST validation (no SAPIEN required)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from closed_loop_cap.executor.sandbox import (  # noqa: E402
    ast_validate,
    validate_actor_refs,
)


@pytest.mark.unit
def test_ast_rejects_import() -> None:
    rep = ast_validate("import os", known_actor_names={"self.hammer"})
    assert not rep.ok and rep.signal_id == "L2-S2"


@pytest.mark.unit
def test_ast_rejects_eval() -> None:
    rep = ast_validate("eval('1+1')", known_actor_names=set())
    assert not rep.ok and rep.signal_id == "L2-S2"


@pytest.mark.unit
def test_ast_rejects_dunder_attr() -> None:
    rep = ast_validate("self.__class__", known_actor_names=set())
    assert not rep.ok and rep.signal_id == "L2-S2"


@pytest.mark.unit
def test_ast_rejects_open() -> None:
    rep = ast_validate("open('/etc/passwd')", known_actor_names=set())
    assert not rep.ok


@pytest.mark.unit
def test_ast_accepts_simple_skill_call() -> None:
    code = 'self.move(self.grasp_actor(self.hammer, arm_tag=ArmTag("left")))'
    rep = ast_validate(code, known_actor_names={"self.hammer"})
    assert rep.ok, rep.detail


@pytest.mark.unit
def test_ast_syntax_error_marked_l2_s1() -> None:
    rep = ast_validate("self.move(", known_actor_names=set())
    assert not rep.ok and rep.signal_id == "L2-S1"


@pytest.mark.unit
def test_ref_check_catches_bogus_actor() -> None:
    code = "self.move(self.grasp_actor(self.unicorn, arm_tag=ArmTag('left')))"
    rep = validate_actor_refs(code, known_actor_names={"self.hammer"})
    assert not rep.ok and rep.signal_id == "L2-S3"
    assert "self.unicorn" in rep.detail


@pytest.mark.unit
def test_ref_check_accepts_base_task_attr() -> None:
    # self.robot is a known Base_Task attribute; should not be flagged
    code = "x = self.robot\nself.move(self.grasp_actor(self.hammer, arm_tag=ArmTag('left')))"
    rep = validate_actor_refs(code, known_actor_names={"self.hammer"})
    assert rep.ok, rep.detail
