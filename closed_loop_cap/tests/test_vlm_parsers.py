"""Unit tests for closed_loop_cap.vlm.parsers and .schema validators.

No SAPIEN / Gemini SDK required — pure Python.
Run from repo root:
    pytest closed_loop_cap/tests/test_vlm_parsers.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from closed_loop_cap.vlm.parsers import (  # noqa: E402
    ParseError,
    parse_code_snippet,
    parse_json_response,
)
from closed_loop_cap.vlm.schema import (  # noqa: E402
    ALLOWED_ARM_TAGS,
    ALLOWED_SKILL_TYPES,
    validate_planner_payload,
)


# -----------------------------
# parse_json_response
# -----------------------------

@pytest.mark.unit
def test_parse_json_fenced_block() -> None:
    raw = 'Here is the plan:\n```json\n{"subtasks": [{"id": 1}]}\n```\nDone.'
    assert parse_json_response(raw) == {"subtasks": [{"id": 1}]}


@pytest.mark.unit
def test_parse_json_bare_object() -> None:
    raw = 'Result: {"a": 1, "b": [2, 3]}'
    assert parse_json_response(raw) == {"a": 1, "b": [2, 3]}


@pytest.mark.unit
def test_parse_json_untagged_fence_prefers_first() -> None:
    raw = '```\n{"a": 1}\n```'
    assert parse_json_response(raw) == {"a": 1}


@pytest.mark.unit
def test_parse_json_prefers_json_tagged_over_other_fences() -> None:
    raw = '```python\nprint("hi")\n```\n```json\n{"ok": true}\n```'
    assert parse_json_response(raw) == {"ok": True}


@pytest.mark.unit
def test_parse_json_empty_raises() -> None:
    with pytest.raises(ParseError):
        parse_json_response("")


@pytest.mark.unit
def test_parse_json_no_object_raises() -> None:
    with pytest.raises(ParseError):
        parse_json_response("no braces here")


@pytest.mark.unit
def test_parse_json_malformed_raises() -> None:
    with pytest.raises(ParseError):
        parse_json_response("```json\n{broken,,}\n```")


# -----------------------------
# parse_code_snippet
# -----------------------------

@pytest.mark.unit
def test_parse_code_python_fenced() -> None:
    raw = 'Sure:\n```python\nself.move(self.open_gripper(ArmTag("left")))\n```'
    assert "self.move" in parse_code_snippet(raw)


@pytest.mark.unit
def test_parse_code_py_fenced() -> None:
    raw = '```py\nx = 1\n```'
    assert parse_code_snippet(raw) == "x = 1"


@pytest.mark.unit
def test_parse_code_generic_fence() -> None:
    raw = '```\nself.move(...)\n```'
    assert parse_code_snippet(raw) == "self.move(...)"


@pytest.mark.unit
def test_parse_code_bare_codeish_string() -> None:
    raw = "def play():\n    self.move(...)"
    assert "def play" in parse_code_snippet(raw)


@pytest.mark.unit
def test_parse_code_empty_raises() -> None:
    with pytest.raises(ParseError):
        parse_code_snippet("")


@pytest.mark.unit
def test_parse_code_no_fence_no_codeish_raises() -> None:
    with pytest.raises(ParseError):
        parse_code_snippet("this is just prose with no code indicators")


# -----------------------------
# validate_planner_payload
# -----------------------------

_VALID_SUBTASK = {
    "id": 1,
    "instruction": "Grasp the red hammer.",
    "skill_type": "grasp",
    "target_actor": "self.hammer",
    "arm_tag": "left",
    "success_hint": "Gripper in contact with hammer",
}


@pytest.mark.unit
def test_validate_happy_path() -> None:
    resp, errs = validate_planner_payload(
        {"subtasks": [_VALID_SUBTASK]},
        known_actors={"self.hammer"},
    )
    assert errs == []
    assert resp is not None
    assert resp.subtasks[0].skill_type == "grasp"


@pytest.mark.unit
def test_validate_empty_subtasks() -> None:
    _, errs = validate_planner_payload({"subtasks": []}, known_actors=set())
    assert any("must not be empty" in e for e in errs)


@pytest.mark.unit
def test_validate_missing_field() -> None:
    bad = dict(_VALID_SUBTASK)
    del bad["arm_tag"]
    _, errs = validate_planner_payload({"subtasks": [bad]}, known_actors={"self.hammer"})
    assert any("missing fields" in e for e in errs)


@pytest.mark.unit
def test_validate_bad_skill_type() -> None:
    bad = dict(_VALID_SUBTASK, skill_type="teleport")
    _, errs = validate_planner_payload({"subtasks": [bad]}, known_actors={"self.hammer"})
    assert any("skill_type" in e and "L1-S4" in e for e in errs)


@pytest.mark.unit
def test_validate_bad_arm_tag() -> None:
    bad = dict(_VALID_SUBTASK, arm_tag="middle")
    _, errs = validate_planner_payload({"subtasks": [bad]}, known_actors={"self.hammer"})
    assert any("arm_tag" in e and "L1-S5" in e for e in errs)


@pytest.mark.unit
def test_validate_hallucinated_actor() -> None:
    bad = dict(_VALID_SUBTASK, target_actor="self.unicorn")
    _, errs = validate_planner_payload({"subtasks": [bad]}, known_actors={"self.hammer"})
    assert any("hallucination" in e.lower() or "L1-S3" in e for e in errs)


@pytest.mark.unit
def test_validate_skipped_when_known_actors_empty() -> None:
    # When we don't yet know the actor set, we should not over-reject.
    _, errs = validate_planner_payload({"subtasks": [_VALID_SUBTASK]}, known_actors=set())
    assert all("hallucination" not in e.lower() for e in errs)


@pytest.mark.unit
def test_allowed_constants_match_doc() -> None:
    # Guard against silent drift between code and failure_detection_and_recovery.md §8 item 12.
    assert set(ALLOWED_SKILL_TYPES) == {
        "grasp",
        "place",
        "move_to_pose",
        "move_by_displacement",
        "open_gripper",
        "close_gripper",
    }
    assert set(ALLOWED_ARM_TAGS) == {"left", "right"}
