"""Typed shapes for VLM requests and planner responses.

Kept free of Pydantic dependency for Phase 2 to avoid adding a runtime dep
before we know we need validator features. Plain dataclasses + a light
validate() function cover our needs (field presence, type, value membership)
and keep the retry loop able to report field-level errors for L1-S1/S2/S3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ArmTagLiteral = Literal["left", "right"]
SkillTypeLiteral = Literal[
    "grasp",
    "place",
    "move_to_pose",
    "move_by_displacement",
    "open_gripper",
    "close_gripper",
]

ALLOWED_SKILL_TYPES: tuple[str, ...] = (
    "grasp",
    "place",
    "move_to_pose",
    "move_by_displacement",
    "open_gripper",
    "close_gripper",
)
ALLOWED_ARM_TAGS: tuple[str, ...] = ("left", "right")


@dataclass(frozen=True)
class SubtaskSpec:
    id: int
    instruction: str
    skill_type: str       # one of ALLOWED_SKILL_TYPES
    target_actor: str     # must resolve to a self.<name> in the task env
    arm_tag: str          # one of ALLOWED_ARM_TAGS
    success_hint: str


@dataclass(frozen=True)
class PlannerResponse:
    subtasks: tuple[SubtaskSpec, ...]


@dataclass(frozen=True)
class VLMRequest:
    system: str
    user_text: str
    images: tuple[bytes, ...] = ()        # PNG-encoded bytes, RGB
    response_format: str = "text"         # "text" | "json" | "code"


@dataclass(frozen=True)
class VLMResponse:
    raw_text: str
    finish_reason: str
    prompt_tokens: int = 0
    output_tokens: int = 0


def validate_planner_payload(
    payload: dict,
    known_actors: set[str],
) -> tuple[PlannerResponse | None, list[str]]:
    """Validate a decoded planner JSON payload.

    Returns (response, errors). When errors is non-empty, response is None and
    the caller should re-prompt with the errors embedded.
    """
    errors: list[str] = []

    subtasks_raw = payload.get("subtasks")
    if not isinstance(subtasks_raw, list):
        return None, ["'subtasks' must be a list"]

    if len(subtasks_raw) == 0:
        errors.append("'subtasks' must not be empty (L1-S6)")

    validated: list[SubtaskSpec] = []
    for i, item in enumerate(subtasks_raw):
        prefix = f"subtasks[{i}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix}: must be an object")
            continue
        missing = [
            k
            for k in ("id", "instruction", "skill_type", "target_actor", "arm_tag", "success_hint")
            if k not in item
        ]
        if missing:
            errors.append(f"{prefix}: missing fields {missing}")
            continue
        if item["skill_type"] not in ALLOWED_SKILL_TYPES:
            errors.append(
                f"{prefix}: skill_type={item['skill_type']!r} not in {ALLOWED_SKILL_TYPES} (L1-S4)"
            )
        if item["arm_tag"] not in ALLOWED_ARM_TAGS:
            errors.append(
                f"{prefix}: arm_tag={item['arm_tag']!r} not in {ALLOWED_ARM_TAGS} (L1-S5)"
            )
        if known_actors and item["target_actor"] not in known_actors:
            errors.append(
                f"{prefix}: target_actor={item['target_actor']!r} "
                f"not in known actors {sorted(known_actors)} (L1-S3 hallucination)"
            )
        try:
            spec = SubtaskSpec(
                id=int(item["id"]),
                instruction=str(item["instruction"]),
                skill_type=str(item["skill_type"]),
                target_actor=str(item["target_actor"]),
                arm_tag=str(item["arm_tag"]),
                success_hint=str(item["success_hint"]),
            )
            validated.append(spec)
        except (TypeError, ValueError) as exc:
            errors.append(f"{prefix}: field coercion failed: {exc}")

    if errors:
        return None, errors
    return PlannerResponse(subtasks=tuple(validated)), []
