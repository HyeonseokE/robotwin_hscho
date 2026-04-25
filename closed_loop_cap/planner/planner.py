"""Layer 1 — Task Planner.

One VLM call per episode (plus up to max_plan_retries on validation failure).
Input: initial RGB + task instruction + actor list.
Output: PlannerResponse (tuple of SubtaskSpec).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from closed_loop_cap.prompts.planner_prompt import (
    PLANNER_SYSTEM_PROMPT,
    build_planner_retry_text,
    build_planner_user_text,
)
from closed_loop_cap.vlm.client import GeminiClient, encode_rgb
from closed_loop_cap.vlm.parsers import ParseError, parse_json_response
from closed_loop_cap.vlm.schema import (
    PlannerResponse,
    VLMRequest,
    validate_planner_payload,
)

logger = logging.getLogger(__name__)


class PlannerFailure(RuntimeError):
    """Raised when the planner cannot produce a valid plan within the retry budget."""

    def __init__(self, reason: str, errors: list[str]) -> None:
        super().__init__(reason)
        self.errors = errors


@dataclass(frozen=True)
class _PlannerInputs:
    task_instruction: str
    actor_names: tuple[str, ...]
    known_actors: frozenset[str]


def _canonical_actor_set(actor_names: list[str]) -> frozenset[str]:
    """Map ["hammer", "block"] → {"self.hammer", "self.block"} for hallucination checks."""
    return frozenset(n if n.startswith("self.") else f"self.{n}" for n in actor_names)


def plan_subtasks(
    initial_rgb: np.ndarray,
    task_instruction: str,
    actor_names: list[str],
    client: GeminiClient,
    *,
    max_retries: int = 2,
    image_max_side_px: int = 768,
) -> PlannerResponse:
    """Run Layer 1.

    Args:
        initial_rgb: HxWx3 uint8 RGB frame of the initial scene.
        task_instruction: natural-language task (e.g. "shake the bottle").
        actor_names: list of actor attribute names on the task env
            (without the "self." prefix; e.g. ["bottle"]).
        client: a configured GeminiClient.
        max_retries: number of extra calls on validation failure (total ≤ 1 + max_retries).
        image_max_side_px: downscale cap before sending to the API.

    Raises:
        PlannerFailure: every attempt returned invalid JSON / schema / actor refs.
    """
    png = encode_rgb(initial_rgb, max_side=image_max_side_px)
    known = _canonical_actor_set(actor_names)
    errors: list[str] = []

    user_text = build_planner_user_text(task_instruction, actor_names)
    for attempt in range(max_retries + 1):
        req = VLMRequest(
            system=PLANNER_SYSTEM_PROMPT,
            user_text=user_text,
            images=(png,),
            response_format="json",
        )
        logger.info("planner: attempt %d/%d", attempt + 1, max_retries + 1)
        resp = client.call(req)

        try:
            payload = parse_json_response(resp.raw_text)
        except ParseError as exc:
            errors = [f"L1-S1 JSON parse error: {exc}"]
            logger.warning("planner attempt %d JSON parse failed: %s", attempt + 1, exc)
            user_text = build_planner_retry_text(task_instruction, actor_names, errors)
            continue

        plan, verrs = validate_planner_payload(payload, known_actors=known)
        if plan is not None:
            logger.info("planner: accepted plan with %d subtasks", len(plan.subtasks))
            return plan

        errors = verrs
        logger.warning("planner attempt %d validation failed: %s", attempt + 1, verrs)
        user_text = build_planner_retry_text(task_instruction, actor_names, verrs)

    raise PlannerFailure(
        f"planner failed after {max_retries + 1} attempts",
        errors=errors,
    )
