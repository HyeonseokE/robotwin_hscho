"""Layer 2 — generate code for one subtask and execute it.

generate_subtask_code() builds the codegen prompt, calls Gemini, parses out a
Python snippet. execute_snippet() runs it in the sandbox and returns an ExecResult
the orchestrator can feed into judge_after_exec().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from closed_loop_cap.env.task_env import EnvHandle
from closed_loop_cap.executor.sandbox import (
    ExecOutcome,
    StaticReport,
    ast_validate,
    sandbox_exec,
    validate_actor_refs,
)
from closed_loop_cap.prompts.codegen_prompt import (
    CODEGEN_SYSTEM_PROMPT,
    build_codegen_user_text,
)
from closed_loop_cap.skills.skill_catalog import format_skill_catalog
from closed_loop_cap.vlm.client import GeminiClient, encode_rgb
from closed_loop_cap.vlm.parsers import ParseError, parse_code_snippet
from closed_loop_cap.vlm.schema import SubtaskSpec, VLMRequest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecResult:
    """Bundles the generated code with its execution outcome."""

    code: str                  # raw snippet as executed (after fence strip)
    static_report: StaticReport
    outcome: ExecOutcome | None  # None when static validation blocked exec


def generate_subtask_code(
    current_rgb: np.ndarray,
    subtask: SubtaskSpec,
    actor_names: list[str],
    client: GeminiClient,
    *,
    actor_details: dict | None = None,
    full_task_description: str | None = None,
    previous_failure_hint: str | None = None,
    domain_hints: str | None = None,
    image_max_side_px: int = 768,
    skill_catalog_text: str | None = None,
) -> str:
    """One Gemini call → one Python snippet (unsanitized).

    The orchestrator is responsible for AST validation (via execute_snippet)
    and retrying with an updated previous_failure_hint on failure.
    """
    catalog = skill_catalog_text if skill_catalog_text is not None else format_skill_catalog()
    user_text = build_codegen_user_text(
        subtask=subtask,
        actor_names=actor_names,
        skill_catalog=catalog,
        actor_details=actor_details,
        full_task_description=full_task_description,
        previous_failure_hint=previous_failure_hint,
        domain_hints=domain_hints,
    )
    png = encode_rgb(current_rgb, max_side=image_max_side_px)
    req = VLMRequest(
        system=CODEGEN_SYSTEM_PROMPT,
        user_text=user_text,
        images=(png,),
        response_format="code",
    )
    resp = client.call(req)
    try:
        return parse_code_snippet(resp.raw_text)
    except ParseError as exc:
        # Re-raise so the retry loop records this as L2-S1-ish format failure.
        raise ParseError(f"codegen output could not be extracted: {exc}") from exc


def execute_snippet(
    handle: EnvHandle,
    code: str,
    *,
    timeout_s: float,
    known_actor_names: set[str],
) -> ExecResult:
    """Validate + run a snippet in the sandbox.

    known_actor_names MUST include the "self." prefix (e.g. {"self.hammer"}).
    """
    static = ast_validate(code, known_actor_names=known_actor_names)
    if not static.ok:
        return ExecResult(code=code, static_report=static, outcome=None)

    ref_check = validate_actor_refs(code, known_actor_names=known_actor_names)
    if not ref_check.ok:
        return ExecResult(code=code, static_report=ref_check, outcome=None)

    outcome = sandbox_exec(code, task_env=handle.task_env, timeout_s=timeout_s)
    return ExecResult(code=code, static_report=static, outcome=outcome)
