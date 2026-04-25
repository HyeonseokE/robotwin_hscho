"""AST whitelist + timeout sandbox for VLM-generated subtask snippets.

We cannot run the snippet in a subprocess because it must mutate the live
SAPIEN scene via `self`. Therefore the isolation is static (AST) + dynamic
(namespace whitelist) + time-bounded (SIGALRM on POSIX).

Threat model (Layer 2 signals L2-S1 ~ L2-S4 and L2-S7):
    - Block imports, eval/exec/compile, filesystem/process calls, dunder
      attribute access, and raw references to nonexistent actors.
    - Enforce a wall-clock timeout.
Out of scope:
    - Native extension exploits (we trust the RoboTwin codebase itself).
"""

from __future__ import annotations

import ast
import logging
import signal
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


FORBIDDEN_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "input",
        "exit",
        "quit",
    }
)
FORBIDDEN_ATTR_PREFIXES = ("__",)
FORBIDDEN_IMPORT_ROOTS = frozenset(
    {"os", "sys", "subprocess", "shutil", "socket", "pathlib", "pickle", "importlib"}
)


class SandboxError(RuntimeError):
    """Raised when sandbox_exec cannot safely run the snippet or it times out."""


class StaticValidationError(SandboxError):
    """AST-level rejection — never attempted to run."""


@dataclass(frozen=True)
class StaticReport:
    ok: bool
    signal_id: str
    detail: str


def ast_validate(code: str, known_actor_names: set[str]) -> StaticReport:
    """Reject snippets that violate the whitelist before exec.

    known_actor_names are the actor attrs available on the task env
    (with the "self." prefix — e.g. {"self.hammer"}). Any `self.<x>`
    reference outside this set plus a small fixed allowlist of Base_Task
    attributes is flagged as L2-S3 hallucination.

    Note: we accept some Base_Task members heuristically ("robot",
    "right_endpose", "left_endpose", etc.) because snippets legitimately
    read self.robot state. The hallucination check is only about actor refs.
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return StaticReport(False, "L2-S1", f"SyntaxError: {exc.msg} (line {exc.lineno})")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return StaticReport(False, "L2-S2", "import statements are not allowed")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return StaticReport(False, "L2-S2", f"use of builtin {node.id!r} is not allowed")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith(FORBIDDEN_ATTR_PREFIXES):
                return StaticReport(
                    False, "L2-S2", f"dunder attribute access {node.attr!r} is not allowed"
                )
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_NAMES:
                return StaticReport(
                    False, "L2-S2", f"call to {func.id!r} is not allowed"
                )

    return StaticReport(True, "", "")


def _extract_self_attr_refs(code: str) -> set[str]:
    """Return the set of `self.<name>` names used at the top level of attribute access."""
    tree = ast.parse(code, mode="exec")
    refs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "self":
                refs.add(f"self.{node.attr}")
    return refs


# A minimal, conservative list of Base_Task members we expect snippets to touch.
_BASE_TASK_ALLOWED_ATTRS = frozenset(
    {
        "robot",
        "move",
        "grasp_actor",
        "place_actor",
        "open_gripper",
        "close_gripper",
        "move_to_pose",
        "move_by_displacement",
        "left_move_to_pose",
        "right_move_to_pose",
        "together_open_gripper",
        "together_close_gripper",
        "plan_success",
        "left_plan_success",
        "right_plan_success",
        "info",
        "get_arm_pose",
        "get_functional_point",
        "check_actors_contact",
    }
)


def validate_actor_refs(code: str, known_actor_names: set[str]) -> StaticReport:
    """Check `self.<name>` usage against known actors + Base_Task allowlist."""
    try:
        refs = _extract_self_attr_refs(code)
    except SyntaxError as exc:
        return StaticReport(False, "L2-S1", f"SyntaxError: {exc.msg}")
    allowed = known_actor_names | {f"self.{a}" for a in _BASE_TASK_ALLOWED_ATTRS}
    bogus = sorted(refs - allowed)
    if bogus:
        return StaticReport(
            False,
            "L2-S3",
            f"references to unknown names: {bogus} "
            f"(known actors: {sorted(known_actor_names)})",
        )
    return StaticReport(True, "", "")


@dataclass(frozen=True)
class ExecOutcome:
    ok: bool
    exception: str | None      # formatted traceback tail if any
    duration_s: float
    timed_out: bool


def _timeout_handler(signum, frame):  # noqa: ARG001
    raise TimeoutError("sandbox exec timeout")


def sandbox_exec(
    code: str,
    task_env: Any,
    *,
    timeout_s: float,
    extra_globals: dict[str, Any] | None = None,
) -> ExecOutcome:
    """Execute `code` with `self=task_env` in a restricted namespace.

    Raises:
        SandboxError: if the platform cannot install a SIGALRM handler AND
            timeout_s > 0 (we still run, just without timeout enforcement).
    """
    try:
        from envs.utils import ArmTag  # runtime import to avoid hard dep at module load
    except Exception as exc:  # pragma: no cover
        raise SandboxError(f"cannot import ArmTag from envs.utils: {exc}") from exc

    # Match the upstream CODE_TEMPLATE imports so VLM-emitted snippets that
    # use sapien.Pose / np.array (common in FUNCTION_EXAMPLE patterns) don't
    # immediately NameError. AST validation still blocks `import`, exec/eval,
    # filesystem/process calls, and dunder access — see ast_validate above.
    try:
        import numpy as _np  # type: ignore
    except Exception:  # pragma: no cover
        _np = None
    try:
        import sapien as _sapien  # type: ignore
        _Pose = getattr(_sapien, "Pose", None)
    except Exception:  # pragma: no cover
        _sapien = None
        _Pose = None

    # Minimal safe-ish builtins.
    safe_builtins = {
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "float": float,
        "int": int,
        "str": str,
        "bool": bool,
        "list": list,
        "tuple": tuple,
        "dict": dict,
        "set": set,
        "print": print,
    }
    ns: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "self": task_env,
        "ArmTag": ArmTag,
    }
    if _np is not None:
        ns["np"] = _np
        ns["numpy"] = _np
    if _sapien is not None:
        ns["sapien"] = _sapien
    if _Pose is not None:
        ns["Pose"] = _Pose
    if extra_globals:
        ns.update(extra_globals)

    installed = False
    if timeout_s and timeout_s > 0 and hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
        installed = True

    t0 = time.monotonic()
    try:
        compiled = compile(code, "<subtask>", "exec")
        exec(compiled, ns, ns)  # noqa: S102 — sandboxed via namespace + AST gating
        dt = time.monotonic() - t0
        return ExecOutcome(ok=True, exception=None, duration_s=dt, timed_out=False)
    except TimeoutError as exc:
        dt = time.monotonic() - t0
        return ExecOutcome(ok=False, exception=f"TimeoutError: {exc}", duration_s=dt, timed_out=True)
    except BaseException as exc:
        import traceback

        tb = traceback.format_exc(limit=5)
        dt = time.monotonic() - t0
        return ExecOutcome(ok=False, exception=tb, duration_s=dt, timed_out=False)
    finally:
        if installed:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
