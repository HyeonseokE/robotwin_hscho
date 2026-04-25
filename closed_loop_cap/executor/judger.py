"""Layer 2 + Layer 3 post-execution judgement.

This module converts the raw exec outcome plus SAPIEN state into a typed
JudgeResult matching failure_detection_and_recovery.md. Every return path has
a signal_id so the orchestrator can log and build retry hints consistently.

We intentionally keep this module free of VLM calls — it reads numeric state
only. The Verifier (optional Phase 5) is where image-based judgement lives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from closed_loop_cap.env.task_env import EnvHandle, RobotState, snapshot_robot_state
from closed_loop_cap.executor.episode_state import EpisodeState
from closed_loop_cap.executor.sandbox import ExecOutcome
from closed_loop_cap.vlm.schema import SubtaskSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JudgeResult:
    ok: bool
    layer: int                  # 1 | 2 | 3
    signal_id: str              # e.g. "L2-S6" | "L3-P1" | "L3-U1"
    detail: str
    hint_for_codegen: str | None
    episode_abort: bool = False


# -----------------------------
# Layer 3 physics-instability checks (L3-U*)
# -----------------------------


def _collect_scene_actors(task_env: Any) -> list[tuple[str, Any]]:
    """Return (name, actor) pairs for actors registered on the task env.

    Tasks typically expose actors via `self.actor_name_dic` and friends
    (see envs/_base_task.py). We fall back to introspection if that attr
    is missing so this works across tasks that keep the dict minimal.
    """
    items: list[tuple[str, Any]] = []
    for attr in ("actor_name_dic", "actor_dic"):
        d = getattr(task_env, attr, None)
        if isinstance(d, dict):
            for k, v in d.items():
                items.append((str(k), v))
    return items


def check_instability(
    handle: EnvHandle,
    linear_velocity_threshold: float,
    table_drop_margin: float,
) -> JudgeResult | None:
    """Return an abort JudgeResult if physics has diverged, else None."""
    table_h = getattr(handle.task_env, "table_height", None)
    for name, actor in _collect_scene_actors(handle.task_env):
        try:
            vel = np.asarray(actor.get_linear_velocity())
        except Exception:
            continue
        if vel.size and np.linalg.norm(vel) > linear_velocity_threshold:
            return JudgeResult(
                ok=False,
                layer=3,
                signal_id="L3-U1",
                detail=f"actor {name} velocity={np.linalg.norm(vel):.2f} m/s exceeds threshold",
                hint_for_codegen=None,
                episode_abort=True,
            )
        try:
            pose = np.asarray(actor.get_pose().p)
        except Exception:
            continue
        if not np.all(np.isfinite(pose)):
            return JudgeResult(
                ok=False, layer=3, signal_id="L3-U2",
                detail=f"actor {name} pose contains NaN/Inf",
                hint_for_codegen=None, episode_abort=True,
            )
        if table_h is not None and pose[2] < float(table_h) - float(table_drop_margin):
            return JudgeResult(
                ok=False, layer=3, signal_id="L3-U3",
                detail=f"actor {name} z={pose[2]:.3f} fell below table {table_h:.3f}",
                hint_for_codegen=None, episode_abort=True,
            )
    return None


# -----------------------------
# Layer 2 runtime signals (L2-S5 / L2-S6 / L2-S7)
# -----------------------------


def _is_no_op(
    before: RobotState,
    after: RobotState,
    ee_delta_threshold_m: float,
    gripper_delta_threshold: float,
) -> bool:
    d_left_ee = float(np.linalg.norm(after.left_ee_pose[:3] - before.left_ee_pose[:3]))
    d_right_ee = float(np.linalg.norm(after.right_ee_pose[:3] - before.right_ee_pose[:3]))
    d_left_g = abs(after.left_gripper - before.left_gripper)
    d_right_g = abs(after.right_gripper - before.right_gripper)
    ee_still = (d_left_ee < ee_delta_threshold_m) and (d_right_ee < ee_delta_threshold_m)
    gripper_still = (d_left_g < gripper_delta_threshold) and (d_right_g < gripper_delta_threshold)
    return ee_still and gripper_still


# -----------------------------
# Layer 3 planner + skill postconditions
# -----------------------------


def _plan_signal(handle: EnvHandle) -> JudgeResult | None:
    te = handle.task_env
    if not bool(getattr(te, "plan_success", True)):
        # Find most specific arm flag.
        if hasattr(te, "left_plan_success") and not bool(te.left_plan_success):
            return JudgeResult(
                ok=False, layer=3, signal_id="L3-P2",
                detail="left arm motion planning failed",
                hint_for_codegen=(
                    "Left arm cannot reach the target. Try the other arm, or adjust "
                    "pre_grasp_dis / target pose."
                ),
            )
        if hasattr(te, "right_plan_success") and not bool(te.right_plan_success):
            return JudgeResult(
                ok=False, layer=3, signal_id="L3-P3",
                detail="right arm motion planning failed",
                hint_for_codegen=(
                    "Right arm cannot reach the target. Try the other arm, or adjust "
                    "pre_grasp_dis / target pose."
                ),
            )
        return JudgeResult(
            ok=False, layer=3, signal_id="L3-P1",
            detail="motion planner failed (plan_success=False)",
            hint_for_codegen=(
                "Motion planning failed. Try a different pre_grasp_dis, different arm_tag, "
                "or a simpler target pose."
            ),
        )
    return None


# Gripper-finger link-name tokens we treat as "this link is a fingertip".
# We accept a small heuristic set rather than hard-coding per embodiment; tasks
# that need more accuracy can override via config later.
_GRIPPER_FINGER_TOKENS = ("finger", "grip", "_link7", "_link8")


def _resolve_target_actor_name(
    task_env: Any,
    target_actor_ref: str,
) -> str | None:
    """Map 'self.hammer' → the string name SAPIEN uses for the actor entity.

    Returns None when the attribute is missing or has no get_name(), so the
    caller can gracefully skip the postcondition (warn-and-pass).
    """
    if not isinstance(target_actor_ref, str) or not target_actor_ref.startswith("self."):
        return None
    attr = target_actor_ref[len("self."):]
    actor = getattr(task_env, attr, None)
    if actor is None:
        return None
    getter = getattr(actor, "get_name", None)
    if getter is None:
        return None
    try:
        name = getter()
    except Exception:  # noqa: BLE001 — defensive, actors may be in a bad state
        return None
    return name if isinstance(name, str) and name else None


def _gripper_finger_link_names(robot: Any, arm_tag: str) -> set[str]:
    """Collect SAPIEN entity names of finger links for the given arm.

    Uses a heuristic token match against link names. Empty result → caller
    treats it as "cannot decide" and skips the contact postcondition.
    """
    entity = getattr(robot, "left_entity" if arm_tag == "left" else "right_entity", None)
    if entity is None:
        return set()
    get_links = getattr(entity, "get_links", None)
    if get_links is None:
        return set()
    names: set[str] = set()
    try:
        links = get_links()
    except Exception:  # noqa: BLE001
        return set()
    for link in links:
        get_name = getattr(link, "get_name", None)
        if get_name is None:
            continue
        try:
            n = get_name()
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(n, str):
            continue
        low = n.lower()
        if any(tok in low for tok in _GRIPPER_FINGER_TOKENS):
            names.add(n)
    return names


def _gripper_touches_target(
    task_env: Any,
    target_name: str,
    arm_tag: str,
) -> bool | None:
    """True/False = decisive; None = cannot determine (no finger names found)."""
    fingers = _gripper_finger_link_names(task_env.robot, arm_tag)
    if not fingers:
        return None
    scene = getattr(task_env, "scene", None)
    if scene is None:
        return None
    try:
        contacts = scene.get_contacts()
    except Exception:  # noqa: BLE001
        return None
    for c in contacts:
        try:
            n1 = c.bodies[0].entity.name
            n2 = c.bodies[1].entity.name
        except Exception:  # noqa: BLE001
            continue
        if n1 == target_name and n2 in fingers:
            return True
        if n2 == target_name and n1 in fingers:
            return True
    return False


def _get_actor_ref(task_env: Any, ref: str) -> Any | None:
    """'self.hammer' → the SAPIEN Actor object on the task env, or None."""
    if not isinstance(ref, str) or not ref.startswith("self."):
        return None
    return getattr(task_env, ref[len("self."):], None)


def _actor_pose_xy(actor: Any) -> np.ndarray | None:
    """Best-effort xy extraction; tolerates Actor variants without get_pose."""
    try:
        p = actor.get_pose().p
        arr = np.asarray(p, dtype=np.float64)
        return arr[:2] if arr.size >= 2 else None
    except Exception:  # noqa: BLE001
        return None


def _postcondition_signal(
    handle: EnvHandle,
    subtask: SubtaskSpec,
    before: RobotState,
    after: RobotState,
    episode_state: EpisodeState,
    *,
    gripper_tol: float = 0.05,
    destination_tol_m: float = 0.10,
) -> JudgeResult | None:
    """Check skill-specific postconditions and mutate `episode_state`.

    The state-aware variant threads one piece of context across subtasks:
    which actor each arm is holding. This lets `place` verify both the
    release of the grasped object AND that the object actually landed
    near the destination — something the single-subtask snapshot could
    not previously tell.
    """
    st = subtask.skill_type
    task_env = handle.task_env

    if st == "open_gripper":
        g = after.left_gripper if subtask.arm_tag == "left" else after.right_gripper
        if abs(g - 1.0) > gripper_tol:
            return JudgeResult(
                ok=False, layer=3, signal_id="L3-SKL-OPEN",
                detail=f"gripper did not open (value={g:.2f})",
                hint_for_codegen="Gripper did not reach open position. Increase pos toward 1.0.",
            )
        # Opening the gripper typically releases anything held.
        episode_state.on_place_success(subtask.arm_tag)
        return None

    if st == "close_gripper":
        g = after.left_gripper if subtask.arm_tag == "left" else after.right_gripper
        if abs(g - 0.0) > gripper_tol:
            return JudgeResult(
                ok=False, layer=3, signal_id="L3-SKL-CLOSE",
                detail=f"gripper did not close (value={g:.2f})",
                hint_for_codegen="Gripper did not reach closed position. Use pos=0.0.",
            )
        return None

    if st == "grasp":
        target_name = _resolve_target_actor_name(task_env, subtask.target_actor)
        if target_name is None:
            logger.warning(
                "grasp postcondition skipped: cannot resolve target %r",
                subtask.target_actor,
            )
            return None
        touches = _gripper_touches_target(task_env, target_name, subtask.arm_tag)
        if touches is None:
            logger.warning(
                "grasp postcondition skipped: no gripper finger links for arm=%s",
                subtask.arm_tag,
            )
            return None
        if not touches:
            return JudgeResult(
                ok=False, layer=3, signal_id="L3-SKL-GRASP",
                detail=(
                    f"after grasp, gripper is not in contact with {target_name!r} "
                    f"(arm={subtask.arm_tag})"
                ),
                hint_for_codegen=(
                    f"Grasp failed: gripper is not in contact with {subtask.target_actor}. "
                    "Try a different contact_point_id, reduce grasp_dis, or verify the "
                    "target is actually graspable from this arm."
                ),
            )
        episode_state.on_grasp_success(subtask.arm_tag, subtask.target_actor)
        return None

    if st == "place":
        arm = subtask.arm_tag
        held_ref = episode_state.held_by(arm)
        dest_actor_ref = subtask.target_actor  # planner convention: destination

        if held_ref is None:
            # No prior grasp recorded — fall back to old semantics: whatever
            # the planner listed as target_actor, check gripper is not holding it.
            fallback_name = _resolve_target_actor_name(task_env, dest_actor_ref)
            if fallback_name is None:
                return None
            still_touches = _gripper_touches_target(task_env, fallback_name, arm)
            if still_touches:
                return JudgeResult(
                    ok=False, layer=3, signal_id="L3-SKL-PLACE",
                    detail=(
                        f"after place, gripper is still in contact with {fallback_name!r} "
                        f"(arm={arm}; no prior grasp in episode state)"
                    ),
                    hint_for_codegen=(
                        f"Place failed: gripper is still holding {dest_actor_ref}. "
                        "Open the gripper wider (pos=1.0) or increase release displacement."
                    ),
                )
            return None

        # --- Full path: we know what the arm was holding ---
        held_name = _resolve_target_actor_name(task_env, held_ref)
        if held_name is not None:
            still_holding = _gripper_touches_target(task_env, held_name, arm)
            if still_holding is True:
                return JudgeResult(
                    ok=False, layer=3, signal_id="L3-SKL-PLACE",
                    detail=(
                        f"after place, gripper is still holding {held_ref} "
                        f"(arm={arm})"
                    ),
                    hint_for_codegen=(
                        f"Place failed: gripper is still holding {held_ref}. "
                        "Open the gripper wider (pos=1.0) or increase the release "
                        "displacement (dis). Confirm is_open=True."
                    ),
                )

        # Destination proximity: held object should have landed near the
        # place destination. Uses actor-center pose; a coarse 10cm default
        # keeps this tolerant of varying destination functional-point IDs.
        held_obj = _get_actor_ref(task_env, held_ref)
        dest_obj = _get_actor_ref(task_env, dest_actor_ref)
        if held_obj is not None and dest_obj is not None:
            held_xy = _actor_pose_xy(held_obj)
            dest_xy = _actor_pose_xy(dest_obj)
            if held_xy is not None and dest_xy is not None:
                d = float(np.linalg.norm(held_xy - dest_xy))
                if d > destination_tol_m:
                    return JudgeResult(
                        ok=False, layer=3, signal_id="L3-SKL-PLACE-DIST",
                        detail=(
                            f"after place, {held_ref} is {d:.3f}m (xy) from "
                            f"{dest_actor_ref} (tol={destination_tol_m:.2f}m)"
                        ),
                        hint_for_codegen=(
                            f"Place failed: {held_ref} ended up {d:.2f}m from "
                            f"{dest_actor_ref}. Pass the destination's "
                            f"functional_point as target_pose "
                            f"(e.g. {dest_actor_ref}.get_functional_point(0, 'pose')), "
                            "adjust pre_dis_axis to 'fp', reduce dis, or check that "
                            "you targeted the right functional_point_id."
                        ),
                    )

        # Place accepted → drop state so downstream subtasks don't assume
        # the arm is still holding something.
        episode_state.on_place_success(arm)
        return None

    # move_to_pose / move_by_displacement postconditions need a target pose,
    # which the SubtaskSpec does not currently carry. Skipped for MVP — the
    # planner-layer signals (L3-P*) already catch unreachable targets.
    return None


# -----------------------------
# Top-level judger
# -----------------------------


def _task_success_shortcut(handle: EnvHandle) -> bool:
    """Best-effort call to Base_Task.check_success(). Any exception → False so
    we never overturn a real failure just because check_success misbehaves."""
    try:
        return bool(handle.task_env.check_success())
    except Exception:  # noqa: BLE001
        return False


def judge_after_exec(
    handle: EnvHandle,
    subtask: SubtaskSpec,
    exec_outcome: ExecOutcome,
    before: RobotState,
    noop_ee_delta_m: float,
    noop_gripper_delta: float,
    instability_velocity_threshold: float,
    instability_table_drop_margin: float,
    *,
    episode_state: EpisodeState | None = None,
    place_destination_tolerance_m: float = 0.10,
) -> JudgeResult:
    """Combine Layer 2 + Layer 3 signals into a single verdict.

    `episode_state` threads per-arm held-object tracking across subtasks so
    the place post-condition can verify the real release + destination. When
    None (legacy call sites), a throw-away EpisodeState is used.
    """
    if episode_state is None:
        episode_state = EpisodeState()

    # L2-S7 timeout
    if exec_outcome.timed_out:
        return JudgeResult(
            ok=False, layer=2, signal_id="L2-S7",
            detail=f"exec timed out after {exec_outcome.duration_s:.1f}s",
            hint_for_codegen="Previous attempt timed out. Simplify the motion or reduce steps.",
        )
    # L2-S5 runtime exception
    if not exec_outcome.ok and exec_outcome.exception:
        tail = "\n".join(exec_outcome.exception.strip().splitlines()[-3:])
        return JudgeResult(
            ok=False, layer=2, signal_id="L2-S5",
            detail=f"runtime exception:\n{tail}",
            hint_for_codegen=(
                f"Previous code raised an exception:\n{tail}\n"
                "Check actor names and skill signatures before retrying."
            ),
        )

    # Physics instability (episode abort)
    instab = check_instability(
        handle,
        linear_velocity_threshold=instability_velocity_threshold,
        table_drop_margin=instability_table_drop_margin,
    )
    if instab is not None:
        return instab

    # Layer 3 planner signals
    plan_sig = _plan_signal(handle)
    if plan_sig is not None:
        return plan_sig

    # L2-S6 no-op detection (after we know planner didn't fail)
    after = snapshot_robot_state(handle)
    if _is_no_op(before, after, noop_ee_delta_m, noop_gripper_delta):
        return JudgeResult(
            ok=False, layer=2, signal_id="L2-S6",
            detail="exec completed but robot did not move (ΔEE and Δgripper ~ 0)",
            hint_for_codegen=(
                "Previous code did not move the robot. You MUST call self.move(...) "
                "with a valid action sequence."
            ),
        )

    # Task-level success shortcut — if the hand-written check_success is already
    # True, the per-subtask nitpicks are irrelevant; accept the step.
    if _task_success_shortcut(handle):
        return JudgeResult(
            ok=True, layer=0, signal_id="task_done",
            detail="check_success()=True",
            hint_for_codegen=None,
        )

    # Skill postcondition (EpisodeState-aware)
    post_sig = _postcondition_signal(
        handle, subtask, before, after, episode_state,
        destination_tol_m=place_destination_tolerance_m,
    )
    if post_sig is not None:
        return post_sig

    return JudgeResult(ok=True, layer=0, signal_id="", detail="", hint_for_codegen=None)
