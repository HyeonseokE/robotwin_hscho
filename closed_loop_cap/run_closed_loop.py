"""Closed-loop CaP orchestrator.

Glues together Phase 1 (env) → Phase 2 (VLM) → Phase 3 (planner) →
Phase 4 (executor/sandbox/judger) into the main loop described in
docs/failure_detection_and_recovery.md §7.

Usage:
    python -m closed_loop_cap.run_closed_loop --task shake_bottle --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import imageio
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from closed_loop_cap.env.task_env import (  # noqa: E402
    EnvHandle,
    capture_rgb,
    close_env,
    is_task_success,
    make_env,
    snapshot_robot_state,
)
from closed_loop_cap.executor.executor import (  # noqa: E402
    ExecResult,
    execute_snippet,
    generate_subtask_code,
)
from closed_loop_cap.executor.episode_state import EpisodeState  # noqa: E402
from closed_loop_cap.executor.judger import JudgeResult, judge_after_exec  # noqa: E402
from closed_loop_cap.planner.planner import PlannerFailure, plan_subtasks  # noqa: E402
from closed_loop_cap.task_registry import load_task_meta  # noqa: E402
from closed_loop_cap.paths import (  # noqa: E402
    default_session_id,
    logs_root as _logs_root,
    save_subdir as _save_subdir,
    trial_dir as _trial_dir,
)
from closed_loop_cap.perturbation.subgoal import (  # noqa: E402
    SubgoalPerturbation,
    SubgoalPerturbationConfig,
)
from closed_loop_cap.vlm.client import GeminiClient  # noqa: E402
from closed_loop_cap.vlm.parsers import ParseError  # noqa: E402
from closed_loop_cap.vlm.schema import SubtaskSpec  # noqa: E402

logger = logging.getLogger("closed_loop_cap")


# -----------------------------
# Report structures
# -----------------------------


@dataclass
class SubtaskAttempt:
    attempt: int
    code: str
    signal_id: str
    detail: str
    ok: bool


@dataclass
class SubtaskRecord:
    subtask: dict                      # serialized SubtaskSpec
    attempts: list[SubtaskAttempt] = field(default_factory=list)
    final_ok: bool = False


@dataclass
class EpisodeReport:
    task: str
    seed: int
    success: bool
    abort_reason: str | None
    num_subtasks: int
    subtasks: list[SubtaskRecord] = field(default_factory=list)
    duration_s: float = 0.0
    trial: int | None = None


# -----------------------------
# Utilities
# -----------------------------


# Keys overlaid from configs/paid_api_config.yaml onto config["vlm"].
# Mirrors AutoDataCollector/pipeline_config/paid_api_config.yaml's role in
# centralizing model assignment per pipeline role.
_PAID_API_MODEL_KEYS: tuple[str, ...] = (
    "planner_model",
    "codegen_model",
    "judge_vlm_model",
    "detect_objects_model",
)


def _apply_paid_api_overrides(config: dict, config_path: Path) -> None:
    """Overlay model assignments from paid_api_config.yaml (if present in the
    same dir as the main config) onto ``config["vlm"]``. Acts as the single
    source of truth for model names; falls back silently when the file is
    absent."""
    paid_api_path = config_path.parent / "paid_api_config.yaml"
    if not paid_api_path.is_file():
        return
    try:
        with paid_api_path.open("r", encoding="utf-8") as f:
            paid_api = yaml.safe_load(f) or {}
    except Exception:  # noqa: BLE001
        logger.exception("failed to load %s; ignoring", paid_api_path)
        return
    if not isinstance(paid_api, dict):
        logger.warning("paid_api_config.yaml is not a mapping; ignoring")
        return
    vlm_cfg = config.setdefault("vlm", {})
    for key in _PAID_API_MODEL_KEYS:
        if key in paid_api and paid_api[key] is not None:
            vlm_cfg[key] = paid_api[key]


def _apply_recording_overlay(config: dict, config_path: Path) -> dict:
    """Auto-overlay ``recording_config.yaml`` from the same directory as
    *config_path* onto the pipeline config (logging / dataset / camera /
    perturbation sections). Returns a new dict; original is not mutated."""
    recording_path = config_path.parent / "recording_config.yaml"
    if not recording_path.is_file():
        return config
    try:
        from closed_loop_cap.configs.recording_config_loader import load_and_merge
    except Exception:  # noqa: BLE001
        logger.exception("failed to import recording_config_loader; skipping overlay")
        return config
    try:
        return load_and_merge(config, recording_path)
    except Exception:  # noqa: BLE001
        logger.exception("failed to merge %s; ignoring", recording_path)
        return config


def _load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    _apply_paid_api_overrides(config, path)
    config = _apply_recording_overlay(config, path)
    return config


def _build_client(config: dict, role: str = "codegen") -> GeminiClient:
    """Build a GeminiClient for a given pipeline role.

    role ∈ {"planner", "codegen", ...}: selects ``vlm.<role>_model``. Falls
    back to ``codegen_model`` when the role-specific field is missing.
    """
    vlm_cfg = config.get("vlm", {})
    model = vlm_cfg.get(f"{role}_model") or vlm_cfg.get("codegen_model", "gemini-2.0-flash")
    return GeminiClient(
        api_key_path=REPO_ROOT / vlm_cfg.get("api_key_path", "closed_loop_cap/gemini_api_key.json"),
        model=model,
        temperature=vlm_cfg.get("temperature", 0.0),
        max_output_tokens=vlm_cfg.get("max_output_tokens", 4096),
        max_retries=vlm_cfg.get("api_max_retries", 3),
        backoff_base_s=vlm_cfg.get("api_backoff_base_s", 2.0),
    )


def _known_actor_set(actor_names: tuple[str, ...]) -> set[str]:
    return {f"self.{n}" for n in actor_names}


def _save_episode_artifacts(
    out_dir: Path,
    step_idx: int,
    before_rgb: np.ndarray,
    after_rgb: np.ndarray | None,
    code: str,
    judge: JudgeResult,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_dir / f"step_{step_idx:02d}_before.png", before_rgb)
    if after_rgb is not None:
        imageio.imwrite(out_dir / f"step_{step_idx:02d}_after.png", after_rgb)
    (out_dir / f"step_{step_idx:02d}_code.py").write_text(code)
    (out_dir / f"step_{step_idx:02d}_judge.json").write_text(
        json.dumps(
            {
                "ok": judge.ok,
                "layer": judge.layer,
                "signal_id": judge.signal_id,
                "detail": judge.detail,
                "episode_abort": judge.episode_abort,
            },
            indent=2,
        )
    )


# -----------------------------
# Main episode loop
# -----------------------------


def _relocate_traj_pkl(handle: EnvHandle, ep_dir: Path) -> bool:
    """RoboTwin writes `{save_dir}/_traj_data/episode0.pkl`. We redirect the
    canonical artifact to `ep_dir/traj.pkl` (rename, not copy).

    Returns True if the file was relocated, False otherwise.
    """
    src = Path(handle.task_env.save_dir) / "_traj_data" / "episode0.pkl"
    dst = ep_dir / "traj.pkl"
    if not src.is_file():
        logger.warning(
            "traj.pkl source not found at %s (save_dir=%s); "
            "left_joint_path has %d segments",
            src, handle.task_env.save_dir,
            len(getattr(handle.task_env, "left_joint_path", [])),
        )
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    src.rename(dst)
    return True


def _write_subtask_timeline(
    ep_dir: Path, report: "EpisodeReport", handle: EnvHandle,
) -> None:
    """Dump per-subtask boundaries + labels so replay can reconstruct skill
    annotations (see docs §4.2)."""
    timeline: list[dict] = []
    # Use length of joint_path at each subtask boundary as step index. The
    # joint paths grow append-only inside self.move(), so a snapshot of lengths
    # taken per subtask gives exact boundaries. For the MVP we record end-only
    # boundaries post-hoc and let replay assume cumulative step counts.
    # TODO(P5): take snapshots during subtask execution for precise boundaries.
    for rec in report.subtasks:
        spec = rec.subtask if isinstance(rec.subtask, dict) else {}
        timeline.append(
            {
                "subtask_id": spec.get("id"),
                "natural_language": spec.get("instruction", ""),
                "skill_type": spec.get("skill_type", ""),
                "target_actor": spec.get("target_actor", ""),
                "arm_tag": spec.get("arm_tag", ""),
                "success_hint": spec.get("success_hint", ""),
                "final_ok": rec.final_ok,
            }
        )
    (ep_dir / "subtask_timeline.json").write_text(
        json.dumps({"subtasks": timeline}, indent=2, default=str)
    )


def _save_traj_and_timeline(
    handle: EnvHandle, ep_dir: Path, report: "EpisodeReport",
    config: dict, task_name: str, seed: int, trial: int,
) -> None:
    """Save traj.pkl + subtask_timeline.json. Called from every return path."""
    if not config.get("save_successful_traj", True):
        return
    try:
        handle.task_env.save_traj_data(0)
        relocated = _relocate_traj_pkl(handle, ep_dir)
        logger.info("save_traj_data %s seed=%d trial=%d: relocated=%s", task_name, seed, trial, relocated)
    except Exception:
        logger.exception("save_traj_data failed for %s seed=%d trial=%d", task_name, seed, trial)
    try:
        _write_subtask_timeline(ep_dir, report, handle)
    except Exception:
        logger.exception("subtask_timeline write failed for %s seed=%d trial=%d", task_name, seed, trial)


def _is_unstable_seed_error(exc: BaseException) -> bool:
    """Match RoboTwin's UnStableError without a hard import dependency."""
    # envs.utils.create_actor.UnStableError — name-based match keeps this file
    # runnable in environments where RoboTwin isn't installed (unit tests).
    if type(exc).__name__ == "UnStableError":
        return True
    msg = str(exc).lower()
    return "unstable" in msg and "seed" in msg


def run_episode(
    task_name: str,
    seed: int,
    config: dict,
    client: GeminiClient | None = None,
    *,
    planner_client: GeminiClient | None = None,
    codegen_client: GeminiClient | None = None,
    trial: int | None = None,
    session: str | None = None,
) -> EpisodeReport:
    # paid_api_config.yaml drives per-role model selection; callers may pass
    # planner/codegen clients explicitly, or a single legacy `client` that
    # serves both roles (preserved for tests).
    if planner_client is None:
        planner_client = client
    if codegen_client is None:
        codegen_client = client
    t0 = time.monotonic()
    output_dir = Path(REPO_ROOT) / config.get("output_dir", "closed_loop_cap/output")
    # Layout (see docs/replay_dataset_logging.md §2 + paths.py):
    #   output/datasets/<session>/<task>/logs/seed_<N>/trial_<KKK>/
    #   output/datasets/<session>/<task>/recorded_data/
    # trial_KKK is always present (1-indexed, 3-digit); default trial=1.
    effective_trial = trial if trial is not None else 1
    session_id = session or default_session_id()
    ep_dir = _trial_dir(output_dir, session_id, task_name, seed, effective_trial)
    save_subdir = _save_subdir(session_id, task_name, seed, effective_trial)

    meta = load_task_meta(task_name)
    handle: EnvHandle | None = None
    report = EpisodeReport(
        task=task_name, seed=seed, success=False, abort_reason=None,
        num_subtasks=0, trial=effective_trial,
    )

    # Wrap make_env separately so an UnStableError (random object placement
    # that fails physics stability in setup_demo) is recorded as a skip rather
    # than crashing the whole benchmark.
    episode_mp4 = ep_dir / "episode.mp4" if config.get("video", {}).get("enabled", False) else None
    try:
        handle = make_env(
            task_name, seed, config,
            episode_mp4_path=episode_mp4,
            save_subdir=save_subdir,
        )
    except Exception as exc:  # noqa: BLE001
        if _is_unstable_seed_error(exc):
            logger.warning("seed %d for %s unstable — skipping", seed, task_name)
            report.abort_reason = "unstable_seed"
            report.duration_s = time.monotonic() - t0
            try:
                ep_dir.mkdir(parents=True, exist_ok=True)
                (ep_dir / "report.json").write_text(
                    json.dumps(
                        {
                            "task": task_name,
                            "seed": seed,
                            "success": False,
                            "abort_reason": "unstable_seed",
                            "num_subtasks": 0,
                            "duration_s": round(report.duration_s, 3),
                            "subtasks": [],
                        },
                        indent=2,
                    )
                )
            except Exception:  # noqa: BLE001
                logger.exception("failed to write report.json for unstable seed")
            return report
        raise

    try:
        initial_rgb = capture_rgb(handle)
        ep_dir.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(ep_dir / "initial.png", initial_rgb)

        # ---- Layer 1: plan
        try:
            plan = plan_subtasks(
                initial_rgb=initial_rgb,
                task_instruction=meta.description,
                actor_names=list(meta.actor_names),
                client=planner_client,
                max_retries=config.get("max_plan_retries", 2),
                image_max_side_px=config.get("vlm", {}).get("image_max_side_px", 768),
            )
        except PlannerFailure as exc:
            report.abort_reason = f"L1_planner_failed: {exc}"
            (ep_dir / "plan_errors.json").write_text(json.dumps(exc.errors, indent=2))
            return report

        (ep_dir / "plan.json").write_text(
            json.dumps([asdict(s) for s in plan.subtasks], indent=2)
        )
        report.num_subtasks = len(plan.subtasks)

        known_actors = _known_actor_set(meta.actor_names)
        max_subtask_retries = int(config.get("max_subtask_retries", 3))
        max_consecutive_fails = int(config.get("max_consecutive_subtask_fails", 2))
        max_total_steps = int(config.get("max_total_steps", 30))
        timeout_s = float(config.get("subtask_exec_timeout_s", 30.0))
        noop_ee = float(config.get("noop_ee_delta_m", 0.01))
        noop_gripper = float(config.get("noop_gripper_delta", 0.01))
        instab_v = float(config.get("instability_linear_velocity_threshold_m_s", 2.0))
        instab_drop = float(config.get("instability_table_drop_margin_m", 0.1))
        place_dest_tol = float(config.get("place_destination_tolerance_m", 0.10))

        total_steps = 0
        consecutive_fails = 0
        episode_state = EpisodeState()

        # ---- Subgoal perturbation ----
        pert_cfg_raw = config.get("perturbation", {}).get("subgoal", {})
        pert = SubgoalPerturbation(SubgoalPerturbationConfig(
            enabled=bool(pert_cfg_raw.get("enabled", False)),
            sigma=float(pert_cfg_raw.get("sigma", 0.05)),
            clip_factor=float(pert_cfg_raw.get("clip_factor", 2.0)),
        ))
        ep_rng = np.random.default_rng(seed * 10000 + effective_trial)

        # ---- Layer 2/3: per-subtask execution loop
        for st_idx, subtask in enumerate(plan.subtasks):
            rec = SubtaskRecord(subtask=asdict(subtask))
            hint: str | None = None
            st_dir = ep_dir / f"subtask_{subtask.id:02d}"

            for attempt in range(max_subtask_retries):
                total_steps += 1
                if total_steps > max_total_steps:
                    report.abort_reason = f"max_total_steps_exceeded ({max_total_steps})"
                    rec.final_ok = False
                    report.subtasks.append(rec)
                    return report

                before_rgb = capture_rgb(handle)
                before_state = snapshot_robot_state(handle)

                # Codegen (Layer 2). On retries inject the place-actor domain
                # hint (cheap heuristic: only relevant for place skills); on
                # the first attempt keep the prompt compact.
                from closed_loop_cap.prompts.codegen_prompt import DOMAIN_HINTS_PLACE
                domain_hints = (
                    DOMAIN_HINTS_PLACE
                    if (attempt > 0 and subtask.skill_type == "place")
                    else None
                )
                try:
                    code = generate_subtask_code(
                        current_rgb=before_rgb,
                        subtask=subtask,
                        actor_names=list(meta.actor_names),
                        client=codegen_client,
                        actor_details=meta.actor_details,
                        full_task_description=meta.description,
                        previous_failure_hint=hint,
                        domain_hints=domain_hints,
                        image_max_side_px=config.get("vlm", {}).get("image_max_side_px", 768),
                    )
                except ParseError as exc:
                    hint = f"[format] {exc}"
                    rec.attempts.append(
                        SubtaskAttempt(attempt, code="", signal_id="L2-S1", detail=str(exc), ok=False)
                    )
                    continue

                # Start a per-attempt subtask video segment. The frames also
                # flow into the whole-episode mp4 via the recorder's tee.
                if handle.recorder is not None:
                    st_dir.mkdir(parents=True, exist_ok=True)
                    handle.recorder.start_subtask(st_dir / f"attempt_{attempt:02d}.mp4")

                # Subgoal perturbation: sample one offset per subtask. The
                # skill functions' per-Action transit tag decides which move
                # Actions actually receive it (see Base_Task.move).
                offset = pert.sample(rng=ep_rng)
                handle.task_env._subgoal_offset = offset
                if offset is not None:
                    logger.info(
                        "subgoal perturbation subtask=%d skill=%s offset=[%.4f, %.4f, %.4f]",
                        subtask.id, subtask.skill_type, *offset,
                    )

                # Execute + judge
                ex: ExecResult = execute_snippet(
                    handle, code, timeout_s=timeout_s, known_actor_names=known_actors
                )
                handle.task_env._subgoal_offset = None  # clear after execution

                if ex.outcome is None:
                    # Static validation blocked exec → L2-S1/S2/S3
                    judge = JudgeResult(
                        ok=False,
                        layer=2,
                        signal_id=ex.static_report.signal_id,
                        detail=ex.static_report.detail,
                        hint_for_codegen=f"[static] {ex.static_report.detail}",
                    )
                    after_rgb = None
                else:
                    judge = judge_after_exec(
                        handle,
                        subtask,
                        ex.outcome,
                        before_state,
                        noop_ee_delta_m=noop_ee,
                        noop_gripper_delta=noop_gripper,
                        instability_velocity_threshold=instab_v,
                        instability_table_drop_margin=instab_drop,
                        episode_state=episode_state,
                        place_destination_tolerance_m=place_dest_tol,
                    )
                    after_rgb = capture_rgb(handle)

                # Close the per-attempt segment before the judge's cleanup so
                # the mp4 finalizes even on abort/failure paths.
                if handle.recorder is not None:
                    handle.recorder.end_subtask()

                _save_episode_artifacts(st_dir, attempt, before_rgb, after_rgb, code, judge)
                rec.attempts.append(
                    SubtaskAttempt(
                        attempt, code=code, signal_id=judge.signal_id, detail=judge.detail, ok=judge.ok
                    )
                )

                if judge.episode_abort:
                    report.abort_reason = f"physics_instability: {judge.signal_id} {judge.detail}"
                    rec.final_ok = False
                    report.subtasks.append(rec)
                    _save_traj_and_timeline(handle, ep_dir, report, config, task_name, seed, effective_trial)
                    return report

                if judge.ok:
                    rec.final_ok = True
                    break

                hint = judge.hint_for_codegen or f"[{judge.signal_id}] {judge.detail}"

            report.subtasks.append(rec)

            if rec.final_ok:
                consecutive_fails = 0
                # Early stop if the task already succeeded mid-plan.
                if is_task_success(handle):
                    report.success = True
                    _save_traj_and_timeline(handle, ep_dir, report, config, task_name, seed, effective_trial)
                    return report
            else:
                consecutive_fails += 1
                logger.warning(
                    "subtask %d exhausted retries (consecutive_fails=%d/%d)",
                    subtask.id,
                    consecutive_fails,
                    max_consecutive_fails,
                )
                # L1-S7 cascade: ≥ N consecutive subtask failures → the plan
                # itself is probably wrong. Abort rather than burn budget on
                # dependent subtasks that will cascade-fail anyway.
                if consecutive_fails >= max_consecutive_fails:
                    report.abort_reason = (
                        f"L1-S7_cascade: {consecutive_fails} consecutive "
                        f"subtask failures (last subtask_id={subtask.id})"
                    )
                    _save_traj_and_timeline(handle, ep_dir, report, config, task_name, seed, effective_trial)
                    return report
                # Otherwise continue best-effort to the next subtask.
                logger.info(
                    "continuing to next subtask despite failure of subtask_id=%d",
                    subtask.id,
                )

        report.success = is_task_success(handle)
        # Save traj.pkl for BOTH success and failure (see docs §8). Replay pass
        # consumes this to regenerate state/action/images at logging FPS.
        _save_traj_and_timeline(handle, ep_dir, report, config, task_name, seed, effective_trial)

        return report
    finally:
        if handle is not None:
            close_env(handle)
        report.duration_s = time.monotonic() - t0
        if 'ep_dir' in locals():
            try:
                ep_dir.mkdir(parents=True, exist_ok=True)
                (ep_dir / "report.json").write_text(
                    json.dumps(
                        {
                            **{k: v for k, v in asdict(report).items() if k != "subtasks"},
                            "subtasks": [asdict(r) for r in report.subtasks],
                        },
                        indent=2,
                        default=str,
                    )
                )
            except Exception:
                logger.exception("failed to write report.json for %s seed=%d", task_name, seed)


# -----------------------------
# CLI
# -----------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Closed-loop CaP for RoboTwin")
    parser.add_argument("--task", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--trials", type=int, default=1,
        help="repeats per seed. Emits trial_KKK/ per repeat (always).",
    )
    parser.add_argument(
        "--start-trial", type=int, default=1,
        help="first trial index (1-indexed). Default 1 → trial_001.",
    )
    parser.add_argument(
        "--session", default=None,
        help=("Session id for output directory. Default: auto-generated "
              "YYYYMMDD_HHMMSS. Use a stable name to append to an existing "
              "session (e.g. 'perturb_on')."),
    )
    parser.add_argument(
        "--recording-config", default=None,
        help=("Optional recording_config.yaml to overlay camera / feature / "
              "perturbation settings onto the main config."),
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "closed_loop_cap" / "configs" / "default.yaml"),
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be >= 1")
    if args.start_trial < 1:
        parser.error("--start-trial must be >= 1 (1-indexed)")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose >= 2 else logging.INFO if args.verbose == 1 else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = _load_config(Path(args.config))
    if args.recording_config:
        from closed_loop_cap.configs.recording_config_loader import load_and_merge
        config = load_and_merge(config, args.recording_config)
    planner_client = _build_client(config, role="planner")
    codegen_client = _build_client(config, role="codegen")

    # Fix the session id up-front so every trial in this invocation lands in
    # the same session directory.
    session_id = args.session or default_session_id()
    print(f"session: {session_id}")

    # Trial indices are always concrete (1-indexed). --start-trial defaults to 1.
    start = args.start_trial if args.start_trial > 0 else 1
    trial_iter = list(range(start, start + args.trials))

    summary: list[dict] = []
    for seed in args.seeds:
        for trial in trial_iter:
            report = run_episode(
                args.task, seed=seed, config=config,
                planner_client=planner_client, codegen_client=codegen_client,
                trial=trial, session=session_id,
            )
            summary.append(
                {
                    "task": report.task,
                    "seed": report.seed,
                    "trial": report.trial,
                    "success": report.success,
                    "abort_reason": report.abort_reason,
                    "num_subtasks": report.num_subtasks,
                    "duration_s": round(report.duration_s, 2),
                }
            )
            print(
                f"[seed {report.seed} trial={trial}] success={report.success} "
                f"subtasks={report.num_subtasks} abort={report.abort_reason or '-'}"
            )

    total = len(summary)
    unstable = sum(1 for s in summary if s["abort_reason"] == "unstable_seed")
    valid = total - unstable
    ok = sum(1 for s in summary if s["success"])
    if total == 0:
        print("no seeds run")
    else:
        # Report two rates: one over all requested seeds (with unstable as fails)
        # and one over only valid (setup-stable) seeds.
        print(f"\nOverall: {ok}/{total} = {ok / total:.0%}")
        if unstable:
            print(
                f"Excluding {unstable} unstable-seed skip(s): "
                f"{ok}/{valid} = {(ok / valid) if valid else 0:.0%}"
            )
    out_dir = _logs_root(
        Path(REPO_ROOT) / config.get("output_dir", "closed_loop_cap/output"),
        session_id, args.task,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "per_seed": summary,
                "totals": {
                    "requested": total,
                    "unstable_skipped": unstable,
                    "valid": valid,
                    "success": ok,
                },
            },
            indent=2,
        )
    )
    # Success if all valid seeds succeeded (unstable skips don't count as failures).
    return 0 if valid > 0 and ok == valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
