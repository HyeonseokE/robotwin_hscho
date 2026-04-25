"""Thin wrapper around RoboTwin Base_Task for closed-loop CaP.

Responsibilities (Phase 1):
    - Instantiate a task class by name (import-only, no edits to envs/).
    - Call setup_demo with a clean args dict built from a closed_loop_cap config.
    - Expose capture_rgb, is_task_success, close_env, and snapshot_robot_state
      in a stable shape consumable by Phase 2+ modules.

Design notes:
    - We follow trajectory_refinement/initial_trajectory/run_existing_pipeline.py
      for the args-building pattern so setup_demo receives the same keys it
      expects. We do NOT import that module — the wiring is duplicated here
      to keep closed_loop_cap standalone.
    - EnvHandle is frozen so Phase 2 code cannot accidentally reassign task_env
      mid-episode; internal SAPIEN state still mutates as expected.
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TASK_CONFIG_DIR = os.path.join(REPO_ROOT, "task_config")
EMBODIMENT_CONFIG_PATH = os.path.join(TASK_CONFIG_DIR, "_embodiment_config.yml")


@dataclass(frozen=True)
class RobotState:
    """Snapshot used for no-op detection (see failure_detection_and_recovery.md §3.2 L2-S6)."""

    left_ee_pose: np.ndarray  # shape (7,) = [x, y, z, qw, qx, qy, qz]
    right_ee_pose: np.ndarray
    left_gripper: float
    right_gripper: float


@dataclass(frozen=True)
class EnvHandle:
    """Opaque handle carrying a live Base_Task instance plus identity metadata."""

    task_env: Any  # Base_Task subclass instance
    task_name: str
    seed: int
    config: dict = field(default_factory=dict)
    recorder: Any = None  # Optional FfmpegRecorder — set when video.enabled=true


def _load_embodiment_config(robot_file: str) -> dict:
    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_embodiment_file(etype: str, embodiment_types: dict) -> str:
    robot_file = embodiment_types[etype]["file_path"]
    if robot_file is None:
        raise ValueError(f"No embodiment files registered for {etype}")
    if os.path.isabs(robot_file):
        return robot_file
    return os.path.abspath(os.path.join(REPO_ROOT, robot_file))


def _build_setup_args(
    task_name: str,
    config: dict,
    save_subdir: str | None = None,
) -> dict:
    """Compose the kwargs dict that Base_Task.setup_demo expects.

    `save_subdir` (optional) is appended to the sim cache root so parallel
    rollouts of the same task don't clobber each other's episode0.hdf5 / pkl.
    Typical callers pass "<task>/seed_N" or "<task>/seed_N/trial_K".
    """
    env_cfg = config.get("env", {})
    data_type = dict(env_cfg.get("data_type", {}))
    dr = dict(env_cfg.get("domain_randomization", {}))
    embodiment = list(env_cfg.get("embodiment", ["aloha-agilex"]))

    args: dict[str, Any] = {
        "task_name": task_name,
        "render_freq": 0,
        "episode_num": 1,
        "use_seed": False,
        "save_freq": 15,
        "embodiment": embodiment,
        "language_num": 0,
        "domain_randomization": dr,
        "camera": {
            "head_camera_type": "D435",
            "wrist_camera_type": "D435",
            "collect_head_camera": True,
            "collect_wrist_camera": True,
        },
        "data_type": data_type,
        "pcd_down_sample_num": 1024,
        "pcd_crop": True,
        "save_path": os.path.join(
            REPO_ROOT,
            config.get("output_dir", "closed_loop_cap/output"),
            "_sim_cache",
            *([save_subdir] if save_subdir else []),
        ),
        "clear_cache_freq": 5,
        "collect_data": False,
        "eval_video_log": False,
        "need_plan": True,
    }

    with open(EMBODIMENT_CONFIG_PATH, "r", encoding="utf-8") as f:
        embodiment_types = yaml.safe_load(f)

    if len(embodiment) == 1:
        left = right = _resolve_embodiment_file(embodiment[0], embodiment_types)
        args["left_robot_file"] = left
        args["right_robot_file"] = right
        args["dual_arm_embodied"] = True
    elif len(embodiment) == 3:
        args["left_robot_file"] = _resolve_embodiment_file(embodiment[0], embodiment_types)
        args["right_robot_file"] = _resolve_embodiment_file(embodiment[1], embodiment_types)
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment must have 1 or 3 items")

    args["left_embodiment_config"] = _load_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = _load_embodiment_config(args["right_robot_file"])
    args["embodiment_name"] = (
        embodiment[0] if len(embodiment) == 1 else f"{embodiment[0]}+{embodiment[1]}"
    )
    return args


def _instantiate_task(task_name: str) -> Any:
    """Dynamically import envs.<task_name>.<task_name> and instantiate it."""
    module = importlib.import_module(f"envs.{task_name}")
    importlib.reload(module)
    try:
        task_cls = getattr(module, task_name)
    except AttributeError as exc:
        raise ValueError(f"envs.{task_name} has no class named {task_name}") from exc
    return task_cls()


def _install_take_picture_video_tap(task_env: Any, recorder: Any) -> None:
    """Make `_take_picture()` also push an RGB frame to the recorder.

    Base_Task's scripted-controller path (self.move → execute_path) calls
    `_take_picture()` on every physics step, but its video-write hook lives
    in `take_action()` (the policy-eval path) and therefore never fires for
    us. We wrap `_take_picture` so every call captures a head-camera RGB
    frame and forwards it to the FfmpegRecorder's tee stdin.

    We force a fresh `get_obs()` if `now_obs` has not been populated yet
    (it is not guaranteed to exist at tap-install time).
    """
    original = task_env._take_picture
    _hit = {"count": 0, "ok": 0, "err": 0}

    def patched_take_picture() -> None:
        _hit["count"] += 1
        try:
            # Always pull a fresh RGB frame. now_obs is cached between
            # get_obs() calls, so reusing it would stamp the same RGB
            # across dozens of frames and produce a static-looking video
            # even when the arm is actually moving.
            task_env._update_render()
            task_env.cameras.update_picture()
            rgb_by_cam = task_env.cameras.get_rgb()
            rgb = rgb_by_cam["head_camera"]["rgb"]
            if rgb.dtype != np.uint8:
                rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
            recorder.stdin.write(rgb.tobytes())
            _hit["ok"] += 1
        except Exception as exc:  # noqa: BLE001 — never break the sim for recording
            _hit["err"] += 1
            if _hit["err"] <= 3:
                logger.warning("video frame write failed (%d/%d): %s",
                               _hit["err"], _hit["count"], exc)
        original()

    task_env._take_picture = patched_take_picture
    task_env._video_tap_stats = _hit  # surface for post-mortem


def _maybe_build_recorder(config: dict) -> Any:
    """Create a FfmpegRecorder only when video is enabled + ffmpeg available.

    Returns None otherwise; callers treat None as "no video".
    """
    video_cfg = config.get("video", {})
    if not video_cfg.get("enabled", False):
        return None
    # Lazy import so test environments without imageio can still import task_env.
    from closed_loop_cap.video import FfmpegRecorder, camera_size_from_config

    recorder = FfmpegRecorder(
        video_size=camera_size_from_config(config),
        framerate=int(video_cfg.get("framerate", 10)),
        crf=int(video_cfg.get("crf", 23)),
    )
    if not recorder.available:
        return None
    return recorder


def make_env(
    task_name: str,
    seed: int,
    config: dict,
    *,
    episode_mp4_path: "os.PathLike | None" = None,
    save_subdir: str | None = None,
) -> EnvHandle:
    """Build a ready-to-run EnvHandle.

    The returned task_env has already been through setup_demo, so load_actors,
    robot loading, and camera setup are complete. The caller can immediately
    call capture_rgb or run actions on it.

    When `video.enabled` is true in config and `episode_mp4_path` is provided,
    an FfmpegRecorder is attached to the handle and Base_Task's eval_video
    hook is wired to write frames into it automatically via `self.move(...)`.
    """
    recorder = _maybe_build_recorder(config) if episode_mp4_path else None

    task_env = _instantiate_task(task_name)
    args = _build_setup_args(task_name, config, save_subdir=save_subdir)
    # Base_Task gates its per-step write on `self.eval_video_path is not None`;
    # we use the parent directory so Base_Task's legacy path stays meaningful.
    if recorder is not None and episode_mp4_path is not None:
        args["eval_video_save_dir"] = str(Path(episode_mp4_path).parent)
    task_env.setup_demo(now_ep_num=0, seed=seed, **args)

    if recorder is not None and episode_mp4_path is not None:
        ep_path = recorder.start_episode(Path(episode_mp4_path))
        if ep_path is not None:
            task_env._set_eval_video_ffmpeg(recorder)
            # Base_Task only writes to eval_video_ffmpeg inside take_action()
            # (the policy-eval path). Scripted controllers like `self.move(...)`
            # emit frames via `_take_picture()` instead. Monkey-patch that hook
            # so our recorder also captures scripted rollouts.
            _install_take_picture_video_tap(task_env, recorder)
            logger.info("episode video → %s", ep_path)
        else:
            # Recorder failed to spawn ffmpeg (e.g., binary vanished). Keep
            # Base_Task's eval_video_path cleared so take_action's write path
            # stays disabled.
            task_env.eval_video_path = None
            recorder = None

    logger.info("make_env: task=%s seed=%d ready (video=%s)", task_name, seed, recorder is not None)
    return EnvHandle(
        task_env=task_env, task_name=task_name, seed=seed, config=config, recorder=recorder,
    )


def replay_trajectory(
    handle: EnvHandle,
    left_joint_path: list,
    right_joint_path: list,
    step_callback=None,
) -> int:
    """Re-execute the stored per-segment plans (produced by RoboTwin's scripted
    controllers) against a freshly-seeded env.

    Each entry in `left_joint_path`/`right_joint_path` is a mplib plan-result
    dict with keys {status, position, velocity}. Paths are played in lockstep
    segment-by-segment, mirroring Base_Task's `need_plan=False` branch
    (envs/_base_task.py:823-879). After each scene.step() the step_callback
    (if provided) is invoked with the monotonically increasing step index, so
    an ADC-style recorder can capture state/action/images.

    Returns the total number of scene.step() calls executed.
    """
    task_env = handle.task_env
    step_idx = 0
    n_segments = max(len(left_joint_path), len(right_joint_path))

    for seg_i in range(n_segments):
        left_seg = left_joint_path[seg_i] if seg_i < len(left_joint_path) else None
        right_seg = right_joint_path[seg_i] if seg_i < len(right_joint_path) else None
        left_success = bool(left_seg and left_seg.get("status") == "Success")
        right_success = bool(right_seg and right_seg.get("status") == "Success")
        left_n = left_seg["position"].shape[0] if left_success else 0
        right_n = right_seg["position"].shape[0] if right_success else 0
        now_l = now_r = 0

        while now_l < left_n or now_r < right_n:
            if (
                left_success and now_l < left_n
                and (not right_success or now_l / max(left_n, 1) <= now_r / max(right_n, 1))
            ):
                task_env.robot.set_arm_joints(
                    left_seg["position"][now_l],
                    left_seg["velocity"][now_l],
                    "left",
                )
                now_l += 1
            if (
                right_success and now_r < right_n
                and (not left_success or now_r / max(right_n, 1) <= now_l / max(left_n, 1))
            ):
                task_env.robot.set_arm_joints(
                    right_seg["position"][now_r],
                    right_seg["velocity"][now_r],
                    "right",
                )
                now_r += 1
            task_env.scene.step()
            step_idx += 1
            if step_callback is not None:
                step_callback(step_idx, segment_index=seg_i)

    return step_idx


def close_env(handle: EnvHandle) -> None:
    # Shutdown ffmpeg first so any queued bytes get flushed before SAPIEN
    # teardown invalidates the renderer.
    if handle.recorder is not None:
        stats = getattr(handle.task_env, "_video_tap_stats", None)
        if stats is not None:
            logger.info(
                "video tap: %d calls, %d frames written, %d errors",
                stats["count"], stats["ok"], stats["err"],
            )
        try:
            # Closing the tee stdin mirrors Base_Task._del_eval_video_ffmpeg's
            # behavior without relying on Base_Task calling it for us.
            handle.recorder.stdin.close()
            handle.recorder.wait(timeout=5.0)
        except Exception:  # noqa: BLE001
            logger.exception("recorder shutdown failed")
    try:
        handle.task_env.close_env(clear_cache=True)
    except Exception:  # pragma: no cover — SAPIEN teardown is best-effort
        logger.exception("close_env failed for %s seed=%d", handle.task_name, handle.seed)


def capture_rgb(handle: EnvHandle, camera_name: str | None = None) -> np.ndarray:
    """Return a uint8 HxWx3 RGB frame from the requested camera."""
    cam = camera_name or handle.config.get("env", {}).get("camera_name", "head_camera")
    task_env = handle.task_env
    task_env._update_render()
    task_env.cameras.update_picture()
    rgb_by_cam = task_env.cameras.get_rgb()
    if cam not in rgb_by_cam:
        raise KeyError(f"camera {cam!r} not in {list(rgb_by_cam.keys())}")
    rgb = rgb_by_cam[cam]["rgb"]
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
    return rgb


def is_task_success(handle: EnvHandle) -> bool:
    task_env = handle.task_env
    return bool(getattr(task_env, "plan_success", True)) and bool(task_env.check_success())


def snapshot_robot_state(handle: EnvHandle) -> RobotState:
    """Capture EE poses + gripper values for no-op detection."""
    task_env = handle.task_env
    left_pose = np.asarray(task_env.get_arm_pose("left"), dtype=np.float64)
    right_pose = np.asarray(task_env.get_arm_pose("right"), dtype=np.float64)
    left_gripper = float(task_env.robot.get_left_gripper_val())
    right_gripper = float(task_env.robot.get_right_gripper_val())
    return RobotState(
        left_ee_pose=left_pose,
        right_ee_pose=right_pose,
        left_gripper=left_gripper,
        right_gripper=right_gripper,
    )
