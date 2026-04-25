"""RecordingContext — sim-ported from ADC, singleton that gathers per-step
state/images/labels and delegates to a DatasetRecorder.

Core differences vs ADC:
    - No AsyncCameraCapture. We pull RGB synchronously from task_env.cameras
      because SAPIEN render is deterministic per scene.step().
    - camera_latency_steps models real-world staleness via a deque buffer
      (see docs §7).
"""

from __future__ import annotations

import logging
import random
from collections import deque
from typing import Any

import numpy as np

from .labels import SubtaskEntry
from .recorder import DatasetRecorder

logger = logging.getLogger(__name__)


class RecordingContext:
    _recorder: DatasetRecorder | None = None
    _task_env: Any = None
    _cameras_cfg: list[dict] = []
    _is_active: bool = False
    _fps: int = 30
    _camera_latency_steps: int = 0
    _randomize_latency: bool = False

    _image_buffer: deque = deque(maxlen=1)

    # Active label state (updated from the replay driver)
    _current_subtask: SubtaskEntry | None = None
    _current_skill: dict | None = None
    _skill_start_step: int = 0
    _skill_total_steps: int = 1

    # Privileged features
    _privileged_actors: list[str] = []

    # Stats
    _recorded_frames: int = 0

    @classmethod
    def setup(
        cls,
        *,
        recorder: DatasetRecorder,
        task_env: Any,
        cameras_cfg: list[dict],
        fps: int = 30,
        camera_latency_steps: int = 0,
        randomize_latency: bool = False,
        privileged_actors: list[str] | None = None,
    ) -> None:
        cls._recorder = recorder
        cls._task_env = task_env
        cls._cameras_cfg = list(cameras_cfg)
        cls._fps = fps
        cls._camera_latency_steps = max(0, int(camera_latency_steps))
        cls._randomize_latency = bool(randomize_latency)
        cls._privileged_actors = list(privileged_actors or [])
        cls._image_buffer = deque(maxlen=cls._camera_latency_steps + 1)
        cls._is_active = True
        cls._recorded_frames = 0
        logger.info(
            "[RecordingContext] active fps=%d cameras=%s latency_steps=%d",
            fps, [c["name"] for c in cameras_cfg], cls._camera_latency_steps,
        )

    @classmethod
    def clear(cls) -> None:
        cls._recorder = None
        cls._task_env = None
        cls._is_active = False
        cls._current_subtask = None
        cls._current_skill = None
        cls._image_buffer = deque(maxlen=1)

    @classmethod
    def is_active(cls) -> bool:
        return cls._is_active

    # ---- label injection ----

    @classmethod
    def set_subtask(cls, entry: SubtaskEntry | None, target_position: np.ndarray | None = None) -> None:
        cls._current_subtask = entry
        if entry is not None:
            cls._current_skill = {
                "skill_type": entry.skill_type,
                "natural_language": entry.natural_language,
                "verification_question": entry.success_hint,
            }

    @classmethod
    def set_skill_window(cls, start_step: int, total_steps: int) -> None:
        cls._skill_start_step = int(start_step)
        cls._skill_total_steps = max(1, int(total_steps))

    # ---- per-step recording ----

    @classmethod
    def _read_images(cls) -> dict[str, np.ndarray]:
        if cls._task_env is None:
            return {}
        cls._task_env._update_render()
        cls._task_env.cameras.update_picture()
        rgb_by_cam = cls._task_env.cameras.get_rgb()
        # Legacy fallback for flat configs that don't carry an explicit
        # `source` field — the bi_so_follower recording_config always supplies
        # source from cam.source, so this branch is only hit for old configs.
        source_map = {
            "top": "head_camera",
            "left_wrist": "left_camera",
            "right_wrist": "right_camera",
        }
        # Key images by the canonical LeRobot feature key (group-aware) so
        # group-prefixed cameras with duplicate bare names (e.g. left_arm.wrist
        # and right_arm.wrist) don't collide on the dict.
        from closed_loop_cap.dataset.features import _camera_feature_key
        images: dict[str, np.ndarray] = {}
        for cam in cls._cameras_cfg:
            name = cam["name"]
            src = cam.get("source") or source_map.get(name, name)
            if src in rgb_by_cam and "rgb" in rgb_by_cam[src]:
                img = rgb_by_cam[src]["rgb"]
                if img.dtype != np.uint8:
                    img = (np.asarray(img) * 255.0).clip(0, 255).astype(np.uint8)
                images[_camera_feature_key(cam)] = img
        return images

    @classmethod
    def _buffered_images(cls) -> dict[str, np.ndarray]:
        """Apply camera_latency_steps via deque: return the oldest buffered
        frame set (i.e., latency_steps old)."""
        current = cls._read_images()
        cls._image_buffer.append(current)
        if cls._randomize_latency and cls._camera_latency_steps > 0:
            # Pick a random buffer index ∈ [0, len-1] each call.
            idx = random.randint(0, len(cls._image_buffer) - 1)
            return cls._image_buffer[idx]
        return cls._image_buffer[0]

    @classmethod
    def _collect_state(cls) -> dict:
        task_env = cls._task_env
        assert task_env is not None
        left = task_env.robot.get_left_arm_jointState()
        right = task_env.robot.get_right_arm_jointState()
        # left/right are lists: [arm_joints..., gripper_val]
        left_arm = np.asarray(left[:-1], dtype=np.float64)
        left_grip = float(left[-1])
        right_arm = np.asarray(right[:-1], dtype=np.float64)
        right_grip = float(right[-1])
        return {
            "left_arm_rad": left_arm,
            "left_gripper_01": left_grip,
            "right_arm_rad": right_arm,
            "right_gripper_01": right_grip,
        }

    @classmethod
    def _ee_pose(cls, side: str) -> np.ndarray:
        task_env = cls._task_env
        assert task_env is not None
        pose = task_env.get_arm_pose(side)
        # get_arm_pose returns [x, y, z, qw, qx, qy, qz] (or similar).
        # Convert to xyz + xyzw quaternion for dataset consistency.
        arr = np.asarray(pose, dtype=np.float32)
        if arr.shape == (7,):
            # RoboTwin returns wxyz; reshape to xyzw.
            x, y, z, qw, qx, qy, qz = arr
            return np.array([x, y, z, qx, qy, qz, qw], dtype=np.float32)
        return arr

    @classmethod
    def on_step(cls, step_idx: int) -> None:
        if not cls._is_active or cls._recorder is None:
            return
        try:
            state = cls._collect_state()
            # Action mirrors commanded state (closest available in sim — we
            # don't preserve the exact setpoint from trajectory planner).
            action = state
            ee_left = cls._ee_pose("left")
            ee_right = cls._ee_pose("right")
            images = cls._buffered_images()

            skill_info = None
            if cls._current_skill is not None:
                total = max(1, cls._skill_total_steps)
                progress = float(
                    min(1.0, max(0.0, (step_idx - cls._skill_start_step) / total))
                )
                skill_info = {**cls._current_skill, "progress": progress}

            subtask_info = None
            if cls._current_subtask is not None:
                subtask_info = {
                    "natural_language": cls._current_subtask.natural_language,
                    "object_name": cls._current_subtask.target_actor,
                    # target_position filled by the driver via privileged lookup; MVP = origin.
                    "target_position": [0.0, 0.0, 0.0],
                }

            privileged = None
            if cls._privileged_actors:
                from .privileged import collect_actor_poses
                privileged = collect_actor_poses(cls._task_env, cls._privileged_actors)

            cls._recorder.add_frame_native(
                left_arm_rad=state["left_arm_rad"],
                left_gripper_01=state["left_gripper_01"],
                right_arm_rad=state["right_arm_rad"],
                right_gripper_01=state["right_gripper_01"],
                action_left_arm_rad=action["left_arm_rad"],
                action_left_gripper_01=action["left_gripper_01"],
                action_right_arm_rad=action["right_arm_rad"],
                action_right_gripper_01=action["right_gripper_01"],
                ee_pose_left=ee_left,
                ee_pose_right=ee_right,
                images=images,
                skill_info=skill_info,
                subtask_info=subtask_info,
                privileged=privileged,
            )
            cls._recorded_frames += 1
        except Exception:  # noqa: BLE001 — never break replay for logging
            logger.exception("on_step failed at step=%d", step_idx)
