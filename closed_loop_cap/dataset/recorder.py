"""DatasetRecorder — thin wrapper around LeRobotDataset that applies
ADC-normalization at write time and exposes an ADC-style frame API.

See docs/replay_dataset_logging.md §4-§8.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import LimitsCache, to_normalized
from .features import build_features

logger = logging.getLogger(__name__)


class DatasetRecorder:
    """One recorder instance owns one LeRobotDataset root.

    Typical lifecycle:
        rec = DatasetRecorder(root, repo_id="robotwin/<task>", fps=30, ...)
        rec.create(task_env)           # introspects qlimits, builds features
        rec.start_episode(task_desc)
        for each replay step:
            rec.add_frame(frame_dict)
        rec.end_episode(success=True, extras={"seed": 7, "trial": 1, ...})
        rec.finalize()
    """

    def __init__(
        self,
        *,
        root: Path,
        repo_id: str,
        fps: int,
        cameras: list[dict],
        camera_height: int,
        camera_width: int,
        robot_type: str = "robotwin_aloha",
        use_videos: bool = True,
        skill_features: dict | bool = True,
        subtask_features: dict | bool = True,
        observation_features: dict | None = None,
        privileged_actors: list[str] | None = None,
        privileged_features: dict | None = None,
        image_writer_threads: int = 4,
    ) -> None:
        # Ensure LeRobot is importable (it lives under AutoDataCollector/lerobot/src).
        self._ensure_lerobot_path()

        self.root = Path(root)
        self.repo_id = repo_id
        self.fps = fps
        self.cameras = cameras
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.robot_type = robot_type
        self.use_videos = use_videos
        self.skill_features = skill_features
        self.subtask_features = subtask_features
        self.observation_features = observation_features or {}
        self.privileged_actors = list(privileged_actors or [])
        self.privileged_features = privileged_features or {}
        self.image_writer_threads = image_writer_threads

        self.limits = LimitsCache()
        self._dataset = None
        self._features: dict | None = None
        self._episode_count = 0
        self._total_frames = 0
        self._current_task: str | None = None
        self._is_recording = False
        self._frame_count = 0

    @staticmethod
    def _ensure_lerobot_path() -> None:
        candidates = [
            Path("/workspace/AutoDataCollector/lerobot/src"),
        ]
        for p in candidates:
            if p.is_dir() and str(p) not in sys.path:
                sys.path.insert(0, str(p))

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def dataset(self):
        return self._dataset

    def create(self, task_env: Any) -> None:
        """Introspect URDF limits from the live env, build features, and
        initialize the underlying LeRobotDataset on disk."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.limits.capture(task_env)
        self._features = build_features(
            left_arm_dim=self.limits.left_arm_dim,
            right_arm_dim=self.limits.right_arm_dim,
            cameras=self.cameras,
            camera_height=self.camera_height,
            camera_width=self.camera_width,
            skill_features=self.skill_features,
            subtask_features=self.subtask_features,
            observation_features=self.observation_features,
            privileged_actors=self.privileged_actors,
            privileged_features=self.privileged_features,
        )
        # Do NOT mkdir root — LeRobotDataset.create() does it with exist_ok=False.
        # If root already exists (leftover from a prior failed run), remove it.
        if self.root.exists():
            import shutil
            shutil.rmtree(self.root)
        self._dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            features=self._features,
            root=self.root,
            robot_type=self.robot_type,
            use_videos=self.use_videos,
            image_writer_threads=self.image_writer_threads,
        )
        # Stash calibration in info meta for self-describing datasets.
        info = getattr(self._dataset.meta, "info", {})
        if isinstance(info, dict):
            info.setdefault("calibration_limits", self.limits.to_dict())

    def start_episode(self, task: str) -> None:
        if self._is_recording:
            raise RuntimeError("already recording; end_episode() first")
        self._current_task = task
        self._is_recording = True
        self._frame_count = 0
        logger.info("[DatasetRecorder] start_episode task=%r", task)

    def add_frame_native(
        self,
        *,
        left_arm_rad: np.ndarray,
        left_gripper_01: float,
        right_arm_rad: np.ndarray,
        right_gripper_01: float,
        action_left_arm_rad: np.ndarray,
        action_left_gripper_01: float,
        action_right_arm_rad: np.ndarray,
        action_right_gripper_01: float,
        ee_pose_left: np.ndarray,
        ee_pose_right: np.ndarray,
        images: dict[str, np.ndarray],
        skill_info: dict | None = None,
        subtask_info: dict | None = None,
        privileged: dict | None = None,
    ) -> None:
        """Normalize native qpos/gripper, then push a frame to the dataset."""
        if not self._is_recording:
            raise RuntimeError("not recording; start_episode() first")

        state_norm = to_normalized(
            left_arm_rad, left_gripper_01, right_arm_rad, right_gripper_01, self.limits,
        )
        action_norm = to_normalized(
            action_left_arm_rad, action_left_gripper_01,
            action_right_arm_rad, action_right_gripper_01,
            self.limits,
        )

        frame: dict = {
            "observation.state": state_norm,
            "action": action_norm,
            "observation.ee_pose.left": np.asarray(ee_pose_left, dtype=np.float32),
            "observation.ee_pose.right": np.asarray(ee_pose_right, dtype=np.float32),
            "task": self._current_task or "",
        }

        from closed_loop_cap.dataset.features import _camera_feature_key
        feat_keys = set(self._features or {})
        for cam in self.cameras:
            key = _camera_feature_key(cam)
            if key not in feat_keys:
                continue
            img = images.get(key)
            if img is None:
                # Dummy if camera missing, to keep LeRobot schema intact.
                img = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
            elif img.dtype != np.uint8:
                img = (np.asarray(img) * 255.0).clip(0, 255).astype(np.uint8)
            frame[key] = img

        # Skill / subtask labels — only write fields the schema has. features.py
        # drops keys whose recording_config.yaml toggle is false, so we mirror
        # that here instead of maintaining a second copy of the toggle logic.
        def _w(key: str, value) -> None:
            if key in feat_keys:
                frame[key] = value

        si = skill_info or {}
        _w("skill.natural_language", si.get("natural_language", ""))
        _w("skill.type", si.get("skill_type", ""))
        _w("skill.verification_question", si.get("verification_question", ""))
        _w("skill.progress",
           np.array([float(si.get("progress", 0.0))], dtype=np.float32))
        _w("skill.goal_position.joint",
           np.asarray(si.get("goal_joint", state_norm), dtype=np.float32))
        _w("skill.goal_position.left_ee",
           np.asarray(si.get("goal_left_ee", ee_pose_left), dtype=np.float32))
        _w("skill.goal_position.right_ee",
           np.asarray(si.get("goal_right_ee", ee_pose_right), dtype=np.float32))
        _w("skill.goal_position.gripper",
           np.asarray(
               si.get("goal_gripper",
                      [left_gripper_01 * 100.0, right_gripper_01 * 100.0]),
               dtype=np.float32,
           ))

        ti = subtask_info or {}
        _w("subtask.natural_language", ti.get("natural_language", ""))
        _w("subtask.object_name", ti.get("object_name", ""))
        _w("subtask.target_position",
           np.asarray(ti.get("target_position", [0.0, 0.0, 0.0]),
                      dtype=np.float32))

        if privileged:
            for actor, pose7 in privileged.items():
                key = f"observation.oracle.object_pose.{actor}"
                if key in (self._features or {}):
                    frame[key] = np.asarray(pose7, dtype=np.float32)

        self._dataset.add_frame(frame)
        self._frame_count += 1
        self._total_frames += 1

    def end_episode(self, *, extras: dict | None = None) -> None:
        """Finalize the current episode to disk. `extras` are merged into the
        meta/episodes.json custom fields (seed, trial, success, ...)."""
        if not self._is_recording:
            raise RuntimeError("not recording")

        if self._frame_count == 0:
            logger.warning("[DatasetRecorder] ending empty episode; discarding")
            try:
                self._dataset.clear_episode_buffer()
            except Exception:  # noqa: BLE001
                pass
            self._is_recording = False
            self._current_task = None
            return

        try:
            self._dataset.save_episode()
            # Write extras (seed, trial, success, ...) to a side-car jsonl
            # since LeRobot's episode_buffer doesn't accept custom keys.
            if extras:
                self._append_episode_extras(self._episode_count, extras)
            self._episode_count += 1
        finally:
            self._is_recording = False
            self._current_task = None
            self._frame_count = 0

    def _append_episode_extras(self, episode_index: int, extras: dict) -> None:
        """Persist per-episode custom metadata to a side-car JSONL file."""
        import json as _json
        extras_path = self.root / "meta" / "episode_extras.jsonl"
        extras_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"episode_index": episode_index, **extras}
        with extras_path.open("a", encoding="utf-8") as f:
            f.write(_json.dumps(record, default=str) + "\n")

    def finalize(self) -> None:
        if self._dataset is None:
            return
        try:
            self._dataset.finalize()
        except Exception:  # noqa: BLE001 — finalize is best-effort on abort paths
            logger.exception("[DatasetRecorder] finalize failed")

    def push_to_hub(
        self,
        *,
        tags: list[str] | None = None,
        private: bool = False,
        license: str = "apache-2.0",
    ) -> None:
        """Publish the dataset to Hugging Face Hub under `self.repo_id`.

        Requires `huggingface_hub` auth configured (HF_TOKEN or `huggingface-cli
        login`). finalize() must have run first.
        """
        if self._dataset is None:
            raise RuntimeError("dataset not created yet")
        if self._is_recording:
            raise RuntimeError("end current episode before push_to_hub()")
        default_tags = ["robotwin", "closed-loop-cap", "sim"]
        self._dataset.push_to_hub(
            tags=list({*default_tags, *(tags or [])}),
            private=private,
            license=license,
        )
        logger.info("[DatasetRecorder] pushed %s to hub", self.repo_id)
