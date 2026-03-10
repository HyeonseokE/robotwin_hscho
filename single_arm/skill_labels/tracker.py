"""
SkillLabelTracker — records per-frame skill-level semantic labels during trajectory execution.

Each self.move() call in a task's play_once() corresponds to one skill segment.
The tracker assigns 8 labels to every frame captured by _take_picture().

Language labels are generated from structured templates + slot values,
ensuring consistent formatting across all tasks.
"""

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Structured templates: {slot} values are filled by task code or fallback
# ---------------------------------------------------------------------------

SUBTASK_TEMPLATES = {
    "grasp":                "grasp {object}",
    "place":                "place {object} on {target}",
    "move_to_pose":         "move to {target}",
    "move_by_displacement": "move {object} {direction}",
    "close_gripper":        "grip {object}",
    "open_gripper":         "release {object}",
    "back_to_origin":       "return arm to home position",
    "together_move":        "move both arms to target poses",
    "unknown":              "execute robot action",
}

VERIFICATION_TEMPLATES = {
    "grasp":                "is {object} grasped?",
    "place":                "is {object} placed on {target}?",
    "move_to_pose":         "has arm reached {target}?",
    "move_by_displacement": "has {object} moved {direction}?",
    "close_gripper":        "is {object} gripped?",
    "open_gripper":         "is {object} released?",
    "back_to_origin":       "is arm at home position?",
    "together_move":        "have both arms reached target poses?",
    "unknown":              "is action completed?",
}


def _render_template(templates: dict, skill_type: str, slots: dict) -> str:
    """Render a template string with the given slot values."""
    template = templates.get(skill_type, templates["unknown"])
    try:
        return template.format(**slots)
    except KeyError:
        # If a slot is missing, fill with defaults
        defaults = {"object": "object", "target": "target position",
                    "direction": "to target", "arm": "robot"}
        merged = {**defaults, **slots}
        return template.format(**merged)


class SkillLabelTracker:
    """Tracks skill boundaries and generates per-frame label dicts."""

    def __init__(self):
        self.recording_enabled: bool = False
        self.reset_episode()

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset_episode(self):
        """Call at the start of each episode (or in __init__)."""
        self._skill_index = -1  # incremented on start_skill_segment
        self._pending_skill: Optional[dict] = None  # registered but not yet started
        self._active_skill: Optional[dict] = None   # currently executing

        # Per-segment state
        self._start_joint_positions: Optional[np.ndarray] = None
        self._start_ee_pose: Optional[np.ndarray] = None
        self._goal_joint_positions: Optional[np.ndarray] = None
        self._goal_ee_pose: Optional[np.ndarray] = None
        self._segment_frame_count: int = 0
        self._segment_total_frames: int = 0  # estimated, updated lazily

    # ------------------------------------------------------------------
    # Skill registration (called from task code or primitive fallback)
    # ------------------------------------------------------------------

    def register_skill(
        self,
        skill_type: str,
        arm_tag: str = "",
        target_object_name: str = "",
        goal_gripper_position: float = -1.0,
        target: str = "",
        direction: str = "",
    ):
        """Register a skill that is about to execute.

        Called either explicitly (from task code, inserted by add_skill_labels.py)
        or as a fallback (from primitive methods in _base_task.py).

        Template slots are filled from the provided arguments:
          {object}    <- target_object_name
          {target}    <- target (e.g., "basket", "table")
          {direction} <- direction (e.g., "upward", "forward")
          {arm}       <- arm_tag
        """
        if not self.recording_enabled:
            return

        slots = {
            "object": target_object_name or "object",
            "target": target or "target position",
            "direction": direction or "to target",
            "arm": arm_tag or "robot",
        }

        language_subtask = _render_template(SUBTASK_TEMPLATES, skill_type, slots)
        verification_question = _render_template(VERIFICATION_TEMPLATES, skill_type, slots)

        self._pending_skill = {
            "skill_type": skill_type,
            "arm_tag": arm_tag,
            "target_object_name": target_object_name,
            "goal_gripper_position": goal_gripper_position,
            "language_subtask": language_subtask,
            "verification_question": verification_question,
        }

    def has_pending_skill(self) -> bool:
        """True if register_skill() was called but start_skill_segment() hasn't consumed it yet."""
        return self._pending_skill is not None

    # ------------------------------------------------------------------
    # Segment lifecycle (called from move())
    # ------------------------------------------------------------------

    def start_skill_segment(
        self,
        start_joint_positions: np.ndarray,
        start_ee_pose: np.ndarray,
    ):
        """Called at the entry of move(). Promotes pending skill to active."""
        if not self.recording_enabled:
            return
        if self._pending_skill is None:
            return

        self._skill_index += 1
        self._active_skill = self._pending_skill
        self._pending_skill = None

        self._start_joint_positions = np.array(start_joint_positions, dtype=np.float32)
        self._start_ee_pose = np.array(start_ee_pose, dtype=np.float32)
        self._goal_joint_positions = self._start_joint_positions.copy()
        self._goal_ee_pose = self._start_ee_pose.copy()
        self._segment_frame_count = 0
        self._segment_total_frames = 0

    def update_goal(
        self,
        goal_joint_positions: Optional[np.ndarray] = None,
        goal_ee_pose: Optional[np.ndarray] = None,
    ):
        """Update goal state during move() execution."""
        if not self.recording_enabled or self._active_skill is None:
            return
        if goal_joint_positions is not None:
            self._goal_joint_positions = np.array(goal_joint_positions, dtype=np.float32)
        if goal_ee_pose is not None:
            self._goal_ee_pose = np.array(goal_ee_pose, dtype=np.float32)

    def finish_skill_segment(self):
        """Called at exit of move(). Finalises the active skill segment."""
        if not self.recording_enabled:
            return
        self._segment_total_frames = max(self._segment_frame_count, 1)
        # Keep active_skill alive so remaining frames within this segment
        # can still be labelled (the next start_skill_segment will replace it).

    # ------------------------------------------------------------------
    # Per-frame label capture
    # ------------------------------------------------------------------

    def capture_frame_labels(self, current_joint_positions) -> dict:
        """Return a dict of 8 label values for the current frame.

        Called from _take_picture() on every saved frame.
        """
        if self._active_skill is None:
            return {
                "skill_type": "idle",
                "language_subtask": "idle",
                "verification_question": "",
                "skill_index": self._skill_index if self._skill_index >= 0 else 0,
                "progress": 0.0,
                "goal_ee_pose": np.zeros(7, dtype=np.float32),
                "goal_joint_positions": np.zeros(6, dtype=np.float32),
                "goal_gripper_position": -1.0,
            }

        self._segment_frame_count += 1
        progress = self._compute_progress(current_joint_positions)

        goal_ee = self._goal_ee_pose if self._goal_ee_pose is not None else np.zeros(7, dtype=np.float32)
        goal_jp = self._goal_joint_positions if self._goal_joint_positions is not None else np.zeros(6, dtype=np.float32)

        return {
            "skill_type": self._active_skill["skill_type"],
            "language_subtask": self._active_skill["language_subtask"],
            "verification_question": self._active_skill["verification_question"],
            "skill_index": self._skill_index,
            "progress": float(np.clip(progress, 0.0, 1.0)),
            "goal_ee_pose": goal_ee.copy(),
            "goal_joint_positions": goal_jp.copy(),
            "goal_gripper_position": float(self._active_skill.get("goal_gripper_position", -1.0)),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_progress(self, current_joint_positions) -> float:
        """Estimate progress within the current skill segment.

        Uses joint-space distance: progress = 1 - (remaining / total).
        Falls back to frame-count ratio when distances are tiny.
        """
        if self._start_joint_positions is None or self._goal_joint_positions is None:
            return 0.0

        current = np.array(current_joint_positions[:6], dtype=np.float32)
        total_dist = float(np.linalg.norm(self._goal_joint_positions - self._start_joint_positions))

        if total_dist < 1e-4:
            # Gripper-only or tiny movement — use frame count ratio
            if self._segment_total_frames > 0:
                return min(self._segment_frame_count / self._segment_total_frames, 1.0)
            return 0.0

        remaining_dist = float(np.linalg.norm(self._goal_joint_positions - current))
        progress = 1.0 - remaining_dist / total_dist
        return max(progress, 0.0)
