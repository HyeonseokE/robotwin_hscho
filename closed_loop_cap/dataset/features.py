"""LeRobot v3.0 features dict builder for RoboTwin replay logging.

Respects per-feature toggles from recording_config.yaml (see
docs/replay_dataset_logging.md §3 and configs/recording_config.yaml).
"""

from __future__ import annotations

from typing import Any


def _video_feature(height: int, width: int) -> dict:
    return {
        "dtype": "video",
        "shape": (height, width, 3),
        "names": ["height", "width", "channels"],
    }


def _float_feature(shape: tuple, names: list[str] | None = None) -> dict:
    f = {"dtype": "float32", "shape": shape}
    if names is not None:
        f["names"] = names
    return f


def _string_feature() -> dict:
    return {"dtype": "string", "shape": (1,)}


def _camera_feature_key(cam: dict) -> str:
    """Produce the LeRobot feature key for a camera.

    Follows the bi_so_follower convention:
      group == 'shared'  → observation.images.{name}
      group == 'left_arm'  → observation.images.left_{name}
      group == 'right_arm' → observation.images.right_{name}
    """
    group = cam.get("group", "shared")
    name = cam["name"]
    if group == "left_arm":
        return f"observation.images.left_{name}"
    if group == "right_arm":
        return f"observation.images.right_{name}"
    return f"observation.images.{name}"


def _gv(d: Any, key: str, default: bool = True) -> bool:
    """Get a boolean from a possibly-missing nested dict without crashing."""
    if not isinstance(d, dict):
        return default
    val = d.get(key, default)
    if isinstance(val, dict):
        # Nested — truthy if any child is truthy.
        return any(_gv(val, k, True) for k in val)
    return bool(val)


def build_features(
    *,
    left_arm_dim: int,
    right_arm_dim: int,
    cameras: list[dict],
    camera_height: int,
    camera_width: int,
    skill_features: dict | bool | None = True,
    subtask_features: dict | bool | None = True,
    observation_features: dict | None = None,
    privileged_actors: list[str] | None = None,
    privileged_features: dict | None = None,
) -> dict:
    """Compose the `features` dict passed to `LeRobotDataset.create()`.

    Args:
        cameras: list of {"name": str, "group": "shared"|"left_arm"|"right_arm"}.
        skill_features: either a bool (all-or-nothing) or a granular dict
            ``{natural_language, verification_question, type, progress,
            goal_position: {joint, left_ee, right_ee, gripper}}``.
        subtask_features: similarly bool or granular dict.
        observation_features: granular dict
            ``{state, action, ee_pose: {left, right}, images}``. Defaults
            to everything on.
        privileged_actors: actor names used to emit
            observation.oracle.object_pose.<actor> features.
        privileged_features: granular dict
            ``{enabled, object_pose, contact, ee_pose, table_height}``.
    """
    total_dim = left_arm_dim + 1 + right_arm_dim + 1
    joint_names = (
        [f"left_arm_{i}" for i in range(left_arm_dim)]
        + ["left_gripper"]
        + [f"right_arm_{i}" for i in range(right_arm_dim)]
        + ["right_gripper"]
    )

    obs_feats = observation_features or {}
    skill_toggle = skill_features if skill_features is not None else True
    subtask_toggle = subtask_features if subtask_features is not None else True

    features: dict = {}

    if _gv(obs_feats, "state", True):
        features["observation.state"] = _float_feature((total_dim,), joint_names)
    if _gv(obs_feats, "action", True):
        features["action"] = _float_feature((total_dim,), joint_names)

    ee_cfg = obs_feats.get("ee_pose", {}) if isinstance(obs_feats, dict) else {}
    if _gv(ee_cfg, "left", True):
        features["observation.ee_pose.left"] = _float_feature(
            (7,), ["x", "y", "z", "qx", "qy", "qz", "qw"]
        )
    if _gv(ee_cfg, "right", True):
        features["observation.ee_pose.right"] = _float_feature(
            (7,), ["x", "y", "z", "qx", "qy", "qz", "qw"]
        )

    if _gv(obs_feats, "images", True):
        for cam in cameras:
            features[_camera_feature_key(cam)] = _video_feature(
                camera_height, camera_width
            )

    # Skill features
    if skill_toggle:
        s = skill_toggle if isinstance(skill_toggle, dict) else {}
        gp = s.get("goal_position", {}) if isinstance(s, dict) else {}
        if _gv(s, "natural_language", True):
            features["skill.natural_language"] = _string_feature()
        if _gv(s, "type", True):
            features["skill.type"] = _string_feature()
        if _gv(s, "verification_question", True):
            features["skill.verification_question"] = _string_feature()
        if _gv(s, "progress", True):
            features["skill.progress"] = _float_feature((1,))
        if _gv(gp, "joint", True):
            features["skill.goal_position.joint"] = _float_feature(
                (total_dim,), joint_names
            )
        if _gv(gp, "left_ee", True):
            features["skill.goal_position.left_ee"] = _float_feature(
                (7,), ["x", "y", "z", "qx", "qy", "qz", "qw"]
            )
        if _gv(gp, "right_ee", True):
            features["skill.goal_position.right_ee"] = _float_feature(
                (7,), ["x", "y", "z", "qx", "qy", "qz", "qw"]
            )
        if _gv(gp, "gripper", True):
            features["skill.goal_position.gripper"] = _float_feature(
                (2,), ["left", "right"]
            )

    # Subtask features
    if subtask_toggle:
        t = subtask_toggle if isinstance(subtask_toggle, dict) else {}
        if _gv(t, "natural_language", True):
            features["subtask.natural_language"] = _string_feature()
        if _gv(t, "object_name", True):
            features["subtask.object_name"] = _string_feature()
        if _gv(t, "target_position", True):
            features["subtask.target_position"] = _float_feature(
                (3,), ["x", "y", "z"]
            )

    # Privileged oracle features (sim-only).
    pf = privileged_features or {}
    priv_master = bool(pf.get("enabled", bool(privileged_actors)))
    if priv_master and privileged_actors and _gv(pf, "object_pose", True):
        for actor in privileged_actors:
            features[f"observation.oracle.object_pose.{actor}"] = _float_feature(
                (7,), ["x", "y", "z", "qx", "qy", "qz", "qw"]
            )

    return features
