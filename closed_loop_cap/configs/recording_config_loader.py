"""Load recording_config.yaml and merge it into the pipeline config.

The closed_loop_cap pipeline config (default.yaml) carries both runtime
concerns (VLM, retries, env) and recording concerns (fps, cameras,
features). Users who want to control only recording can provide a
``recording_config.yaml`` that is merged on top of the pipeline config —
mirroring the ADC pipeline_config/recording_config_ws*.yaml workflow.

Mapping from recording_config.yaml → pipeline config:

    dataset_repo_id        → dataset.repo_id_template
    recording_fps          → logging.fps
    robot_type             → env.embodiment[0]       (informational only)
    use_videos             → dataset.use_videos
    image_writer_threads   → dataset.image_writer_threads
    include_failures       → logging.include_failures
    camera_latency_steps   → logging.camera_latency_steps
    randomize_latency      → logging.randomize_latency
    cameras                → logging.cameras (flattened)
    skill_features         → dataset.skill_features
    subtask_features       → dataset.subtask_features
    observation_features   → dataset.observation_features
    privileged_features    → logging.privileged_features (master)
                              + dataset.privileged_features (granular)
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_recording_config(path: str | Path) -> dict:
    """Load a raw recording config YAML."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _flatten_cameras(grouped: dict) -> list[dict]:
    """Turn {shared: [...], left_arm: [...], right_arm: [...]} into the flat
    list the existing DatasetRecorder expects.

    Each camera carries its group so feature naming can follow the
    bi_so_follower convention ``observation.images.<prefix><name>``.
    """
    out: list[dict] = []
    for group in ("shared", "left_arm", "right_arm"):
        for cam in grouped.get(group, []) or []:
            if not cam.get("enabled", True):
                continue
            out.append({
                "name": cam["name"],
                "source": cam.get("source") or cam.get("camera", cam["name"]),
                "group": group,
                "width": cam.get("width"),
                "height": cam.get("height"),
                "fps": cam.get("fps"),
            })
    return out


def merge_into_pipeline(pipeline_config: dict, recording: dict) -> dict:
    """Return a new config with recording settings merged in.

    Never mutates the input pipeline_config.
    """
    cfg = deepcopy(pipeline_config)
    cfg.setdefault("logging", {})
    cfg.setdefault("dataset", {})

    if "dataset_repo_id" in recording:
        cfg["dataset"]["repo_id_template"] = recording["dataset_repo_id"]
    if "recording_fps" in recording:
        cfg["logging"]["fps"] = int(recording["recording_fps"])
    if "use_videos" in recording:
        cfg["dataset"]["use_videos"] = bool(recording["use_videos"])
    if "image_writer_threads" in recording:
        cfg["dataset"]["image_writer_threads"] = int(recording["image_writer_threads"])
    if "include_failures" in recording:
        cfg["logging"]["include_failures"] = bool(recording["include_failures"])
    if "camera_latency_steps" in recording:
        cfg["logging"]["camera_latency_steps"] = int(recording["camera_latency_steps"])
    if "randomize_latency" in recording:
        cfg["logging"]["randomize_latency"] = bool(recording["randomize_latency"])

    if "cameras" in recording and isinstance(recording["cameras"], dict):
        cfg["logging"]["cameras"] = _flatten_cameras(recording["cameras"])

    for key in ("skill_features", "subtask_features", "observation_features"):
        if key in recording:
            cfg["dataset"][key] = recording[key]

    priv = recording.get("privileged_features")
    if isinstance(priv, dict):
        cfg["logging"]["privileged_features"] = bool(priv.get("enabled", False))
        cfg["dataset"]["privileged_features"] = priv
    elif priv is not None:
        cfg["logging"]["privileged_features"] = bool(priv)

    if "robot_type" in recording:
        cfg.setdefault("env", {})["_robot_type_hint"] = recording["robot_type"]

    # Perturbation — mirrors perturbation.subgoal in the pipeline config so
    # recording sessions can self-describe their perturbation regime.
    pert = recording.get("perturbation")
    if isinstance(pert, dict):
        cfg.setdefault("perturbation", {})
        sub = pert.get("subgoal")
        if isinstance(sub, dict):
            cfg["perturbation"]["subgoal"] = {
                **cfg.get("perturbation", {}).get("subgoal", {}),
                **sub,
            }

    return cfg


def load_and_merge(
    pipeline_config: dict, recording_path: str | Path,
) -> dict:
    """Convenience: load YAML and merge in one call."""
    rec = load_recording_config(recording_path)
    return merge_into_pipeline(pipeline_config, rec)
