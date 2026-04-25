"""Phase-2 replay pass — streams LeRobot v3.0 frames from a stored traj.pkl.

Reads:
    output/datasets/<task>/logs/seed_<N>/trial_<KKK>/traj.pkl
    output/datasets/<task>/logs/seed_<N>/trial_<KKK>/subtask_timeline.json

Writes:
    output/datasets/<task>/recorded_data/**         (LeRobot dataset root)
    output/datasets/<task>/logs/.../replay.mp4      (per-episode slice of top cam mp4)

Usage:
    python -m closed_loop_cap.run_replay \
        --task beat_block_hammer --seeds 1 2 3 --trials 1 \
        --include-failures --privileged
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from closed_loop_cap.dataset import (  # noqa: E402
    DatasetRecorder,
    RecordingContext,
    SubtaskTimeline,
)
from closed_loop_cap.env.task_env import (  # noqa: E402
    close_env,
    make_env,
    replay_trajectory,
)
from closed_loop_cap.task_registry import load_task_meta  # noqa: E402

logger = logging.getLogger("closed_loop_cap.replay")


def _load_traj(pkl_path: Path) -> tuple[list, list]:
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    return data.get("left_joint_path", []), data.get("right_joint_path", [])


def _load_report(ep_dir: Path) -> dict | None:
    rp = ep_dir / "report.json"
    if not rp.is_file():
        return None
    try:
        import json
        return json.loads(rp.read_text())
    except Exception:  # noqa: BLE001
        return None


def _camera_cfg(config: dict) -> tuple[list[dict], int, int]:
    """Resolve which cameras to log + their (H, W). Defaults match docs §9.1."""
    cams = (
        config.get("logging", {}).get("cameras")
        or [
            {"name": "top", "source": "head_camera"},
            {"name": "left_wrist", "source": "left_camera"},
            {"name": "right_wrist", "source": "right_camera"},
        ]
    )
    # D435 default 320x240; Large_D435 → 640x480.
    cam_type = config.get("env", {}).get("camera", {}).get("head_camera_type", "D435")
    size = {"D435": (320, 240), "Large_D435": (640, 480)}.get(cam_type, (320, 240))
    width, height = size
    return cams, height, width


def _export_per_trial_replays(
    *,
    dataset_root: Path,
    logs_root: Path,
    video_key: str = "observation.images.top",
) -> None:
    """Slice each finalized episode out of the chunk-level mp4 into the matching
    ``logs/seed_<S>/trial_<T>/replay.mp4``.

    LeRobot v3.0 concatenates many episodes into one mp4 per chunk
    (``videos/<key>/chunk-NNN/file-NNN.mp4``) and records per-episode
    ``from_timestamp`` / ``to_timestamp`` in ``meta/episodes/...parquet``.
    The previous hardlink scheme exposed the *whole* chunk mp4 to every trial,
    so all replays looked identical. Here we use ffmpeg ``-c copy`` to write
    one mp4 per trial without re-encoding.
    """
    extras_path = dataset_root / "meta" / "episode_extras.jsonl"
    if not extras_path.is_file():
        logger.warning("replay export: missing %s", extras_path)
        return

    ep_to_trial: dict[int, tuple[int, int]] = {}
    with extras_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ei, seed, trial = rec.get("episode_index"), rec.get("seed"), rec.get("trial")
            if ei is None or seed is None or trial is None:
                continue
            ep_to_trial[int(ei)] = (int(seed), int(trial))

    if not ep_to_trial:
        logger.warning("replay export: no episode_extras records to slice")
        return

    try:
        import pyarrow.parquet as _pq
    except ImportError:
        logger.warning("replay export: pyarrow unavailable; cannot read episode metadata")
        return

    chunk_col = f"videos/{video_key}/chunk_index"
    file_col = f"videos/{video_key}/file_index"
    from_col = f"videos/{video_key}/from_timestamp"
    to_col = f"videos/{video_key}/to_timestamp"
    needed = ["episode_index", chunk_col, file_col, from_col, to_col]

    rows: list[dict] = []
    for parquet_path in sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        try:
            rows.extend(_pq.read_table(parquet_path, columns=needed).to_pylist())
        except Exception:  # noqa: BLE001
            logger.exception("replay export: failed to read %s", parquet_path)

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        logger.warning("replay export: ffmpeg not on PATH; skipping per-trial slicing")
        return

    written = 0
    for row in rows:
        ei = row.get("episode_index")
        if ei is None or int(ei) not in ep_to_trial:
            continue
        seed, trial = ep_to_trial[int(ei)]
        chunk, file_idx = row.get(chunk_col), row.get(file_col)
        t0, t1 = row.get(from_col), row.get(to_col)
        if chunk is None or file_idx is None or t0 is None or t1 is None:
            logger.warning("replay export: incomplete metadata for episode %d", ei)
            continue

        src = (
            dataset_root / "videos" / video_key
            / f"chunk-{int(chunk):03d}" / f"file-{int(file_idx):03d}.mp4"
        )
        if not src.is_file():
            logger.warning("replay export: missing source %s", src)
            continue

        duration = max(0.0, float(t1) - float(t0))
        if duration <= 0.0:
            logger.warning("replay export: non-positive duration for episode %d", ei)
            continue

        dst = logs_root / f"seed_{seed}" / f"trial_{trial:03d}" / "replay.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Drop existing hardlinks/symlinks from the legacy export path.
        if dst.is_symlink() or dst.exists():
            dst.unlink()

        cmd = [
            ffmpeg_bin, "-loglevel", "error", "-y",
            "-ss", f"{float(t0):.6f}",
            "-i", str(src),
            "-t", f"{duration:.6f}",
            "-c", "copy", "-an",
            "-avoid_negative_ts", "make_zero",
            str(dst),
        ]
        try:
            subprocess.run(cmd, check=True)
            written += 1
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "replay export: ffmpeg failed for episode %d (seed=%d trial=%d): %s",
                ei, seed, trial, exc,
            )

    logger.info("replay export: wrote %d per-trial replay.mp4 file(s)", written)


def _replay_one(
    *,
    task_name: str,
    seed: int,
    trial: int,
    config: dict,
    recorder: DatasetRecorder,
    include_failures: bool,
    session: str,
) -> bool:
    from closed_loop_cap.paths import trial_dir as _trial_dir  # local to avoid cycle
    output_dir = REPO_ROOT / config.get("output_dir", "closed_loop_cap/output")
    ep_dir = _trial_dir(output_dir, session, task_name, seed, trial)
    traj_pkl = ep_dir / "traj.pkl"
    if not traj_pkl.is_file():
        logger.warning("missing traj.pkl at %s; skip", traj_pkl)
        return False

    report = _load_report(ep_dir) or {}
    success = bool(report.get("success", False))
    if not success and not include_failures:
        logger.info("skip failure (success=false) %s seed=%d trial=%d", task_name, seed, trial)
        return False

    left_path, right_path = _load_traj(traj_pkl)
    if not left_path and not right_path:
        logger.warning("empty traj at %s; skip", traj_pkl)
        return False

    meta = load_task_meta(task_name)
    task_description = meta.description or f"Complete the {task_name} task."

    handle = None
    try:
        handle = make_env(
            task_name, seed, config,
            episode_mp4_path=None,  # LeRobot handles video; no ffmpeg tee needed.
            save_subdir=(
                f"datasets/{session}/{task_name}/logs/"
                f"seed_{seed}/trial_{trial:03d}/replay"
            ),
        )

        if recorder.dataset is None:
            recorder.create(handle.task_env)

        recorder.start_episode(task_description)

        # Preload timeline so label injection can happen per step.
        timeline_path = ep_dir / "subtask_timeline.json"
        timeline = (
            SubtaskTimeline.load(timeline_path)
            if timeline_path.is_file() else SubtaskTimeline()
        )
        approx_total_steps = sum(
            seg["position"].shape[0] for seg in left_path if seg and seg.get("status") == "Success"
        ) or sum(
            seg["position"].shape[0] for seg in right_path if seg and seg.get("status") == "Success"
        ) or 1
        timeline.assign_step_ranges(approx_total_steps)

        cameras_cfg, cam_h, cam_w = _camera_cfg(config)
        logging_cfg = config.get("logging", {})
        privileged_actors = (
            list(meta.actor_names) if logging_cfg.get("privileged_features", False) else []
        )
        RecordingContext.setup(
            recorder=recorder,
            task_env=handle.task_env,
            cameras_cfg=cameras_cfg,
            fps=int(logging_cfg.get("fps", 30)),
            camera_latency_steps=int(logging_cfg.get("camera_latency_steps", 0)),
            randomize_latency=bool(logging_cfg.get("randomize_latency", False)),
            privileged_actors=privileged_actors,
        )

        _active_id = {"id": None}

        def step_cb(step_idx: int, segment_index: int = 0) -> None:
            entry = timeline.resolve(step_idx)
            if entry is not None and entry.subtask_id != _active_id["id"]:
                RecordingContext.set_subtask(entry)
                RecordingContext.set_skill_window(
                    start_step=entry.step_start,
                    total_steps=max(1, entry.step_end - entry.step_start),
                )
                _active_id["id"] = entry.subtask_id
            RecordingContext.on_step(step_idx)

        replay_trajectory(handle, left_path, right_path, step_callback=step_cb)

        recorder.end_episode(
            extras={
                "seed": seed,
                "trial": trial,
                "success": success,
                "abort_reason": report.get("abort_reason"),
                "num_subtasks": report.get("num_subtasks", 0),
                "task_name": task_name,
            },
        )
        RecordingContext.clear()
        # Per-trial replay.mp4 is written once at the end of main(), after
        # recorder.finalize() has flushed the chunk mp4 + episode metadata.
        return True
    finally:
        if handle is not None:
            close_env(handle)


def _iter_trials(root: Path, seeds: list[int] | None, trials: list[int] | None) -> list[tuple[int, int]]:
    """Walk logs/ for (seed, trial) pairs that have traj.pkl."""
    pairs: list[tuple[int, int]] = []
    if not root.is_dir():
        return pairs
    for seed_dir in sorted(root.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        seed = int(seed_dir.name.removeprefix("seed_"))
        if seeds is not None and seed not in seeds:
            continue
        for trial_dir in sorted(seed_dir.iterdir()):
            if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
                continue
            trial = int(trial_dir.name.removeprefix("trial_"))
            if trials is not None and trial not in trials:
                continue
            if (trial_dir / "traj.pkl").is_file():
                pairs.append((seed, trial))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Closed-loop CaP replay → LeRobot v3.0")
    parser.add_argument("--task", required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="optional filter")
    parser.add_argument("--trials", type=int, nargs="*", default=None, help="optional filter (1-indexed)")
    parser.add_argument("--session", required=True,
                        help="Session id (directory under output/datasets/).")
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "closed_loop_cap" / "configs" / "default.yaml")
    )
    parser.add_argument(
        "--recording-config", default=None,
        help=("Optional recording_config.yaml to override feature / camera / "
              "dataset settings. Mirrors ADC pipeline_config/recording_config_ws*.yaml."),
    )
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--privileged", action="store_true")
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose >= 2 else logging.INFO if args.verbose == 1 else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Same loader as run_closed_loop: auto-overlays paid_api_config.yaml +
    # recording_config.yaml from the main config's directory. An explicit
    # --recording-config wins on top.
    from closed_loop_cap.run_closed_loop import _load_config
    config = _load_config(Path(args.config))
    if args.recording_config:
        from closed_loop_cap.configs.recording_config_loader import load_and_merge
        config = load_and_merge(config, args.recording_config)
    if args.privileged:
        config.setdefault("logging", {})["privileged_features"] = True

    from closed_loop_cap.paths import logs_root as _logs_root, recorded_data_dir
    output_dir = REPO_ROOT / config.get("output_dir", "closed_loop_cap/output")
    logs_root = _logs_root(output_dir, args.session, args.task)
    pairs = _iter_trials(logs_root, args.seeds, args.trials)
    if not pairs:
        print(f"no (seed, trial) pairs with traj.pkl under {logs_root}")
        return 1

    cameras_cfg, cam_h, cam_w = _camera_cfg(config)
    task_meta = load_task_meta(args.task)
    privileged_actors = (
        list(task_meta.actor_names)
        if config.get("logging", {}).get("privileged_features", False) else []
    )
    ds_cfg = config.get("dataset", {})
    recorder = DatasetRecorder(
        root=recorded_data_dir(output_dir, args.session, args.task),
        repo_id=ds_cfg.get("repo_id_template", "robotwin/{task}").format(task=args.task),
        fps=int(config.get("logging", {}).get("fps", 30)),
        cameras=cameras_cfg,
        camera_height=cam_h,
        camera_width=cam_w,
        privileged_actors=privileged_actors,
        use_videos=bool(ds_cfg.get("use_videos", True)),
        image_writer_threads=int(ds_cfg.get("image_writer_threads", 4)),
        skill_features=ds_cfg.get("skill_features", True),
        subtask_features=ds_cfg.get("subtask_features", True),
        observation_features=ds_cfg.get("observation_features"),
        privileged_features=ds_cfg.get("privileged_features"),
    )

    ok = 0
    try:
        for seed, trial in pairs:
            try:
                ran = _replay_one(
                    task_name=args.task, seed=seed, trial=trial,
                    config=config, recorder=recorder,
                    include_failures=args.include_failures,
                    session=args.session,
                )
                if ran:
                    ok += 1
            except Exception:  # noqa: BLE001
                logger.exception("replay crashed for seed=%d trial=%d", seed, trial)
    finally:
        recorder.finalize()

    _export_per_trial_replays(
        dataset_root=Path(recorder.root),
        logs_root=logs_root,
    )

    print(f"replay: {ok}/{len(pairs)} episodes recorded")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
