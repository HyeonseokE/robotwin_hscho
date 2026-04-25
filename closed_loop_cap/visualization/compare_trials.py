"""CLI: Compare EE trajectories between two or more datasets.

A *dataset* here is a named group of trials that should render in a single
colour (e.g. all trials collected with perturbation OFF vs. all trials with
perturbation ON).  Every trajectory inside a dataset is drawn with the same
colour so the visual contrast is between groups.

Dataset spec format: ``NAME[=COLOR][@SESSION]:TRIAL1,TRIAL2,...``

Usage:
    # Two datasets from different sessions (typical OFF vs ON comparison)
    python -m closed_loop_cap.visualization.compare_trials \\
        --task beat_block_hammer --seed 12 \\
        --dataset "OFF=blue@20260416_120000:1,2,3,4,5" \\
        --dataset "ON=red@20260416_130000:1,2,3,4,5"

    # Shared --session fallback; per-spec @SESSION wins when present
    python -m closed_loop_cap.visualization.compare_trials \\
        --task beat_block_hammer --seed 12 \\
        --session 20260416_120000 \\
        --dataset "OFF=blue:1,2,3" \\
        --dataset "ON=red@20260416_130000:1,2,3"
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from closed_loop_cap.paths import seed_logs_dir, viz_dir
from closed_loop_cap.visualization.traj_loader import concat_positions, load_traj
from closed_loop_cap.visualization.fk import (
    compute_ee_from_joint_path,
    get_reference_points,
)
from closed_loop_cap.visualization.plot_ee import (
    Dataset,
    DEFAULT_COLORS,
    plot_dataset_comparison_3d,
    plot_dataset_comparison_3d_html,
)


@dataclass
class _DatasetSpec:
    """Parsed CLI dataset spec.

    ``trials`` is ``None`` when the user wrote ``*`` / ``all`` — in that case
    the trial list is discovered at load time by walking the session directory.
    """
    label: str
    color: str
    session: str
    trials: list[int] | None = None


def _parse_dataset_spec(
    spec: str,
    fallback_color: str,
    fallback_session: str | None,
) -> _DatasetSpec:
    """Parse ``NAME[=COLOR][@SESSION]:TRIAL1,TRIAL2,...``.

    Examples:
        ``OFF:1,2,3``                        → label=OFF, color=fallback,
                                                session=fallback, trials=[1,2,3]
        ``ON=red:4,5,6``                     → label=ON,  color=red,
                                                session=fallback
        ``ON@20260416_120000:1,2``           → explicit session
        ``ON=red@20260416_120000:1,2``       → both
    """
    if ":" not in spec:
        raise argparse.ArgumentTypeError(
            f"--dataset {spec!r} must contain ':' separating name and trials"
        )
    name_part, trials_part = spec.split(":", 1)

    # Extract @SESSION first (the most specific suffix on name_part).
    if "@" in name_part:
        name_part, session = name_part.split("@", 1)
        if not session:
            raise argparse.ArgumentTypeError(
                f"--dataset {spec!r}: empty session after '@'"
            )
    else:
        if fallback_session is None:
            raise argparse.ArgumentTypeError(
                f"--dataset {spec!r}: no session specified "
                "(use @SESSION in spec or pass --session)"
            )
        session = fallback_session

    if "=" in name_part:
        label, color = name_part.split("=", 1)
    else:
        label, color = name_part, fallback_color
    if not label:
        raise argparse.ArgumentTypeError(f"--dataset {spec!r}: missing name")

    trials_part = trials_part.strip()
    if trials_part in ("*", "all"):
        return _DatasetSpec(label=label, color=color, session=session, trials=None)
    try:
        trials = [int(t) for t in trials_part.split(",") if t.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--dataset {spec!r}: trials must be comma-separated ints "
            f"(or '*'/'all') ({exc})"
        )
    if not trials:
        raise argparse.ArgumentTypeError(f"--dataset {spec!r}: no trials listed")
    return _DatasetSpec(label=label, color=color, session=session, trials=trials)


def _discover_session_trials(seed_dir: Path) -> list[int]:
    """List all ``trial_NNN`` indices present under *seed_dir*."""
    out: list[int] = []
    if not seed_dir.is_dir():
        return out
    for td in sorted(seed_dir.iterdir()):
        if td.is_dir() and td.name.startswith("trial_"):
            try:
                out.append(int(td.name.removeprefix("trial_")))
            except ValueError:
                continue
    return out


def _trial_succeeded(trial_path: Path) -> bool:
    """Read ``report.json`` and return whether the episode succeeded."""
    rp = trial_path / "report.json"
    if not rp.is_file():
        return False
    try:
        import json
        return bool(json.loads(rp.read_text()).get("success", False))
    except Exception:  # noqa: BLE001
        return False


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--task", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument(
        "--dataset", action="append", required=True, dest="datasets",
        help=("Dataset group 'NAME[=COLOR][@SESSION]:TRIAL1,TRIAL2,...'. "
              "Repeat for multiple datasets."),
    )
    p.add_argument(
        "--session", default=None,
        help="Fallback session id used when a dataset spec omits @SESSION.",
    )
    p.add_argument("--arm", default="right", choices=["left", "right"])
    p.add_argument("--output-dir", default=None)
    p.add_argument("--subsample", type=int, default=4,
                   help="Take every N-th step for FK")
    p.add_argument(
        "--only-success", action="store_true",
        help=("Keep only trials whose report.json says success=true. Applied "
              "to both explicit trial lists and '*'/'all' expansions."),
    )
    p.add_argument("--config", default=None,
                   help="Config YAML path (default: configs/default.yaml)")
    args = p.parse_args()

    specs: list[_DatasetSpec] = []
    for i, spec in enumerate(args.datasets):
        fallback = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        specs.append(_parse_dataset_spec(spec, fallback, args.session))

    config_path = Path(args.config) if args.config else (
        REPO_ROOT / "closed_loop_cap" / "configs" / "default.yaml"
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_base = Path(REPO_ROOT) / config.get("output_dir", "closed_loop_cap/output")

    # Expand '*'/'all' + optionally filter by report.json success.
    print("Datasets:")
    for ds in specs:
        logs_base = seed_logs_dir(output_base, ds.session, args.task, args.seed)
        if ds.trials is None:
            ds.trials = _discover_session_trials(logs_base)
        if args.only_success:
            kept = [
                t for t in ds.trials
                if _trial_succeeded(logs_base / f"trial_{t:03d}")
            ]
            dropped = set(ds.trials) - set(kept)
            if dropped:
                print(f"  [only-success] {ds.label}: dropped {sorted(dropped)}")
            ds.trials = kept
        print(f"  {ds.label} [{ds.color}] @ {ds.session}: trials={ds.trials}")

    # Load joint paths per (dataset, trial). Each dataset may live in a
    # different session directory.
    per_ds_joints: list[list[tuple[int, np.ndarray]]] = []
    for ds in specs:
        logs_base = seed_logs_dir(output_base, ds.session, args.task, args.seed)
        loaded: list[tuple[int, np.ndarray]] = []
        for trial in ds.trials:
            traj_path = logs_base / f"trial_{trial:03d}" / "traj.pkl"
            if not traj_path.exists():
                print(f"  WARNING: {traj_path} not found, skipping")
                continue
            traj = load_traj(traj_path)
            joint_pos = concat_positions(traj.get(f"{args.arm}_joint_path", []))
            print(f"  {ds.label}/trial_{trial:03d}: {len(joint_pos)} steps")
            loaded.append((trial, joint_pos))
        per_ds_joints.append(loaded)

    # SAPIEN FK — shared environment across all datasets (same seed).
    print(f"\nLoading SAPIEN environment (task={args.task}, seed={args.seed})...")
    from closed_loop_cap.env.task_env import make_env, close_env

    handle = make_env(args.task, args.seed, config)
    datasets: list[Dataset] = []
    try:
        refs = get_reference_points(handle.task_env)
        print(f"  left_arm_base:    {refs['left_arm_base']}")
        print(f"  right_arm_base:   {refs['right_arm_base']}")
        print(f"  initial_left_ee:  {refs['initial_left_ee']}")
        print(f"  initial_right_ee: {refs['initial_right_ee']}")

        for spec, joint_paths in zip(specs, per_ds_joints):
            ds = Dataset(label=spec.label, color=spec.color, trajectories=[])
            for trial, jp in joint_paths:
                if len(jp) == 0:
                    ds.trajectories.append(np.empty((0, 3)))
                    continue
                print(f"  FK for {ds.label}/trial_{trial:03d} "
                      f"({len(jp)} steps, subsample={args.subsample})...")
                ee = compute_ee_from_joint_path(
                    handle.task_env, jp, arm=args.arm, subsample=args.subsample,
                )
                ds.trajectories.append(ee)
                handle.task_env.robot.move_to_homestate()
                for _ in range(10):
                    handle.task_env.scene.step()
            datasets.append(ds)
    finally:
        close_env(handle)

    ds_names = "_vs_".join(ds.label for ds in datasets)
    out_dir = Path(args.output_dir) if args.output_dir else (
        viz_dir(output_base, specs[0].session, args.task)
        / f"seed_{args.seed}_{ds_names}"
    )
    title = (
        f"EE Trajectory — {args.task} seed={args.seed} "
        f"({args.arm} arm, dataset compare)"
    )

    plot_dataset_comparison_3d(
        datasets, title, str(out_dir / "ee_comparison_3d.png"), refs=refs,
    )
    plot_dataset_comparison_3d_html(
        datasets, title, str(out_dir / "ee_comparison_3d.html"), refs=refs,
    )
    print(f"\nAll outputs → {out_dir}")


if __name__ == "__main__":
    main()
