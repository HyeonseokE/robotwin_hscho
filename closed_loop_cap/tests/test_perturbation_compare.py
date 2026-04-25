"""Compare traj.pkl files from perturbation OFF vs ON trials.

Usage:
    python -m closed_loop_cap.tests.test_perturbation_compare \
        --task beat_block_hammer --seed 12 \
        --baseline-trial 1 --perturbed-trials 2 3
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def load_traj(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _concat_positions(segments: list) -> np.ndarray:
    """Concatenate position arrays from plan-result dicts into one (N, 6) array."""
    arrays = []
    for seg in segments:
        if isinstance(seg, dict) and seg.get("status") == "Success":
            arrays.append(np.asarray(seg["position"]))
    if not arrays:
        return np.empty((0, 6))
    return np.concatenate(arrays, axis=0)


def compare_joint_paths(
    baseline: dict, perturbed: dict, label: str,
) -> dict:
    """Compare left/right joint paths between two trajectories."""
    results = {}
    for arm in ("left", "right"):
        key = f"{arm}_joint_path"
        bp = _concat_positions(baseline.get(key, []))
        pp = _concat_positions(perturbed.get(key, []))

        if bp.size == 0 and pp.size == 0:
            results[arm] = {"status": "both_empty"}
            continue

        if bp.size == 0 or pp.size == 0:
            results[arm] = {
                "status": "one_empty",
                "baseline_steps": len(bp),
                "perturbed_steps": len(pp),
            }
            continue

        # Shape may differ (different number of steps due to different IK solutions)
        min_len = min(len(bp), len(pp))

        bp_trim = bp[:min_len]
        pp_trim = pp[:min_len]
        diff = np.abs(bp_trim - pp_trim)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        identical = np.allclose(bp_trim, pp_trim, atol=1e-6)

        results[arm] = {
            "baseline_steps": len(bp),
            "perturbed_steps": len(pp),
            "compared_steps": min_len,
            "max_joint_diff_rad": round(max_diff, 6),
            "mean_joint_diff_rad": round(mean_diff, 6),
            "identical": identical,
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--baseline-trial", type=int, default=1)
    parser.add_argument("--perturbed-trials", type=int, nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        default="closed_loop_cap/output",
    )
    args = parser.parse_args()

    base = Path(args.output_dir) / "datasets" / args.task / "logs" / f"seed_{args.seed}"
    baseline_path = base / f"trial_{args.baseline_trial:03d}" / "traj.pkl"

    if not baseline_path.exists():
        print(f"ERROR: baseline traj not found: {baseline_path}")
        return 1

    baseline = load_traj(baseline_path)
    print(f"=== Baseline: trial_{args.baseline_trial:03d} ===")
    for arm in ("left", "right"):
        key = f"{arm}_joint_path"
        path_data = baseline.get(key, [])
        n = len(path_data) if isinstance(path_data, (list, np.ndarray)) else 0
        print(f"  {arm}: {n} segments")

    for trial in args.perturbed_trials:
        trial_path = base / f"trial_{trial:03d}" / "traj.pkl"
        if not trial_path.exists():
            print(f"\n=== trial_{trial:03d}: MISSING (traj.pkl not found) ===")
            continue

        perturbed = load_traj(trial_path)
        results = compare_joint_paths(baseline, perturbed, f"trial_{trial:03d}")

        print(f"\n=== trial_{trial:03d} (perturbation ON) vs baseline ===")
        for arm, r in results.items():
            if r.get("identical"):
                print(f"  {arm}: IDENTICAL (no perturbation effect)")
            else:
                print(
                    f"  {arm}: DIFFERENT  "
                    f"steps={r.get('baseline_steps')}→{r.get('perturbed_steps')}  "
                    f"max_diff={r.get('max_joint_diff_rad', '?')} rad  "
                    f"mean_diff={r.get('mean_joint_diff_rad', '?')} rad"
                )

    # Also compare perturbed trials against each other
    perturbed_trajs = []
    for trial in args.perturbed_trials:
        trial_path = base / f"trial_{trial:03d}" / "traj.pkl"
        if trial_path.exists():
            perturbed_trajs.append((trial, load_traj(trial_path)))

    if len(perturbed_trajs) >= 2:
        print(f"\n=== trial_{perturbed_trajs[0][0]:03d} vs trial_{perturbed_trajs[1][0]:03d} (both ON) ===")
        results = compare_joint_paths(
            perturbed_trajs[0][1], perturbed_trajs[1][1], "on_vs_on",
        )
        for arm, r in results.items():
            if r.get("identical"):
                print(f"  {arm}: IDENTICAL (same RNG? should differ across trials)")
            else:
                print(
                    f"  {arm}: DIFFERENT  "
                    f"steps={r.get('baseline_steps')}→{r.get('perturbed_steps')}  "
                    f"max_diff={r.get('max_joint_diff_rad', '?')} rad  "
                    f"mean_diff={r.get('mean_joint_diff_rad', '?')} rad"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
