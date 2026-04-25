"""Skill-segmented EE trajectory visualizer for closed-loop CaP traj.pkl files.

Mirrors ``AutoDataCollector/lerobot/scripts/skill_visualizer/visualize.py``
but loads data from traj.pkl + subtask_timeline.json instead of LeRobot
parquet datasets, and uses SAPIEN FK (exact robot model) for EE computation.

Produces:
  1. 4-view 3D matplotlib PNG  (per-skill colour, arm origins, initial EE)
  2. Interactive plotly HTML    (hover shows skill name + step index)

Usage:
    python -m closed_loop_cap.visualization.skill_visualizer.visualize \
        --task beat_block_hammer --seed 12 --trial 1 \
        --arm right
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from closed_loop_cap.paths import trial_dir as _trial_dir, viz_dir
from closed_loop_cap.visualization.traj_loader import load_traj
from closed_loop_cap.visualization.fk import (
    compute_ee_from_joint_path,
    get_reference_points,
)
from closed_loop_cap.visualization.plot_ee import (
    _VIEWS_3D,
    WAYPOINT_SIZE, ENDPOINT_SIZE, BASE_SIZE, INITIAL_EE_SIZE,
)


# ---------------------------------------------------------------------------
# Segment loading: traj.pkl segments + subtask_timeline labels
# ---------------------------------------------------------------------------

def _load_segments_and_labels(
    traj_path: Path,
    timeline_path: Path,
    arm: str,
) -> tuple[list[np.ndarray], list[str], list[tuple[int, int]]]:
    """Load joint-path segments and assign skill labels from timeline.

    Each element in ``{arm}_joint_path`` is one plan-result dict (one IK
    solve).  ``grasp_actor`` produces 2 segments (pre-grasp + grasp);
    ``place_actor`` produces 2 (pre-place + place); transit skills produce 1.

    We map segments to subtasks greedily: within each subtask the number
    of move actions is deterministic, so we just walk the timeline and
    consume segments in order.
    """
    traj = load_traj(traj_path)
    raw_segs = traj.get(f"{arm}_joint_path", [])

    # Extract successful position arrays
    segments: list[np.ndarray] = []
    for seg in raw_segs:
        if isinstance(seg, dict) and seg.get("status") == "Success":
            segments.append(np.asarray(seg["position"]))

    # Build cumulative step ranges
    offset = 0
    seg_ranges: list[tuple[int, int]] = []
    for seg in segments:
        seg_ranges.append((offset, offset + len(seg)))
        offset += len(seg)

    # Load timeline for labels
    if timeline_path.exists():
        with open(timeline_path) as f:
            tl = json.load(f)
        subtasks = tl.get("subtasks", [])
    else:
        subtasks = []

    # Assign labels: walk subtasks in order, each consumes some segments.
    # Heuristic: interaction skills (grasp/place) generate 2 move segments,
    # transit/gripper skills generate 0-1.
    INTERACTION_SEG_COUNT = {"grasp": 2, "place": 2}

    seg_labels: list[str] = []
    seg_idx = 0
    for st in subtasks:
        skill = st.get("skill_type", "")
        nl = st.get("natural_language", skill)
        count = INTERACTION_SEG_COUNT.get(skill, 1)
        for _ in range(count):
            if seg_idx >= len(segments):
                break
            seg_labels.append(f"{skill}: {nl}")
            seg_idx += 1

    # Fill remaining unlabelled segments
    while len(seg_labels) < len(segments):
        seg_labels.append(f"segment {len(seg_labels)}")

    return segments, seg_labels, seg_ranges


# ---------------------------------------------------------------------------
# SAPIEN FK for each segment
# ---------------------------------------------------------------------------

def _compute_ee_segments(
    task_env,
    segments: list[np.ndarray],
    arm: str,
    subsample: int = 1,
) -> list[np.ndarray]:
    """Run FK per segment, returning list of (M, 3) EE position arrays."""
    ee_segs: list[np.ndarray] = []
    for seg in segments:
        ee = compute_ee_from_joint_path(task_env, seg, arm=arm, subsample=subsample)
        ee_segs.append(ee)
        # Reset to segment end state for continuity
    return ee_segs


# ---------------------------------------------------------------------------
# Plotting: matplotlib (lerobot skill_visualizer style)
# ---------------------------------------------------------------------------

def _short_label(text: str, max_len: int = 50) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _compute_axis_limits(ee_segments: list[np.ndarray], extra_points: list[np.ndarray]):
    all_pts = [s for s in ee_segments if len(s) > 0] + [np.atleast_2d(p) for p in extra_points]
    if not all_pts:
        return np.zeros(3), np.ones(3), np.ones(3)
    cat = np.concatenate(all_pts, axis=0)
    mins, maxs = cat.min(axis=0), cat.max(axis=0)
    spans = np.where((maxs - mins) < 1e-6, 1e-3, maxs - mins)
    margin = spans * 0.15
    return mins - margin, maxs + margin, spans + 2 * margin


def plot_skill_trajectory_3d(
    ee_segments: list[np.ndarray],
    seg_labels: list[str],
    seg_ranges: list[tuple[int, int]],
    refs: dict,
    title: str,
    output_path: str,
    step_label_stride: int = 25,
):
    """4-view 3D plot with per-skill colours, arm origins, and initial EE."""
    cmap = plt.colormaps.get_cmap("tab20")
    n_segs = max(len(ee_segments), 2)

    extra = [
        refs["left_arm_base"], refs["right_arm_base"],
        refs["initial_left_ee"], refs["initial_right_ee"],
    ]
    lo, hi, box = _compute_axis_limits(ee_segments, extra)

    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    fig.suptitle(title, fontsize=13)

    for vi, (vname, elev, azim) in enumerate(_VIEWS_3D):
        ax = fig.add_subplot(2, 2, vi + 1, projection="3d")

        # Per-skill segments — waypoints only (no line), one marker per sampled step
        for i, (traj, label, (s, e)) in enumerate(
            zip(ee_segments, seg_labels, seg_ranges)
        ):
            if len(traj) == 0:
                continue
            color = cmap(i / n_segs)
            legend = f"[{s}-{e - 1}] {_short_label(label)}"
            ax.scatter(
                traj[:, 0], traj[:, 1], traj[:, 2],
                color=color, s=WAYPOINT_SIZE, alpha=0.7, label=legend,
            )
            # Highlight start/end of each skill segment
            ax.scatter(*traj[0], color=color, marker="o", s=ENDPOINT_SIZE,
                       edgecolors="black", linewidths=0.3, zorder=5)
            ax.scatter(*traj[-1], color=color, marker="x", s=ENDPOINT_SIZE,
                       linewidths=1.2, zorder=5)

            # Step index labels
            if step_label_stride and step_label_stride > 0:
                n = len(traj)
                idxs = list(range(0, n, step_label_stride))
                if idxs and idxs[-1] != n - 1:
                    idxs.append(n - 1)
                for k in idxs:
                    ax.text(
                        traj[k, 0], traj[k, 1], traj[k, 2],
                        f"{s + k}", color=color, fontsize=5, alpha=0.8, zorder=6,
                    )

        # Per-arm bases
        ax.scatter(*refs["left_arm_base"], color="blue", marker="*",
                   s=BASE_SIZE, zorder=10, edgecolors="black",
                   linewidths=0.5, label="Left arm base")
        ax.scatter(*refs["right_arm_base"], color="red", marker="*",
                   s=BASE_SIZE, zorder=10, edgecolors="black",
                   linewidths=0.5, label="Right arm base")

        # Initial EE (both arms)
        ax.scatter(
            *refs["initial_left_ee"], facecolors="none", edgecolors="blue",
            marker="D", s=INITIAL_EE_SIZE, linewidths=1.5, zorder=10,
            label="Initial L-EE",
        )
        ax.scatter(
            *refs["initial_right_ee"], facecolors="none", edgecolors="red",
            marker="D", s=INITIAL_EE_SIZE, linewidths=1.5, zorder=10,
            label="Initial R-EE",
        )

        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(box)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(vname)
        if vi == 0:
            ax.legend(loc="upper left", fontsize=6, framealpha=0.85)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plotting: plotly (interactive HTML, lerobot skill_visualizer style)
# ---------------------------------------------------------------------------

def _matplotlib_color_to_hex(rgba) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
    )


def plot_skill_trajectory_3d_html(
    ee_segments: list[np.ndarray],
    seg_labels: list[str],
    seg_ranges: list[tuple[int, int]],
    refs: dict,
    title: str,
    output_path: str,
    step_label_stride: int = 25,
):
    """Interactive 3D skill-segmented plot (plotly)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed — skipping HTML plot.")
        return

    cmap = plt.colormaps.get_cmap("tab20")
    n_segs = max(len(ee_segments), 2)
    fig = go.Figure()

    for i, (traj, label, (s, e)) in enumerate(
        zip(ee_segments, seg_labels, seg_ranges)
    ):
        if len(traj) == 0:
            continue
        color = _matplotlib_color_to_hex(cmap(i / n_segs))
        n = len(traj)
        label_idxs = set(range(0, n, step_label_stride)) if step_label_stride else set()
        if step_label_stride:
            label_idxs.add(n - 1)
        text_labels = [str(s + k) if k in label_idxs else "" for k in range(n)]
        short = _short_label(label, 80)
        hover = [
            f"skill {i}: {short}<br>frame {s + k}<br>"
            f"x={traj[k, 0]:.4f}<br>y={traj[k, 1]:.4f}<br>z={traj[k, 2]:.4f}"
            for k in range(n)
        ]
        legend_name = f"[{s}-{e - 1}] {short}"
        group = f"skill{i}"
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
            mode="markers+text",
            marker=dict(size=2, color=color, opacity=0.7),
            text=text_labels,
            textfont=dict(color=color, size=8),
            textposition="top center",
            hovertext=hover, hoverinfo="text",
            name=legend_name, legendgroup=group,
        ))
        fig.add_trace(go.Scatter3d(
            x=[traj[0, 0]], y=[traj[0, 1]], z=[traj[0, 2]],
            mode="markers",
            marker=dict(size=4, color=color, symbol="circle",
                        line=dict(color="black", width=0.5)),
            legendgroup=group, showlegend=False,
            hovertext=f"skill {i} start (frame {s})", hoverinfo="text",
        ))
        fig.add_trace(go.Scatter3d(
            x=[traj[-1, 0]], y=[traj[-1, 1]], z=[traj[-1, 2]],
            mode="markers", marker=dict(size=4, color=color, symbol="x"),
            legendgroup=group, showlegend=False,
            hovertext=f"skill {i} end (frame {e - 1})", hoverinfo="text",
        ))

    # Per-arm bases
    for arm_label, key, colour in [
        ("Left arm base", "left_arm_base", "blue"),
        ("Right arm base", "right_arm_base", "red"),
    ]:
        bp = refs[key]
        fig.add_trace(go.Scatter3d(
            x=[bp[0]], y=[bp[1]], z=[bp[2]], mode="markers",
            marker=dict(size=8, color=colour, symbol="diamond",
                        line=dict(color="black", width=1)),
            name=arm_label,
            hovertext=f"{arm_label} ({bp[0]:.3f}, {bp[1]:.3f}, {bp[2]:.3f})",
            hoverinfo="text",
        ))

    # Initial EE (both arms)
    for arm_label, key, colour in [
        ("Initial L-EE", "initial_left_ee", "blue"),
        ("Initial R-EE", "initial_right_ee", "red"),
    ]:
        ee = refs[key]
        fig.add_trace(go.Scatter3d(
            x=[ee[0]], y=[ee[1]], z=[ee[2]], mode="markers",
            marker=dict(size=6, color=colour, symbol="diamond-open",
                        line=dict(color=colour, width=2)),
            name=arm_label,
            hovertext=f"{arm_label} ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})",
            hoverinfo="text",
        ))

    extra = [
        refs["left_arm_base"], refs["right_arm_base"],
        refs["initial_left_ee"], refs["initial_right_ee"],
    ]
    lo, hi, _ = _compute_axis_limits(ee_segments, extra)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="X (m)", range=[lo[0], hi[0]]),
            yaxis=dict(title="Y (m)", range=[lo[1], hi[1]]),
            zaxis=dict(title="Z (m)", range=[lo[2], hi[2]]),
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--task", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--trial", type=int, default=1)
    p.add_argument("--session", required=True,
                   help="Session id (directory name under output/datasets/).")
    p.add_argument("--arm", default="right", choices=["left", "right"])
    p.add_argument("--subsample", type=int, default=2,
                   help="Take every N-th step for FK")
    p.add_argument("--step-label-stride", type=int, default=25)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--config", default=None)
    args = p.parse_args()

    import yaml
    config_path = Path(args.config) if args.config else (
        REPO_ROOT / "closed_loop_cap" / "configs" / "default.yaml"
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_base = Path(REPO_ROOT) / config.get("output_dir", "closed_loop_cap/output")
    ep_dir = _trial_dir(
        output_base, args.session, args.task, args.seed, args.trial,
    )
    traj_path = ep_dir / "traj.pkl"
    timeline_path = ep_dir / "subtask_timeline.json"

    if not traj_path.exists():
        print(f"ERROR: {traj_path} not found")
        return 1

    print(f"Loading segments from {traj_path} ...")
    segments, seg_labels, seg_ranges = _load_segments_and_labels(
        traj_path, timeline_path, args.arm,
    )
    print(f"  {len(segments)} segments, {sum(len(s) for s in segments)} total steps")
    for i, (label, (s, e)) in enumerate(zip(seg_labels, seg_ranges)):
        print(f"    [{i}] steps {s:>5}–{e - 1:<5}  {_short_label(label, 80)}")

    # SAPIEN FK
    print(f"\nLoading SAPIEN environment (task={args.task}, seed={args.seed})...")
    from closed_loop_cap.env.task_env import make_env, close_env
    handle = make_env(args.task, args.seed, config)
    try:
        refs = get_reference_points(handle.task_env)
        print(f"  left_arm_base:    {refs['left_arm_base']}")
        print(f"  right_arm_base:   {refs['right_arm_base']}")
        print(f"  initial_left_ee:  {refs['initial_left_ee']}")
        print(f"  initial_right_ee: {refs['initial_right_ee']}")

        print(f"\n  Computing FK ({args.arm} arm, subsample={args.subsample})...")
        ee_segments = _compute_ee_segments(
            handle.task_env, segments, arm=args.arm, subsample=args.subsample,
        )
    finally:
        close_env(handle)

    # Output
    out_dir = Path(args.output_dir) if args.output_dir else (
        viz_dir(output_base, args.session, args.task)
        / f"seed_{args.seed}_trial_{args.trial:03d}_skills"
    )
    title = (
        f"Skill-segmented EE trajectory — {args.task} "
        f"seed={args.seed} trial={args.trial} ({args.arm} arm, "
        f"{len(segments)} skills, {sum(len(s) for s in segments)} steps)"
    )

    plot_skill_trajectory_3d(
        ee_segments, seg_labels, seg_ranges, refs,
        title, str(out_dir / "ee_trajectory_3d.png"),
        step_label_stride=args.step_label_stride,
    )
    plot_skill_trajectory_3d_html(
        ee_segments, seg_labels, seg_ranges, refs,
        title, str(out_dir / "ee_trajectory_3d.html"),
        step_label_stride=args.step_label_stride,
    )
    print(f"\nAll outputs → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
