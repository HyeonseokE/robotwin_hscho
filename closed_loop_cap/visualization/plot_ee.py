"""EE trajectory plotting functions (matplotlib + plotly)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Dataset:
    """Group of trajectories sharing a single colour and legend label.

    Used for dataset-vs-dataset comparisons (e.g. perturbation OFF vs ON),
    where every trajectory in the group should render identically so the
    visual contrast is between groups, not between individual trials.
    """
    label: str
    color: str
    trajectories: list[np.ndarray] = field(default_factory=list)

_VIEWS_3D = [
    ("Isometric", 25, 45),
    ("Top (XY)", 90, -90),
    ("Front (XZ)", 0, -90),
    ("Side (YZ)", 0, 0),
]

# Default palette — extend or override via the *colors* parameter.
DEFAULT_COLORS = [
    "#2196F3",  # blue
    "#FF5722",  # deep orange
    "#4CAF50",  # green
    "#9C27B0",  # purple
    "#FF9800",  # orange
    "#00BCD4",  # cyan
    "#E91E63",  # pink
    "#795548",  # brown
]

# Marker sizes (shared across matplotlib plots in this module).
WAYPOINT_SIZE = 3
ENDPOINT_SIZE = 18
BASE_SIZE = 80
INITIAL_EE_SIZE = 60


def _axis_limits(
    trajectories: list[tuple[str, np.ndarray]],
    extra_points: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_pts = [t for _, t in trajectories if len(t) > 0]
    if extra_points:
        all_pts += [np.atleast_2d(p) for p in extra_points]
    if not all_pts:
        return np.zeros(3), np.ones(3), np.ones(3)
    cat = np.concatenate(all_pts, axis=0)
    mins, maxs = cat.min(axis=0), cat.max(axis=0)
    spans = np.where((maxs - mins) < 1e-6, 1e-3, maxs - mins)
    margin = spans * 0.15
    return mins - margin, maxs + margin, spans + 2 * margin


def _axis_limits_datasets(
    datasets: list[Dataset],
    extra_points: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_pts: list[np.ndarray] = []
    for ds in datasets:
        all_pts += [t for t in ds.trajectories if len(t) > 0]
    if extra_points:
        all_pts += [np.atleast_2d(p) for p in extra_points]
    if not all_pts:
        return np.zeros(3), np.ones(3), np.ones(3)
    cat = np.concatenate(all_pts, axis=0)
    mins, maxs = cat.min(axis=0), cat.max(axis=0)
    spans = np.where((maxs - mins) < 1e-6, 1e-3, maxs - mins)
    margin = spans * 0.15
    return mins - margin, maxs + margin, spans + 2 * margin


def _refs_extra_points(refs: dict | None) -> list[np.ndarray]:
    if not refs:
        return []
    return [
        refs["left_arm_base"], refs["right_arm_base"],
        refs["initial_left_ee"], refs["initial_right_ee"],
    ]


REF_COLOR = "#2E7D32"  # deep green — used uniformly for both arm bases + init EE


def _draw_reference_points(ax, refs: dict) -> None:
    """Scatter per-arm bases (star) and initial EE poses (diamond) on a 3D axis."""
    ax.scatter(
        *refs["left_arm_base"], color=REF_COLOR, marker="*", s=BASE_SIZE,
        edgecolors="black", linewidths=0.5, zorder=10, label="Left arm base",
    )
    ax.scatter(
        *refs["right_arm_base"], color=REF_COLOR, marker="*", s=BASE_SIZE,
        edgecolors="black", linewidths=0.5, zorder=10, label="Right arm base",
    )
    ax.scatter(
        *refs["initial_left_ee"], facecolors="none", edgecolors=REF_COLOR,
        marker="D", s=INITIAL_EE_SIZE, linewidths=1.5, zorder=10,
        label="Initial L-EE",
    )
    ax.scatter(
        *refs["initial_right_ee"], facecolors="none", edgecolors=REF_COLOR,
        marker="D", s=INITIAL_EE_SIZE, linewidths=1.5, zorder=10,
        label="Initial R-EE",
    )


# ---------- matplotlib (static PNG) ----------


def plot_comparison_3d(
    trajectories: list[tuple[str, np.ndarray]],
    title: str,
    output_path: str,
    colors: list[str] | None = None,
    refs: dict | None = None,
):
    """Render a 4-view 3D EE trajectory comparison to *output_path* (PNG).

    Trajectories are drawn as waypoint scatter (no connecting line) to make
    the sampled-points structure obvious.

    Args:
        trajectories: List of ``(label, ee_xyz)`` pairs where ``ee_xyz``
            is ``(N, 3)``.
        title: Figure suptitle.
        output_path: Destination file path (PNG).
        colors: Optional per-trajectory colour list (default: DEFAULT_COLORS).
        refs: Optional reference-points dict from
            :func:`closed_loop_cap.visualization.fk.get_reference_points`.
            When given, per-arm bases and initial EE markers are drawn.
    """
    palette = colors or DEFAULT_COLORS
    lo, hi, box = _axis_limits(trajectories, _refs_extra_points(refs))

    fig = plt.figure(figsize=(16, 13), constrained_layout=True)
    fig.suptitle(title, fontsize=13)

    for vi, (vname, elev, azim) in enumerate(_VIEWS_3D):
        ax = fig.add_subplot(2, 2, vi + 1, projection="3d")
        for i, (label, traj) in enumerate(trajectories):
            if len(traj) == 0:
                continue
            c = palette[i % len(palette)]
            ax.scatter(
                traj[:, 0], traj[:, 1], traj[:, 2],
                color=c, s=WAYPOINT_SIZE, alpha=0.7,
                label=f"{label} ({len(traj)} pts)",
            )
            ax.scatter(*traj[0], color=c, marker="o", s=ENDPOINT_SIZE,
                       edgecolors="black", linewidths=0.3, zorder=5)
            ax.scatter(*traj[-1], color=c, marker="x", s=ENDPOINT_SIZE,
                       linewidths=1.2, zorder=5)

        if refs:
            _draw_reference_points(ax, refs)
        else:
            ax.scatter(0, 0, 0, color="black", marker="*", s=BASE_SIZE,
                       zorder=10, label="Origin")

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
            ax.legend(loc="upper left", fontsize=7, framealpha=0.85)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


# ---------- plotly (interactive HTML) ----------


def plot_comparison_3d_html(
    trajectories: list[tuple[str, np.ndarray]],
    title: str,
    output_path: str,
    colors: list[str] | None = None,
    refs: dict | None = None,
):
    """Render an interactive 3D EE trajectory comparison to *output_path* (HTML).

    Requires ``plotly``; silently skips if not installed. Trajectories are
    drawn as waypoint markers only (no connecting line).
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed — skipping HTML plot.")
        return

    palette = colors or DEFAULT_COLORS
    fig = go.Figure()

    for i, (label, traj) in enumerate(trajectories):
        if len(traj) == 0:
            continue
        c = palette[i % len(palette)]
        hover = [
            f"{label}<br>step {k}<br>"
            f"x={traj[k, 0]:.4f}<br>y={traj[k, 1]:.4f}<br>z={traj[k, 2]:.4f}"
            for k in range(len(traj))
        ]
        group = f"t{i}"
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
            mode="markers",
            marker=dict(size=2, color=c, opacity=0.7),
            hovertext=hover, hoverinfo="text",
            name=f"{label} ({len(traj)} pts)",
            legendgroup=group,
        ))
        fig.add_trace(go.Scatter3d(
            x=[traj[0, 0]], y=[traj[0, 1]], z=[traj[0, 2]],
            mode="markers",
            marker=dict(size=4, color=c, symbol="circle",
                        line=dict(color="black", width=0.5)),
            legendgroup=group, showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter3d(
            x=[traj[-1, 0]], y=[traj[-1, 1]], z=[traj[-1, 2]],
            mode="markers",
            marker=dict(size=4, color=c, symbol="x"),
            legendgroup=group, showlegend=False, hoverinfo="skip",
        ))

    if refs:
        for arm_label, key in [
            ("Left arm base", "left_arm_base"),
            ("Right arm base", "right_arm_base"),
        ]:
            bp = refs[key]
            fig.add_trace(go.Scatter3d(
                x=[bp[0]], y=[bp[1]], z=[bp[2]], mode="markers",
                marker=dict(size=8, color=REF_COLOR, symbol="diamond",
                            line=dict(color="black", width=1)),
                name=arm_label,
                hovertext=f"{arm_label} ({bp[0]:.3f}, {bp[1]:.3f}, {bp[2]:.3f})",
                hoverinfo="text",
            ))
        for arm_label, key in [
            ("Initial L-EE", "initial_left_ee"),
            ("Initial R-EE", "initial_right_ee"),
        ]:
            ee = refs[key]
            fig.add_trace(go.Scatter3d(
                x=[ee[0]], y=[ee[1]], z=[ee[2]], mode="markers",
                marker=dict(size=6, color=REF_COLOR, symbol="diamond-open",
                            line=dict(color=REF_COLOR, width=2)),
                name=arm_label,
                hovertext=f"{arm_label} ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})",
                hoverinfo="text",
            ))

    lo, hi, _ = _axis_limits(trajectories, _refs_extra_points(refs))
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


# ---------- Dataset-group comparison (same colour within a group) ----------


def plot_dataset_comparison_3d(
    datasets: list[Dataset],
    title: str,
    output_path: str,
    refs: dict | None = None,
):
    """Render a 4-view 3D comparison where each dataset group shares a colour.

    Use this when comparing two or more collections of trajectories
    (e.g. perturbation OFF vs ON). Every trajectory inside a
    :class:`Dataset` renders with the same colour and a single legend entry
    labels the entire group, so the visual contrast is between datasets
    rather than between individual trials.
    """
    lo, hi, box = _axis_limits_datasets(datasets, _refs_extra_points(refs))

    fig = plt.figure(figsize=(16, 13), constrained_layout=True)
    fig.suptitle(title, fontsize=13)

    for vi, (vname, elev, azim) in enumerate(_VIEWS_3D):
        ax = fig.add_subplot(2, 2, vi + 1, projection="3d")
        for ds in datasets:
            n_nonempty = sum(1 for t in ds.trajectories if len(t) > 0)
            legend_shown = False
            for traj in ds.trajectories:
                if len(traj) == 0:
                    continue
                label = (
                    f"{ds.label} ({n_nonempty} trajs)"
                    if not legend_shown else None
                )
                ax.plot(
                    traj[:, 0], traj[:, 1], traj[:, 2],
                    color=ds.color, linewidth=0.8, alpha=0.75,
                    label=label,
                )
                legend_shown = True
                # endpoints for each trajectory (smaller, same colour)
                ax.scatter(*traj[0], color=ds.color, marker="o",
                           s=ENDPOINT_SIZE, edgecolors="black",
                           linewidths=0.3, zorder=5)
                ax.scatter(*traj[-1], color=ds.color, marker="x",
                           s=ENDPOINT_SIZE, linewidths=1.2, zorder=5)

        if refs:
            _draw_reference_points(ax, refs)

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
            ax.legend(loc="upper left", fontsize=7, framealpha=0.85)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_dataset_comparison_3d_html(
    datasets: list[Dataset],
    title: str,
    output_path: str,
    refs: dict | None = None,
):
    """Interactive plotly variant of :func:`plot_dataset_comparison_3d`.

    Individual trials stay togglable (each gets its own trace) but share
    a legendgroup so clicking the dataset legend toggles them all at once.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed — skipping HTML plot.")
        return

    fig = go.Figure()

    for ds_idx, ds in enumerate(datasets):
        group = f"ds{ds_idx}"
        n_nonempty = sum(1 for t in ds.trajectories if len(t) > 0)
        legend_shown = False
        for ti, traj in enumerate(ds.trajectories):
            if len(traj) == 0:
                continue
            hover = [
                f"{ds.label} / traj {ti}<br>step {k}<br>"
                f"x={traj[k, 0]:.4f}<br>y={traj[k, 1]:.4f}<br>z={traj[k, 2]:.4f}"
                for k in range(len(traj))
            ]
            fig.add_trace(go.Scatter3d(
                x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
                mode="lines",
                line=dict(color=ds.color, width=1.5),
                opacity=0.75,
                hovertext=hover, hoverinfo="text",
                name=f"{ds.label} ({n_nonempty} trajs)" if not legend_shown else None,
                showlegend=not legend_shown,
                legendgroup=group,
            ))
            legend_shown = True
            fig.add_trace(go.Scatter3d(
                x=[traj[0, 0]], y=[traj[0, 1]], z=[traj[0, 2]],
                mode="markers",
                marker=dict(size=4, color=ds.color, symbol="circle",
                            line=dict(color="black", width=0.5)),
                legendgroup=group, showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter3d(
                x=[traj[-1, 0]], y=[traj[-1, 1]], z=[traj[-1, 2]],
                mode="markers",
                marker=dict(size=4, color=ds.color, symbol="x"),
                legendgroup=group, showlegend=False, hoverinfo="skip",
            ))

    if refs:
        for arm_label, key in [
            ("Left arm base", "left_arm_base"),
            ("Right arm base", "right_arm_base"),
        ]:
            bp = refs[key]
            fig.add_trace(go.Scatter3d(
                x=[bp[0]], y=[bp[1]], z=[bp[2]], mode="markers",
                marker=dict(size=8, color=REF_COLOR, symbol="diamond",
                            line=dict(color="black", width=1)),
                name=arm_label,
                hovertext=f"{arm_label} ({bp[0]:.3f}, {bp[1]:.3f}, {bp[2]:.3f})",
                hoverinfo="text",
            ))
        for arm_label, key in [
            ("Initial L-EE", "initial_left_ee"),
            ("Initial R-EE", "initial_right_ee"),
        ]:
            ee = refs[key]
            fig.add_trace(go.Scatter3d(
                x=[ee[0]], y=[ee[1]], z=[ee[2]], mode="markers",
                marker=dict(size=6, color=REF_COLOR, symbol="diamond-open",
                            line=dict(color=REF_COLOR, width=2)),
                name=arm_label,
                hovertext=f"{arm_label} ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})",
                hoverinfo="text",
            ))

    lo, hi, _ = _axis_limits_datasets(datasets, _refs_extra_points(refs))
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
