"""Visualization utilities for closed-loop CaP trajectories."""

from closed_loop_cap.visualization.traj_loader import concat_positions, load_traj
from closed_loop_cap.visualization.fk import (
    compute_ee_from_joint_path,
    get_reference_points,
)
from closed_loop_cap.visualization.plot_ee import (
    Dataset,
    plot_comparison_3d,
    plot_comparison_3d_html,
    plot_dataset_comparison_3d,
    plot_dataset_comparison_3d_html,
)

__all__ = [
    "load_traj",
    "concat_positions",
    "compute_ee_from_joint_path",
    "get_reference_points",
    "Dataset",
    "plot_comparison_3d",
    "plot_comparison_3d_html",
    "plot_dataset_comparison_3d",
    "plot_dataset_comparison_3d_html",
]
