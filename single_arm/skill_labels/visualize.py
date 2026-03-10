"""
visualize.py — Generate 4 types of skill-label visualisation PNGs from HDF5 data.

1. skill_timeline.png: language_subtask bars + skill_type bars + progress curve
2. skill_coordinates.png: goal_ee_pose XYZ over time
3. skill_joints.png: goal_joint_positions over time
4. skill_summary.png: table of language_subtask | verification_question | skill_type | frame range
"""

import os
import numpy as np

try:
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.table import Table
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


# ============================================================
# HDF5 loader
# ============================================================

def load_skill_labels_from_hdf5(hdf5_path: str) -> dict:
    """Load the skill_labels group from an HDF5 file.

    Returns a dict with keys matching the HDF5 datasets, values as numpy arrays.
    String datasets are decoded to lists of Python str.
    """
    labels = {}
    with h5py.File(hdf5_path, "r") as f:
        if "skill_labels" not in f:
            raise KeyError(f"No 'skill_labels' group in {hdf5_path}")
        grp = f["skill_labels"]
        for key in grp:
            data = grp[key][:]
            if data.dtype.kind in ("S", "O"):  # byte-string or object (vlen)
                data = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in data]
            labels[key] = data
    return labels


# ============================================================
# Colour palette helpers
# ============================================================

_SKILL_COLORS = {
    "grasp": "#e74c3c",
    "place": "#3498db",
    "move_to_pose": "#2ecc71",
    "move_by_displacement": "#f39c12",
    "close_gripper": "#9b59b6",
    "open_gripper": "#1abc9c",
    "back_to_origin": "#95a5a6",
    "together_move": "#e67e22",
    "idle": "#ecf0f1",
    "unknown": "#bdc3c7",
}


def _color_for(skill_type: str) -> str:
    return _SKILL_COLORS.get(skill_type, _SKILL_COLORS["unknown"])


def _skill_segment_ranges(skill_index):
    """Return list of (start, end, skill_idx) tuples from skill_index array."""
    segments = []
    if len(skill_index) == 0:
        return segments
    cur = skill_index[0]
    start = 0
    for i in range(1, len(skill_index)):
        if skill_index[i] != cur:
            segments.append((start, i - 1, int(cur)))
            cur = skill_index[i]
            start = i
    segments.append((start, len(skill_index) - 1, int(cur)))
    return segments


# ============================================================
# Plot 1: Timeline
# ============================================================

def _plot_timeline(labels, ax_subtask, ax_type, ax_progress):
    """Draw skill timeline: subtask bars, type bars, progress curve."""
    T = len(labels["skill_index"])
    segments = _skill_segment_ranges(labels["skill_index"])

    skill_types = labels["skill_type"]
    subtasks = labels["language_subtask"]
    progress = np.array(labels["progress"], dtype=np.float32)

    # Subtask bars
    for start, end, idx in segments:
        color = _color_for(skill_types[start])
        ax_subtask.barh(0, end - start + 1, left=start, height=0.8, color=color, edgecolor="white", linewidth=0.5)
        mid = (start + end) / 2
        label_text = subtasks[start]
        if len(label_text) > 30:
            label_text = label_text[:27] + "..."
        ax_subtask.text(mid, 0, label_text, ha="center", va="center", fontsize=6, color="white", fontweight="bold")

    ax_subtask.set_yticks([])
    ax_subtask.set_ylabel("subtask", fontsize=8)
    ax_subtask.set_xlim(0, T)

    # Skill type bars
    for start, end, idx in segments:
        color = _color_for(skill_types[start])
        ax_type.barh(0, end - start + 1, left=start, height=0.8, color=color, edgecolor="white", linewidth=0.5)
        mid = (start + end) / 2
        ax_type.text(mid, 0, skill_types[start], ha="center", va="center", fontsize=6, color="white")

    ax_type.set_yticks([])
    ax_type.set_ylabel("skill_type", fontsize=8)
    ax_type.set_xlim(0, T)

    # Progress curve
    ax_progress.plot(range(T), progress, color="#2c3e50", linewidth=1)
    ax_progress.set_ylabel("progress", fontsize=8)
    ax_progress.set_xlabel("frame", fontsize=8)
    ax_progress.set_xlim(0, T)
    ax_progress.set_ylim(-0.05, 1.05)

    # Segment boundaries
    for start, end, idx in segments:
        for ax in [ax_subtask, ax_type, ax_progress]:
            ax.axvline(x=start, color="gray", linewidth=0.3, linestyle="--")


# ============================================================
# Plot 2: Coordinates
# ============================================================

def _plot_coordinates(labels, axes):
    """Plot goal_ee_pose XYZ + quaternion."""
    goal_ee = np.array(labels["goal_ee_pose"], dtype=np.float32)
    T = len(goal_ee)
    frames = np.arange(T)

    coord_labels = ["X", "Y", "Z", "qw", "qx", "qy", "qz"]
    for i, (ax, lbl) in enumerate(zip(axes, coord_labels)):
        if i < goal_ee.shape[1]:
            ax.plot(frames, goal_ee[:, i], linewidth=0.8)
        ax.set_ylabel(lbl, fontsize=7)
        ax.set_xlim(0, T)
        if i < len(axes) - 1:
            ax.set_xticklabels([])

    axes[-1].set_xlabel("frame", fontsize=8)


# ============================================================
# Plot 3: Joints
# ============================================================

def _plot_joints(labels, axes):
    """Plot goal_joint_positions."""
    goal_jp = np.array(labels["goal_joint_positions"], dtype=np.float32)
    T = len(goal_jp)
    frames = np.arange(T)

    for i, ax in enumerate(axes):
        if i < goal_jp.shape[1]:
            ax.plot(frames, goal_jp[:, i], linewidth=0.8, color="#e74c3c", label="goal")
        ax.set_ylabel(f"J{i}", fontsize=7)
        ax.set_xlim(0, T)
        if i < len(axes) - 1:
            ax.set_xticklabels([])

    axes[-1].set_xlabel("frame", fontsize=8)


# ============================================================
# Plot 4: Summary table
# ============================================================

def _plot_summary(labels, ax):
    """Draw a summary table of skill segments."""
    segments = _skill_segment_ranges(labels["skill_index"])
    skill_types = labels["skill_type"]
    subtasks = labels["language_subtask"]
    questions = labels["verification_question"]

    col_labels = ["#", "skill_type", "language_subtask", "verification_question", "frames"]
    rows = []
    for start, end, idx in segments:
        rows.append([
            str(idx),
            skill_types[start],
            subtasks[start],
            questions[start],
            f"{start}-{end}",
        ])

    ax.axis("off")
    if not rows:
        ax.text(0.5, 0.5, "No skill segments found", ha="center", va="center")
        return

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.4)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternating row colors
    for i in range(len(rows)):
        color = "#f8f9fa" if i % 2 == 0 else "#ffffff"
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(color)


# ============================================================
# Public API
# ============================================================

def generate_skill_visualizations(
    hdf5_path: str,
    output_dir: str = None,
    episode_index: int = 0,
):
    """Generate 4 PNG visualisations from an HDF5 file's skill_labels group.

    Args:
        hdf5_path: Path to the HDF5 episode file.
        output_dir: Directory to save PNGs. Defaults to same dir as hdf5_path.
        episode_index: Episode number for filename prefix.
    """
    if not HAS_DEPS:
        print("[SkillViz] matplotlib or h5py not available, skipping visualisation.")
        return

    labels = load_skill_labels_from_hdf5(hdf5_path)
    T = len(labels["skill_index"])
    if T == 0:
        print("[SkillViz] No frames in skill_labels, skipping.")
        return

    if output_dir is None:
        output_dir = os.path.dirname(hdf5_path)
    os.makedirs(output_dir, exist_ok=True)

    prefix = f"ep{episode_index}"

    # --- 1. Timeline ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 4), sharex=True,
                                          gridspec_kw={"height_ratios": [1, 1, 2]})
    _plot_timeline(labels, ax1, ax2, ax3)
    fig.suptitle(f"Skill Timeline — Episode {episode_index}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_skill_timeline.png"), dpi=150)
    plt.close(fig)

    # --- 2. Coordinates ---
    n_coords = min(7, np.array(labels["goal_ee_pose"]).shape[1]) if len(labels["goal_ee_pose"]) > 0 else 3
    fig, axes = plt.subplots(n_coords, 1, figsize=(14, 2 * n_coords), sharex=True)
    if n_coords == 1:
        axes = [axes]
    _plot_coordinates(labels, axes)
    fig.suptitle(f"Goal EE Pose — Episode {episode_index}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_skill_coordinates.png"), dpi=150)
    plt.close(fig)

    # --- 3. Joints ---
    n_joints = min(6, np.array(labels["goal_joint_positions"]).shape[1]) if len(labels["goal_joint_positions"]) > 0 else 6
    fig, axes = plt.subplots(n_joints, 1, figsize=(14, 2 * n_joints), sharex=True)
    if n_joints == 1:
        axes = [axes]
    _plot_joints(labels, axes)
    fig.suptitle(f"Goal Joint Positions — Episode {episode_index}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_skill_joints.png"), dpi=150)
    plt.close(fig)

    # --- 4. Summary table ---
    fig, ax = plt.subplots(1, 1, figsize=(16, max(3, 0.5 * len(_skill_segment_ranges(labels["skill_index"])) + 1)))
    _plot_summary(labels, ax)
    fig.suptitle(f"Skill Summary — Episode {episode_index}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_skill_summary.png"), dpi=150)
    plt.close(fig)

    print(f"[SkillViz] Saved 4 PNGs to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualise skill labels from HDF5")
    parser.add_argument("hdf5_path", type=str, help="Path to episode HDF5 file")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()
    generate_skill_visualizations(args.hdf5_path, args.output_dir, args.episode)
