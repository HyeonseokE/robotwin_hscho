"""Forward Kinematics helpers for converting joint paths to EE positions."""
from __future__ import annotations

import numpy as np


def get_reference_points(task_env) -> dict:
    """Extract per-arm base positions and initial EE positions.

    Each arm's first joint is attached to its own base link
    (``fl_base_link`` / ``fr_base_link`` on aloha-agilex). Returned values
    are xyz numpy arrays in world frame.
    """
    robot = task_env.robot
    left_base = robot.left_arm_joints[0].get_parent_link().entity.pose
    right_base = robot.right_arm_joints[0].get_parent_link().entity.pose
    return {
        "left_arm_base": np.array(left_base.p),
        "right_arm_base": np.array(right_base.p),
        "initial_left_ee": np.array(robot.get_left_ee_pose()[:3]),
        "initial_right_ee": np.array(robot.get_right_ee_pose()[:3]),
    }


def compute_ee_from_joint_path(
    task_env,
    joint_positions: np.ndarray,
    arm: str = "right",
    subsample: int = 1,
) -> np.ndarray:
    """Compute EE xyz positions via instant-kinematic FK.

    Sets joint positions directly with ``articulation.set_qpos`` (no physics
    simulation, no PD lag), then reads the EE pose through
    ``robot.get_{arm}_ee_pose()``. Produces geometrically exact FK — what the
    EE position *would be* at each commanded joint configuration.

    Args:
        task_env: Initialised ``Base_Task`` instance.
        joint_positions: ``(N, 6)`` arm joint angles in radians.
        arm: ``"left"`` or ``"right"``.
        subsample: Take every *N*-th step (1 = all steps).

    Returns:
        ``(M, 3)`` array of EE ``(x, y, z)`` positions.
    """
    robot = task_env.robot
    entity = robot.left_entity if arm == "left" else robot.right_entity
    arm_joints = robot.left_arm_joints if arm == "left" else robot.right_arm_joints

    # Map our 6 arm joints into the full active-joints index space.
    active_joints = entity.get_active_joints()
    arm_indices = [active_joints.index(j) for j in arm_joints]

    saved_qpos = np.asarray(entity.get_qpos()).copy()
    working_qpos = saved_qpos.copy()

    positions: list[np.ndarray] = []
    for i in range(0, len(joint_positions), subsample):
        for idx, val in zip(arm_indices, joint_positions[i]):
            working_qpos[idx] = float(val)
        entity.set_qpos(working_qpos)
        ee = robot.get_left_ee_pose() if arm == "left" else robot.get_right_ee_pose()
        positions.append(np.asarray(ee[:3]))

    # Restore the original articulation state.
    entity.set_qpos(saved_qpos)
    return np.array(positions)
