import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np


def smooth_with_topp(robot, arm_tag, joint_path, step=1/250):
    """
    관절 경로에 TOPP 시간 최적 보간 적용.

    Args:
        robot: Robot 인스턴스
        arm_tag: "left" or "right"
        joint_path: np.ndarray (M, n_joints) - plan_screw 연쇄로 얻은 관절 경로
        step: 보간 시간 스텝 (기본 1/250)

    Returns:
        dict: {
            "status": "Success" | "TOPP_Failed",
            "position": np.ndarray (N, n_joints),
            "velocity": np.ndarray (N, n_joints),
        }
    """
    if arm_tag == "left":
        mplib_planner = robot.left_mplib_planner
    else:
        mplib_planner = robot.right_mplib_planner

    if joint_path.shape[0] < 2:
        return {
            "status": "TOPP_Failed",
            "position": None,
            "velocity": None,
        }

    try:
        result = mplib_planner.TOPP(joint_path, step=step)

        # TOPP returns (times, positions, velocities, accelerations, duration)
        if isinstance(result, tuple) and len(result) >= 3:
            positions = np.array(result[1])    # (N, n_joints)
            velocities = np.array(result[2])   # (N, n_joints)
        elif isinstance(result, dict):
            positions = np.array(result["position"])
            velocities = np.array(result["velocity"])
        else:
            return {
                "status": "TOPP_Failed",
                "position": None,
                "velocity": None,
            }

        return {
            "status": "Success",
            "position": positions,
            "velocity": velocities,
        }

    except Exception as e:
        print(f"[smooth_with_topp] TOPP failed: {e}")
        return {
            "status": "TOPP_Failed",
            "position": None,
            "velocity": None,
        }
