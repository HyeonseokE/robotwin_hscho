import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import sapien


def solve_ik_chain(robot, arm_tag, waypoints, start_qpos=None, subsample=1):
    """
    Cartesian 웨이포인트 리스트에 대해 plan_screw를 연쇄적으로 호출하여
    관절 경로를 생성한다.

    plan_screw는 현재 관절 상태에서 목표 EE 포즈까지의 선형 보간 경로를 반환한다.
    IK와 달리 collision checking 없이 동작하며, 인접 웨이포인트 간의 부드러운
    전이를 보장한다.

    Args:
        robot: Robot 인스턴스
        arm_tag: "left" or "right"
        waypoints: list of [x, y, z, qw, qx, qy, qz]
        start_qpos: 초기 관절 상태 (None이면 현재 entity qpos 사용)
        subsample: 웨이포인트 서브샘플 간격 (1이면 모든 웨이포인트 사용)

    Returns:
        dict: {
            "status": "Success" | "IK_Failed",
            "joint_path": np.ndarray (M, n_joints) - 모든 경로 연결,
            "ik_fail_indices": list[int] - 실패한 웨이포인트 인덱스,
        }
    """
    if arm_tag == "left":
        mplib_planner = robot.left_mplib_planner
        entity = robot.left_entity
    else:
        mplib_planner = robot.right_mplib_planner
        entity = robot.right_entity

    # 시작 관절 상태
    if start_qpos is not None:
        curr_qpos = np.array(start_qpos)
    else:
        curr_qpos = entity.get_qpos()

    # 서브샘플
    if subsample > 1:
        indices = list(range(0, len(waypoints), subsample))
        if indices[-1] != len(waypoints) - 1:
            indices.append(len(waypoints) - 1)
        sampled_wps = [waypoints[i] for i in indices]
    else:
        sampled_wps = waypoints
        indices = list(range(len(waypoints)))

    all_positions = []
    ik_fail_indices = []

    for i, wp in enumerate(sampled_wps):
        # Gripper 포즈 → end-link 포즈 변환
        endlink_pose = robot._trans_from_gripper_to_endlink(wp, arm_tag=arm_tag)
        target_pose = sapien.Pose(endlink_pose.p, endlink_pose.q)

        # plan_screw: 현재 qpos에서 목표 포즈까지의 선형 보간
        result = mplib_planner.plan_screw(
            curr_qpos, target_pose, arms_tag=arm_tag, log=False,
        )

        if result["status"] == "Success":
            positions = result["position"]
            all_positions.append(positions)
            # 다음 스텝의 시작 qpos = 현재 결과의 마지막 joint state
            curr_qpos = np.zeros_like(curr_qpos)
            n_move_joints = positions.shape[1]
            curr_qpos[:n_move_joints] = positions[-1]
        else:
            ik_fail_indices.append(indices[i] if subsample > 1 else i)

    if len(all_positions) == 0:
        return {
            "status": "IK_Failed",
            "joint_path": np.array([]),
            "ik_fail_indices": ik_fail_indices,
        }

    joint_path = np.vstack(all_positions)
    overall_status = "Success" if len(ik_fail_indices) == 0 else "IK_Failed"

    return {
        "status": overall_status,
        "joint_path": joint_path,
        "ik_fail_indices": ik_fail_indices,
    }
