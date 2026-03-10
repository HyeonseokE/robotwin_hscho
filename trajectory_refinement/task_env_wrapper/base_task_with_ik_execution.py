import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from copy import deepcopy

from envs._base_task import Base_Task
from trajectory_refinement.ee_waypoint_to_joint_trajectory.solve_ik_chain import solve_ik_chain
from trajectory_refinement.ee_waypoint_to_joint_trajectory.smooth_with_topp import smooth_with_topp
from trajectory_refinement.ee_waypoint_to_joint_trajectory.replay_trajectory_in_sim import replay_trajectory


class BaseTaskWithIKExecution(Base_Task):
    """
    Base_Task를 상속하여 IK 기반 Cartesian 웨이포인트 실행 기능과
    EE 포즈 기록 기능을 추가한다. 기존 코드는 수정하지 않는다.
    """

    def __init__(self):
        super().__init__()
        self.ee_trajectory_buffer = {"left": [], "right": []}
        self._record_ee_freq = 0  # 0이면 기록 비활성화

    def take_dense_action(self, control_seq, save_freq=-1, record_ee_freq=0):
        """
        부모 take_dense_action을 오버라이드하여 EE 포즈 기록 기능 추가.
        record_ee_freq > 0이면 매 record_ee_freq 스텝마다 EE 포즈를 버퍼에 기록.
        """
        if record_ee_freq <= 0 and self._record_ee_freq <= 0:
            return super().take_dense_action(control_seq, save_freq=save_freq)

        freq = record_ee_freq if record_ee_freq > 0 else self._record_ee_freq

        left_arm, left_gripper, right_arm, right_gripper = (
            control_seq["left_arm"],
            control_seq["left_gripper"],
            control_seq["right_arm"],
            control_seq["right_gripper"],
        )

        actual_save_freq = self.save_freq if save_freq == -1 else save_freq
        if actual_save_freq is not None:
            self._take_picture()

        max_control_len = 0
        if left_arm is not None:
            max_control_len = max(max_control_len, left_arm["position"].shape[0])
        if left_gripper is not None:
            max_control_len = max(max_control_len, left_gripper["num_step"])
        if right_arm is not None:
            max_control_len = max(max_control_len, right_arm["position"].shape[0])
        if right_gripper is not None:
            max_control_len = max(max_control_len, right_gripper["num_step"])

        for control_idx in range(max_control_len):
            if left_arm is not None and control_idx < left_arm["position"].shape[0]:
                self.robot.set_arm_joints(
                    left_arm["position"][control_idx],
                    left_arm["velocity"][control_idx],
                    "left",
                )

            if left_gripper is not None and control_idx < left_gripper["num_step"]:
                self.robot.set_gripper(
                    left_gripper["result"][control_idx],
                    "left",
                    left_gripper["per_step"],
                )

            if right_arm is not None and control_idx < right_arm["position"].shape[0]:
                self.robot.set_arm_joints(
                    right_arm["position"][control_idx],
                    right_arm["velocity"][control_idx],
                    "right",
                )

            if right_gripper is not None and control_idx < right_gripper["num_step"]:
                self.robot.set_gripper(
                    right_gripper["result"][control_idx],
                    "right",
                    right_gripper["per_step"],
                )

            self.scene.step()

            if self.render_freq and control_idx % self.render_freq == 0:
                self._update_render()
                self.viewer.render()

            if actual_save_freq is not None and control_idx % actual_save_freq == 0:
                self._update_render()
                self._take_picture()

            # EE 포즈 기록
            if control_idx % freq == 0:
                if left_arm is not None:
                    self.ee_trajectory_buffer["left"].append(
                        self.robot.get_left_ee_pose()
                    )
                if right_arm is not None:
                    self.ee_trajectory_buffer["right"].append(
                        self.robot.get_right_ee_pose()
                    )

        if actual_save_freq is not None:
            self._take_picture()

        return True

    def record_ee_pose(self, arm_tag):
        """현재 EE 포즈를 버퍼에 기록."""
        pose = self.get_arm_pose(arm_tag)
        self.ee_trajectory_buffer[str(arm_tag)].append(pose)

    def get_recorded_ee_trajectory(self, arm_tag):
        """기록된 EE 포즈 리스트 반환."""
        return self.ee_trajectory_buffer[str(arm_tag)]

    def clear_ee_trajectory_buffer(self, arm_tag=None):
        """EE 포즈 버퍼 초기화."""
        if arm_tag is None:
            self.ee_trajectory_buffer = {"left": [], "right": []}
        else:
            self.ee_trajectory_buffer[str(arm_tag)] = []

    def move_cartesian_waypoints(self, arm_tag, waypoints, topp_step=1/250, max_ik_fail_fraction=0.3):
        """
        Dense Cartesian 웨이포인트 → IK 연쇄 → TOPP → take_dense_action 실행.

        Args:
            arm_tag: "left" or "right"
            waypoints: list of [x, y, z, qw, qx, qy, qz]
            topp_step: TOPP 보간 스텝 크기
            max_ik_fail_fraction: 허용 가능한 IK 실패 비율

        Returns:
            dict: {"status", "ik_fail_indices", "position", "velocity"}
        """
        ik_result = solve_ik_chain(self.robot, arm_tag, waypoints)

        if ik_result["status"] == "IK_Failed":
            fail_ratio = len(ik_result["ik_fail_indices"]) / len(waypoints)
            if fail_ratio > max_ik_fail_fraction:
                print(f"[move_cartesian_waypoints] IK failure ratio {fail_ratio:.2%} exceeds threshold {max_ik_fail_fraction:.2%}")
                return {
                    "status": "IK_Failed",
                    "ik_fail_indices": ik_result["ik_fail_indices"],
                    "position": None,
                    "velocity": None,
                }

        joint_path = ik_result["joint_path"]
        if len(joint_path) < 2:
            print("[move_cartesian_waypoints] Not enough valid IK solutions for TOPP")
            return {
                "status": "IK_Failed",
                "ik_fail_indices": ik_result["ik_fail_indices"],
                "position": None,
                "velocity": None,
            }

        topp_result = smooth_with_topp(self.robot, arm_tag, joint_path, step=topp_step)

        if topp_result["status"] != "Success":
            return {
                "status": "TOPP_Failed",
                "ik_fail_indices": ik_result["ik_fail_indices"],
                "position": None,
                "velocity": None,
            }

        replay_trajectory(self, arm_tag, topp_result["position"], topp_result["velocity"])

        return {
            "status": "Success",
            "ik_fail_indices": ik_result["ik_fail_indices"],
            "position": topp_result["position"],
            "velocity": topp_result["velocity"],
        }
