import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np


def replay_trajectory(task_env, arm_tag, position, velocity, save_freq=-1):
    """
    보간된 궤적을 take_dense_action으로 실행.

    Args:
        task_env: BaseTaskWithIKExecution 인스턴스
        arm_tag: "left" or "right"
        position: np.ndarray (N, n_joints) - 관절 위치 궤적
        velocity: np.ndarray (N, n_joints) - 관절 속도 궤적
        save_freq: 이미지 저장 빈도 (-1이면 기본값 사용)
    """
    arm_result = {
        "position": np.array(position),
        "velocity": np.array(velocity),
    }

    control_seq = {
        "left_arm": arm_result if arm_tag == "left" else None,
        "left_gripper": None,
        "right_arm": arm_result if arm_tag == "right" else None,
        "right_gripper": None,
    }

    task_env.take_dense_action(control_seq, save_freq=save_freq)
