import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import imageio


def capture_keyframes(task_env, task_name, episode_id, iteration_id, save_dir="./camera_images",
                      num_keyframes=5, total_steps=None):
    """
    실행 중 또는 실행 후 키프레임 이미지를 캡처하여 저장.
    save_camera_images를 활용하여 관찰 이미지를 저장.

    Args:
        task_env: 태스크 환경 인스턴스
        task_name: 태스크 이름
        episode_id: 에피소드 번호
        iteration_id: 반복 루프 ID
        save_dir: 이미지 저장 기본 디렉토리
        num_keyframes: 캡처할 키프레임 수
        total_steps: 총 스텝 수 (None이면 현재 상태만 캡처)
    """
    generate_dir_name = f"iteration_{iteration_id}"

    # 현재 시점의 관찰 이미지를 캡처
    task_env.save_camera_images(
        task_name=task_name,
        step_name=f"step0_final_state",
        generate_num_id=generate_dir_name,
        save_dir=save_dir,
    )

    return os.path.join(save_dir, task_name, generate_dir_name)


def capture_during_execution(task_env, task_name, episode_id, iteration_id,
                             step_name, save_dir="./camera_images"):
    """
    실행 중 특정 시점에 키프레임을 캡처.

    Args:
        task_env: 태스크 환경 인스턴스
        task_name: 태스크 이름
        episode_id: 에피소드 번호
        iteration_id: 반복 루프 ID
        step_name: 스텝 이름 (예: "step1_grasp_complete")
        save_dir: 이미지 저장 기본 디렉토리
    """
    generate_dir_name = f"iteration_{iteration_id}"

    task_env.save_camera_images(
        task_name=task_name,
        step_name=step_name,
        generate_num_id=generate_dir_name,
        save_dir=save_dir,
    )


def capture_keyframe_sequence(task_env, task_name, episode_id, iteration_id,
                              step_names, save_dir="./camera_images"):
    """
    여러 스텝 이름으로 연속 키프레임 캡처.

    Args:
        step_names: list of step name strings
    """
    for step_name in step_names:
        capture_during_execution(
            task_env, task_name, episode_id, iteration_id,
            step_name, save_dir,
        )
