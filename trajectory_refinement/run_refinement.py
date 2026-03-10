#!/usr/bin/env python3
"""
VLM 반복 궤적 최적화 파이프라인 메인 진입점.

Usage:
    python trajectory_refinement/run_refinement.py <task_name> [--max-iterations N] [--check-num N]

Example:
    python trajectory_refinement/run_refinement.py shake_bottle
    python trajectory_refinement/run_refinement.py shake_bottle --max-iterations 3 --check-num 3
"""

import sys
import os
import json
import argparse
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# observation_agent.py uses "from gpt_agent import ..." which requires code_gen/ in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code_gen"))

import yaml
import numpy as np

from trajectory_refinement.initial_trajectory.run_existing_pipeline import (
    run_initial_trajectory,
    _setup_task_args,
    _create_refinable_task,
)
from trajectory_refinement.initial_trajectory.extract_ee_poses_from_traj import (
    filter_stationary_waypoints,
)
from trajectory_refinement.ee_waypoint_to_joint_trajectory.solve_ik_chain import solve_ik_chain
from trajectory_refinement.ee_waypoint_to_joint_trajectory.smooth_with_topp import smooth_with_topp
from trajectory_refinement.ee_waypoint_to_joint_trajectory.replay_trajectory_in_sim import replay_trajectory
from trajectory_refinement.vlm_analysis.capture_keyframes import (
    capture_keyframes,
    capture_during_execution,
)
from trajectory_refinement.vlm_analysis.analyze_execution_with_vlm import analyze_execution
from trajectory_refinement.llm_waypoint_editing.refine_waypoints_with_llm import refine_waypoints
from trajectory_refinement.prompts.llm_waypoint_edit_prompt import LLM_SYSTEM_PROMPT


def load_config(config_path=None):
    """Load refinement configuration."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "configs", "refinement_config.yml")

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    # 기본 설정
    return {
        "max_iterations": 5,
        "check_num": 5,
        "success_threshold": 0.5,
        "record_ee_freq": 15,
        "sample_interval": 1,
        "topp_step": 1 / 250,
        "max_ik_fail_fraction": 0.3,
        "save_dir": "./camera_images",
        "output_dir": "./trajectory_refinement/output",
    }


def save_waypoints(waypoints, filepath):
    """웨이포인트를 JSON으로 저장."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(waypoints, f, indent=2)
    print(f"[save_waypoints] Saved to {filepath}")


def determine_active_arm(left_waypoints, right_waypoints):
    """더 많은 웨이포인트가 기록된 쪽을 활성 팔로 결정."""
    if len(left_waypoints) > len(right_waypoints):
        return "left", left_waypoints
    elif len(right_waypoints) > len(left_waypoints):
        return "right", right_waypoints
    else:
        # 둘 다 같으면 움직임이 더 큰 쪽 선택
        def calc_movement(wps):
            if len(wps) < 2:
                return 0
            total = 0
            for i in range(1, len(wps)):
                total += np.linalg.norm(
                    np.array(wps[i][:3]) - np.array(wps[i - 1][:3])
                )
            return total

        if calc_movement(left_waypoints) >= calc_movement(right_waypoints):
            return "left", left_waypoints
        else:
            return "right", right_waypoints


def run_evaluation_episodes(task_name, arm_tag, waypoints, args, config,
                            iteration_id, check_num):
    """
    N개 에피소드에서 웨이포인트를 평가.

    각 에피소드에서:
    1. play_once()로 전체 태스크 실행 (grasp + 동작)
    2. 키프레임 캡처 + check_success
    3. 웨이포인트에 대한 IK chain 검증 (실패 인덱스 수집)

    Returns:
        tuple: (success_rate, last_task_env, last_ik_result)
    """
    success_count = 0
    valid_count = 0
    last_task_env = None
    last_ik_result = None
    seed = 0
    max_seed = check_num + 20

    while valid_count < check_num and seed < max_seed:
        task_env = None
        try:
            task_env = _create_refinable_task(task_name)
            task_env.setup_demo(now_ep_num=valid_count, seed=seed, **args)
        except Exception as e:
            print(f"  Seed {seed}: SKIP (unstable: {e})")
            seed += 1
            if task_env is not None:
                try:
                    task_env.close()
                except:
                    pass
            continue

        try:
            # 키프레임 캡처 - 실행 전
            capture_during_execution(
                task_env, task_name, valid_count, iteration_id,
                "step0_initial_state",
                save_dir=config.get("save_dir", "./camera_images"),
            )

            # play_once로 전체 태스크 실행
            task_env.play_once()

            # 키프레임 캡처 - 실행 후
            capture_during_execution(
                task_env, task_name, valid_count, iteration_id,
                "step1_after_execution",
                save_dir=config.get("save_dir", "./camera_images"),
            )

            # 성공 여부 확인
            success = task_env.plan_success and task_env.check_success()
            if success:
                success_count += 1
                print(f"  Episode {valid_count} (seed={seed}): SUCCESS")
            else:
                print(f"  Episode {valid_count} (seed={seed}): FAIL")

            # 웨이포인트에 대한 IK chain 검증 (실패 인덱스 수집)
            # play_once 후의 joint state를 seed로 사용
            ik_result = solve_ik_chain(
                task_env.robot, arm_tag, waypoints,
                start_qpos=task_env.robot.left_entity.get_qpos() if arm_tag == "left" else task_env.robot.right_entity.get_qpos(),
                subsample=10,  # 검증용이므로 서브샘플링
            )
            last_ik_result = ik_result

            valid_count += 1
            last_task_env = task_env
            task_env.close()

        except Exception as e:
            print(f"  Episode {valid_count} (seed={seed}): ERROR - {e}")
            traceback.print_exc()
            valid_count += 1
            try:
                task_env.close()
            except:
                pass

        seed += 1

    actual_count = max(valid_count, 1)
    success_rate = success_count / actual_count
    return success_rate, last_task_env, last_ik_result


def main(task_name, max_iterations=None, check_num=None, success_threshold=None,
         config_path=None):
    """
    VLM 반복 궤적 최적화 메인 루프.

    Args:
        task_name: 태스크 이름 (예: "shake_bottle")
        max_iterations: 최대 반복 횟수
        check_num: 성공률 검증 에피소드 수
        success_threshold: 목표 성공률
        config_path: 설정 파일 경로
    """
    config = load_config(config_path)

    max_iterations = max_iterations or config.get("max_iterations", 5)
    check_num = check_num or config.get("check_num", 5)
    success_threshold = success_threshold or config.get("success_threshold", 0.5)

    output_dir = os.path.join(config.get("output_dir", "./trajectory_refinement/output"), task_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"=" * 60)
    print(f"VLM Trajectory Refinement Pipeline")
    print(f"Task: {task_name}")
    print(f"Max iterations: {max_iterations}")
    print(f"Check episodes: {check_num}")
    print(f"Success threshold: {success_threshold}")
    print(f"Output directory: {output_dir}")
    print(f"=" * 60)

    # =========================================================
    # Phase 0: 초기 궤적 생성
    # =========================================================
    print(f"\n[Phase 0] Generating initial trajectory...")
    task_env, left_wps, right_wps, task_info, args = run_initial_trajectory(
        task_name,
        seed=0,
        record_ee_freq=config.get("record_ee_freq", 15),
    )

    # 활성 팔 결정
    arm_tag, waypoints = determine_active_arm(left_wps, right_wps)
    print(f"  Active arm: {arm_tag}, waypoints: {len(waypoints)}")

    # 정지 웨이포인트 필터링
    waypoints = filter_stationary_waypoints(waypoints)
    print(f"  After filtering: {len(waypoints)} waypoints")

    # 초기 웨이포인트 저장
    save_waypoints(waypoints, os.path.join(output_dir, "initial_waypoints.json"))

    # task_info 구성
    if task_info is None:
        task_info = {}
    task_info_for_vlm = {
        "description": task_info.get("info", {}).get("description", f"Robot task: {task_name}"),
        "goal": task_info.get("info", {}).get("goal", f"Complete the {task_name} task successfully"),
    }

    task_env.close()

    # =========================================================
    # Phase 1+: VLM 반복 개선 루프
    # =========================================================
    best_waypoints = waypoints
    best_success_rate = 0.0
    messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]

    for iteration in range(max_iterations):
        print(f"\n{'=' * 60}")
        print(f"[Iteration {iteration + 1}/{max_iterations}]")
        print(f"{'=' * 60}")

        # 2a. N개 에피소드에서 실행 및 평가
        print(f"\n  Evaluating with {check_num} episodes...")
        success_rate, last_env, last_ik_result = run_evaluation_episodes(
            task_name, arm_tag, waypoints, args, config,
            iteration_id=iteration, check_num=check_num,
        )

        print(f"\n  Success rate: {success_rate:.0%} ({int(success_rate * check_num)}/{check_num})")

        # 최고 성공률 업데이트
        if success_rate > best_success_rate:
            best_waypoints = waypoints
            best_success_rate = success_rate
            save_waypoints(best_waypoints, os.path.join(output_dir, "best_waypoints.json"))
            print(f"  New best! Saved best_waypoints.json")

        # 성공률 달성 시 종료
        if success_rate >= success_threshold:
            print(f"\n  Target success rate {success_threshold:.0%} achieved! Stopping.")
            break

        # 2b. VLM 분석
        print(f"\n  Running VLM analysis...")
        try:
            vlm_feedback = analyze_execution(
                episode_id=0,
                task_name=task_name,
                task_info=task_info_for_vlm,
                current_waypoints_json=waypoints,
                save_dir=config.get("save_dir", "./camera_images"),
                iteration_id=iteration,
            )
            print(f"  VLM feedback: {vlm_feedback[:200]}...")
        except Exception as e:
            print(f"  VLM analysis failed: {e}")
            vlm_feedback = f"VLM analysis failed. The current success rate is {success_rate:.0%}. Please try adjusting waypoints to improve task execution."

        # 2c. LLM 웨이포인트 수정
        print(f"\n  Refining waypoints with LLM...")
        ik_fail_indices = last_ik_result["ik_fail_indices"] if last_ik_result else []

        try:
            waypoints, messages = refine_waypoints(
                task_info=task_info_for_vlm,
                current_waypoints=waypoints,
                vlm_feedback=vlm_feedback,
                ik_fail_indices=ik_fail_indices,
                messages=messages,
                arm_tag=arm_tag,
            )
            print(f"  Refined waypoints: {len(waypoints)} points")
        except Exception as e:
            print(f"  LLM refinement failed: {e}")
            traceback.print_exc()
            continue

        # 수정된 웨이포인트 저장
        save_waypoints(
            waypoints,
            os.path.join(output_dir, f"refined_waypoints_iter{iteration}.json"),
        )

    # =========================================================
    # 최종 결과
    # =========================================================
    print(f"\n{'=' * 60}")
    print(f"Refinement Complete")
    print(f"Best success rate: {best_success_rate:.0%}")
    print(f"Best waypoints saved to: {os.path.join(output_dir, 'best_waypoints.json')}")
    print(f"{'=' * 60}")

    # 최종 저장
    save_waypoints(best_waypoints, os.path.join(output_dir, "best_waypoints.json"))

    return best_waypoints, best_success_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Trajectory Refinement Pipeline")
    parser.add_argument("task_name", type=str, help="Task name (e.g., shake_bottle)")
    parser.add_argument("--max-iterations", type=int, default=None, help="Max refinement iterations")
    parser.add_argument("--check-num", type=int, default=None, help="Number of evaluation episodes")
    parser.add_argument("--success-threshold", type=float, default=None, help="Target success rate")
    parser.add_argument("--config", type=str, default=None, help="Config file path")

    args = parser.parse_args()

    main(
        task_name=args.task_name,
        max_iterations=args.max_iterations,
        check_num=args.check_num,
        success_threshold=args.success_threshold,
        config_path=args.config,
    )
