import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import importlib
import yaml
import traceback

from trajectory_refinement.initial_trajectory.extract_ee_poses_from_traj import extract_ee_poses


# 태스크별 config 로딩 (code_gen/test_gen_code.py의 setup_task_config 로직 재사용)
CONFIGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "task_config")


def _get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def _create_task_config(task_config_path, task_name):
    """태스크 config 템플릿에서 새 config 생성."""
    template = {
        "task_name": task_name,
        "render_freq": 0,
        "episode_num": 10,
        "use_seed": False,
        "save_freq": 15,
        "embodiment": ["aloha-agilex"],
        "language_num": 5,
        "domain_randomization": {
            "random_background": False,
            "cluttered_table": False,
            "clean_background_rate": 1,
            "random_head_camera_dis": 0,
            "random_table_height": 0,
            "random_light": False,
            "crazy_random_light_rate": 0,
        },
        "camera": {
            "head_camera_type": "D435",
            "wrist_camera_type": "D435",
            "collect_head_camera": True,
            "collect_wrist_camera": True,
        },
        "data_type": {
            "rgb": True,
            "third_view": False,
            "depth": False,
            "pointcloud": False,
            "endpose": True,
            "qpos": True,
            "mesh_segmentation": False,
            "actor_segmentation": False,
        },
        "pcd_down_sample_num": 1024,
        "pcd_crop": True,
        "save_path": "./data",
        "clear_cache_freq": 5,
        "collect_data": True,
        "eval_video_log": True,
    }
    with open(task_config_path, "w") as f:
        yaml.dump(template, f, default_flow_style=False, sort_keys=False)
    print(f"[_create_task_config] Created config: {task_config_path}")


def _setup_task_args(task_name):
    """
    태스크 config와 로봇 config를 로딩하여 args dict를 구성.
    code_gen/test_gen_code.py의 setup_task_config와 동일한 로직.
    """
    task_config_path = os.path.join(CONFIGS_PATH, f"{task_name}.yml")
    if not os.path.isfile(task_config_path):
        _create_task_config(task_config_path, task_name)
        print(f"[_setup_task_args] Created missing config: {task_config_path}")

    with open(task_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["domain_randomization"] = {
        "random_background": False,
        "cluttered_table": False,
        "clean_background_rate": 0.0,
        "random_head_camera_dis": 0,
        "random_table_height": 0.0,
        "random_light": False,
        "crazy_random_light_rate": 0.0,
        "random_embodiment": False,
    }

    embodiment_type = args.get("embodiment")
    # 문자열이면 리스트로 변환 (호환성)
    if isinstance(embodiment_type, str):
        embodiment_type = [embodiment_type]
    args["task_name"] = task_name
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(etype):
        robot_file = _embodiment_types[etype]["file_path"]
        if robot_file is None:
            raise Exception("No embodiment files")
        return robot_file if os.path.isabs(robot_file) else os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", robot_file)
        )

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise Exception("Embodiment items should be 1 or 3")

    args["left_embodiment_config"] = _get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = _get_embodiment_config(args["right_robot_file"])
    args["need_plan"] = True
    args["save_path"] = "./data/test"

    return args


def _create_refinable_task(task_name):
    """
    태스크 클래스를 동적으로 로드하고 BaseTaskWithIKExecution과 다중 상속하여
    Refinable 클래스를 생성.
    """
    from trajectory_refinement.task_env_wrapper.base_task_with_ik_execution import BaseTaskWithIKExecution

    # 기존 envs에서 태스크 클래스 로드
    envs_module = importlib.import_module(f"envs.{task_name}")
    importlib.reload(envs_module)
    task_class = getattr(envs_module, task_name)

    # 동적으로 Refinable 클래스 생성 (MRO: BaseTaskWithIKExecution → task_class → Base_Task)
    refinable_class = type(
        f"Refinable_{task_name}",
        (BaseTaskWithIKExecution, task_class),
        {},
    )

    return refinable_class()


def run_initial_trajectory(task_name, seed=0, record_ee_freq=15, max_seed_retries=20):
    """
    기존 play_once()를 실행하여 초기 궤적 생성.
    실행 중 EE 포즈를 기록.

    Args:
        task_name: 태스크 이름 (예: "shake_bottle")
        seed: 시작 랜덤 시드
        record_ee_freq: EE 포즈 기록 빈도 (매 N 스텝)
        max_seed_retries: 불안정 시드 재시도 최대 횟수

    Returns:
        tuple: (task_env, left_ee_waypoints, right_ee_waypoints, task_info, args)
    """
    args = _setup_task_args(task_name)

    for attempt_seed in range(seed, seed + max_seed_retries):
        task_env = _create_refinable_task(task_name)
        task_env._record_ee_freq = record_ee_freq

        try:
            task_env.setup_demo(now_ep_num=0, seed=attempt_seed, **args)
            break
        except Exception as e:
            print(f"[run_initial_trajectory] seed={attempt_seed} failed setup: {e}")
            try:
                task_env.close()
            except:
                pass
            if attempt_seed == seed + max_seed_retries - 1:
                raise RuntimeError(f"All {max_seed_retries} seeds failed for {task_name}") from e
            continue

    # play_once 실행 (기존 IK+RRT 사용)
    task_info = task_env.play_once()

    # EE 궤적 추출
    left_waypoints = extract_ee_poses(task_env, "left")
    right_waypoints = extract_ee_poses(task_env, "right")

    # 성공 여부 확인
    success = task_env.plan_success and task_env.check_success()
    print(f"[run_initial_trajectory] seed={attempt_seed}, success={success}")
    print(f"[run_initial_trajectory] EE waypoints: left={len(left_waypoints)}, right={len(right_waypoints)}")

    return task_env, left_waypoints, right_waypoints, task_info, args
