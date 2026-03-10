import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
# observation_agent.py uses "from gpt_agent import ..." which requires code_gen/ in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "code_gen"))


def analyze_execution(episode_id, task_name, task_info, current_waypoints_json,
                      save_dir="./camera_images", iteration_id=0):
    """
    기존 observation_agent.observe_task_execution()을 재사용하여 VLM 분석.
    current_waypoints_json을 "problematic_code" 인자로 전달하여
    Kimi VLM이 키프레임을 분석하고 피드백 텍스트를 반환.

    Args:
        episode_id: 에피소드 번호
        task_name: 태스크 이름
        task_info: {"description": ..., "goal": ...}
        current_waypoints_json: 현재 웨이포인트 (dict 또는 list)
        save_dir: 이미지 저장 기본 디렉토리
        iteration_id: 반복 루프 ID

    Returns:
        str: VLM 피드백 텍스트
    """
    from code_gen.observation_agent import observe_task_execution

    # 웨이포인트를 문자열로 변환
    if isinstance(current_waypoints_json, str):
        waypoints_str = current_waypoints_json
    else:
        waypoints_str = json.dumps(current_waypoints_json, indent=2)

    return observe_task_execution(
        episode_id=episode_id,
        task_name=task_name,
        task_info=task_info,
        problematic_code=waypoints_str,
        save_dir=save_dir,
        generate_dir_name=f"iteration_{iteration_id}",
    )
