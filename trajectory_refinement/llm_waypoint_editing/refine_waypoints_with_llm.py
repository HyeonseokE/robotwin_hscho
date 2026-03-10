import sys
import os
import json
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from trajectory_refinement.prompts.llm_waypoint_edit_prompt import build_waypoint_edit_prompt


def refine_waypoints(task_info, current_waypoints, vlm_feedback, ik_fail_indices,
                     messages, arm_tag="left"):
    """
    DeepSeek LLM에게 VLM 피드백 + 현재 웨이포인트 + IK 실패 정보를 제공하고
    수정된 웨이포인트 JSON을 받음.

    Args:
        task_info: {"description": ..., "goal": ...}
        current_waypoints: list of [x,y,z,qw,qx,qy,qz]
        vlm_feedback: VLM 분석 피드백 텍스트
        ik_fail_indices: IK 실패한 웨이포인트 인덱스 리스트
        messages: 대화 히스토리 (list of {"role": ..., "content": ...})
        arm_tag: "left" or "right"

    Returns:
        tuple: (refined_waypoints, updated_messages)
            - refined_waypoints: list of [x,y,z,qw,qx,qy,qz]
            - updated_messages: 업데이트된 대화 히스토리
    """
    from code_gen.gpt_agent import generate

    # 프롬프트 구성
    user_prompt = build_waypoint_edit_prompt(
        task_info=task_info,
        current_waypoints=current_waypoints,
        vlm_feedback=vlm_feedback,
        ik_fail_indices=ik_fail_indices,
        arm_tag=arm_tag,
    )

    # 대화 히스토리에 추가
    messages.append({"role": "user", "content": user_prompt})

    # LLM 호출
    response = generate(messages, gpt="deepseek", temperature=0.2)

    # 대화 히스토리 업데이트
    messages.append({"role": "assistant", "content": response})

    # JSON 파싱
    refined_waypoints = _parse_waypoints_from_response(response)

    if refined_waypoints is None:
        print("[refine_waypoints] Failed to parse waypoints from LLM response, using current waypoints")
        return current_waypoints, messages

    return refined_waypoints, messages


def _parse_waypoints_from_response(response):
    """
    LLM 응답에서 웨이포인트 JSON을 파싱.
    """
    # JSON 블록 추출 시도
    json_patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
        r'\[\s*\[[\s\S]*?\]\s*\]',
    ]

    for pattern in json_patterns:
        match = re.search(pattern, response)
        if match:
            try:
                json_str = match.group(1) if '```' in pattern else match.group(0)
                data = json.loads(json_str)

                # {"waypoints": [...]} 형식 처리
                if isinstance(data, dict) and "waypoints" in data:
                    data = data["waypoints"]

                # 유효성 검사
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], list) and len(data[0]) == 7:
                        return data

            except json.JSONDecodeError:
                continue

    return None
