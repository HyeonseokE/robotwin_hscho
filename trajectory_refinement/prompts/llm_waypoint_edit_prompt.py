import json

LLM_SYSTEM_PROMPT = """You are a robot trajectory optimization expert.
Your job is to refine Cartesian waypoints for a robot arm to improve task execution quality.

Each waypoint is a 7D vector: [x, y, z, qw, qx, qy, qz]
- x, y, z: position in meters (world frame)
- qw, qx, qy, qz: orientation as unit quaternion

When modifying waypoints:
1. Make small, targeted adjustments (typically < 0.05m position, < 0.1 quaternion)
2. Preserve the overall trajectory structure
3. Ensure smooth transitions between consecutive waypoints
4. Keep orientations as valid unit quaternions (norm ≈ 1)
5. Consider the robot's workspace limits

Always respond with the complete modified waypoints as a JSON array."""


def build_waypoint_edit_prompt(task_info, current_waypoints, vlm_feedback,
                               ik_fail_indices, arm_tag="left"):
    """
    LLM 웨이포인트 수정 프롬프트 생성.

    Args:
        task_info: {"description": ..., "goal": ...}
        current_waypoints: list of [x,y,z,qw,qx,qy,qz]
        vlm_feedback: VLM 분석 피드백 텍스트
        ik_fail_indices: IK 실패한 웨이포인트 인덱스 리스트
        arm_tag: "left" or "right"

    Returns:
        str: LLM 프롬프트
    """
    waypoints_json = json.dumps(current_waypoints, indent=2)

    prompt = f"""## Task Information
- Task description: {task_info.get('description', 'N/A')}
- Task goal: {task_info.get('goal', 'N/A')}
- Active arm: {arm_tag}

## VLM Execution Feedback
{vlm_feedback}

## Current Waypoints ({len(current_waypoints)} points)
Each waypoint: [x, y, z, qw, qx, qy, qz]
```json
{waypoints_json}
```

## IK Failure Information
"""

    if ik_fail_indices:
        prompt += f"""The following waypoint indices failed IK solving: {ik_fail_indices}
These waypoints may be unreachable for the robot. Consider:
- Adjusting their positions to be within the robot's workspace
- Modifying orientations to be more achievable
- Removing them if they are not essential
"""
    else:
        prompt += "All waypoints passed IK solving successfully.\n"

    prompt += """
## Instructions
Based on the VLM feedback and IK failure information, please modify the waypoints to:
1. Fix any identified execution problems
2. Resolve IK failures by adjusting unreachable waypoints
3. Improve the overall motion quality for this specific task

Respond with the complete modified waypoints as a JSON array:
```json
[
  [x, y, z, qw, qx, qy, qz],
  ...
]
```
"""

    return prompt
