VLM_SYSTEM_PROMPT = """You are a robot task execution analysis expert.
You analyze image sequences from robot manipulation tasks and provide detailed feedback
about the execution quality, identifying problems and suggesting improvements."""


def build_vlm_analysis_prompt(task_name, task_info, step_names, waypoints_json=None):
    """
    VLM 분석용 프롬프트 생성.

    Args:
        task_name: 태스크 이름
        task_info: {"description": ..., "goal": ...}
        step_names: 캡처된 스텝 이름 리스트
        waypoints_json: 현재 웨이포인트 (선택)

    Returns:
        str: VLM 프롬프트
    """
    prompt = f"""Analyze the execution of the following robot task:

Task name: {task_name}
Task description: {task_info.get('description', 'No description provided')}
Task goal: {task_info.get('goal', 'No goal provided')}

You will be shown images from each step of the task execution. Please analyze:

1. **Motion Quality**: Is the robot's motion smooth and natural? Are there any jerky or unnatural movements?
2. **Task Semantics**: Does the motion match what the task requires? For example, if the task is "shake bottle", does the shaking motion look realistic?
3. **Trajectory Issues**: Are there any trajectory problems like:
   - Unnecessary detours or excessive movements
   - Insufficient amplitude (e.g., shaking too little)
   - Wrong direction or orientation
   - Collisions or near-misses
4. **Success Assessment**: Was the task completed successfully? If not, what went wrong?

Execution steps shown: {', '.join(step_names)}
"""

    if waypoints_json:
        prompt += f"""
Current Cartesian waypoints (EE poses as [x, y, z, qw, qx, qy, qz]):
```json
{waypoints_json}
```

Please suggest specific modifications to these waypoints to improve the execution.
Focus on:
- Which waypoint indices need adjustment
- What direction/magnitude of change would help
- Any waypoints that should be added or removed
"""

    return prompt
