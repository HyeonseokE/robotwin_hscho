"""Planner system prompt (Layer 1 in failure_detection_and_recovery.md).

The planner gets the initial scene image + task instruction + actor list and
returns an ordered list of subtasks. Each subtask is one atomic skill call;
we keep skill_type restricted to six primitives the executor actually supports.
"""

from __future__ import annotations

from closed_loop_cap.vlm.schema import ALLOWED_ARM_TAGS, ALLOWED_SKILL_TYPES

PLANNER_SYSTEM_PROMPT = """\
You are the task planner for a bimanual robot simulator (RoboTwin / SAPIEN).
You will be shown:
  - the initial RGB image of the scene (head camera, top-down-ish view),
  - a task instruction in natural language,
  - the list of actors (objects) present in the scene.

World frame: +x = right, +y = forward, +z = up. Distance unit is meters.

Your job: decompose the task into a short ordered list of SUBTASKS. Each subtask
must be executable by exactly ONE call to one of the allowed skills. Do not plan
motions at the joint level — that is the next layer's responsibility.

Allowed skill_type values (STRICT whitelist):
  - "grasp"                : close a gripper on a target actor
  - "place"                : release an actor at / onto a target location
  - "move_to_pose"         : move the end-effector to an absolute 7-D pose
  - "move_by_displacement" : move the end-effector by a relative displacement
  - "open_gripper"         : open a gripper (no target actor; still provide one relevant to the step)
  - "close_gripper"        : close a gripper (same caveat)

Allowed arm_tag values: "left", "right".

Output format — you MUST respond with a single ```json fenced block that
matches this schema exactly (no extra keys, no trailing prose):

```json
{
  "subtasks": [
    {
      "id": 1,
      "instruction": "short natural-language description of this step",
      "skill_type": "grasp",
      "target_actor": "self.<name_from_actor_list>",
      "arm_tag": "left",
      "success_hint": "one short sentence describing how we will know this step succeeded"
    }
  ]
}
```

Hard rules:
  - id starts at 1 and is contiguous.
  - target_actor MUST be one of the provided actor keys, prefixed with "self."
    (e.g. if actor_list has "hammer", use "self.hammer"). Never invent an actor.
  - skill_type MUST be one of the six values above.
  - arm_tag MUST be "left" or "right".
  - Keep the list short (typically 2–6 subtasks). Do not add decorative steps.
  - Prefer the arm that is geometrically closer to the target when you can tell
    from the image (objects on the left half → left arm, and vice versa).
"""


def build_planner_user_text(
    task_instruction: str,
    actor_names: list[str],
) -> str:
    """Compose the user-role text that accompanies the initial image."""
    actor_block = "\n".join(f"  - {n}" for n in actor_names) if actor_names else "  (none provided)"
    return (
        f"Task instruction:\n  {task_instruction.strip()}\n\n"
        f"Actors in the scene (reference as self.<name>):\n{actor_block}\n\n"
        f"Allowed skill_type: {list(ALLOWED_SKILL_TYPES)}\n"
        f"Allowed arm_tag: {list(ALLOWED_ARM_TAGS)}\n\n"
        f"Return ONLY the JSON block described in the system instructions."
    )


def build_planner_retry_text(
    task_instruction: str,
    actor_names: list[str],
    previous_errors: list[str],
) -> str:
    """Retry prompt that tells the planner why the last output was rejected."""
    base = build_planner_user_text(task_instruction, actor_names)
    errs = "\n".join(f"  - {e}" for e in previous_errors)
    return (
        f"{base}\n\n"
        f"Your previous response was rejected. Issues:\n{errs}\n\n"
        f"Fix ALL of the above and resend a single valid JSON block."
    )
