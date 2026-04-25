"""Subtask code-generator prompt (Layer 2 in failure_detection_and_recovery.md).

Adopts as much of the upstream `code_gen/prompt.py` BASIC_INFO contract as
fits the snippet-only execution model so the VLM does not invent its own
coordinate / Pose conventions.
"""

from __future__ import annotations

import json

from closed_loop_cap.vlm.schema import SubtaskSpec

# Lifted (and condensed) from code_gen/prompt.py BASIC_INFO so the codegen
# Gemini call shares the same world-frame / Pose vocabulary as the original
# DeepSeek pipeline. We omit irrelevant lines (dist=1m is implicit in SAPIEN).
_POSE_AND_FRAME_RULES = """\
Pose and coordinate conventions:
  - Distances are in meters.
  - Pose is a 7-D vector [x, y, z, qw, qx, qy, qz]. For a sapien.Pose object,
    `.p` returns [x, y, z] and `.q` returns [qw, qx, qy, qz].
  - World frame: +x = right, +y = forward, +z = up.
  - Each actor exposes interaction points via
        `actor.get_functional_point(point_id: int, return_type: str)`
    where `return_type` is one of "pose" (sapien.Pose), "p" (xyz), or "q" (quat).
  - The `Actor` instances are accessed as `self.<actor_name>` on the task env;
    the Available Actors block below enumerates the legal names plus their
    `functional_points` (use these IDs verbatim; do not invent new IDs).
"""

# Domain-specific failure hints distilled from
# code_gen/task_generation.py:176-182 — common place_actor pitfalls. Injected
# only on retries so the first-attempt prompt stays compact.
DOMAIN_HINTS_PLACE = """\
If a previous place_actor attempt failed, common causes are:
  1. pre_dis_axis is wrong — try {'grasp', 'fp'} or an explicit np.ndarray.
  2. functional_point_id is wrong — pick another id from the Actor's points.
  3. pre_dis / dis are too small or too large — adjust by ±0.02m.
  4. constrain is wrong — toggle between 'free' and 'align'.
  5. The instruction had an explicit note (e.g., "do not open gripper after place")
     that was ignored — re-read the note before regenerating.
"""

CODEGEN_SYSTEM_PROMPT = f"""\
You generate a short Python snippet that executes ONE subtask inside a
bimanual robot simulator (RoboTwin / SAPIEN).

Environment contract:
  - Your code runs with `self` already bound to a Base_Task instance.
  - You also have `np` (numpy), `sapien`, `Pose` (alias of sapien.Pose),
    and `ArmTag` (constructor) in scope. No other imports are allowed.
  - You MUST trigger physical motion via `self.move(...)`; just building an
    action tuple without calling move does nothing.
  - Arm tags MUST be constructed as `ArmTag("left")` or `ArmTag("right")` —
    never pass raw strings to skill functions expecting ArmTag.
  - Reference objects as `self.<name>` using only the actor names provided.

{_POSE_AND_FRAME_RULES}
Forbidden:
  - `import` statements, `__import__`, `open(`, `eval`, `exec`, `compile`,
    filesystem/network calls, subprocess, `os.*`, `sys.*`.
  - Attribute access to dunder names like `__class__`, `__globals__`, etc.
  - Multi-step loops that re-plan; emit one call chain per subtask.

Output format — respond with a SINGLE ```python fenced block and nothing
else. No prose, no comments outside the block, no multiple blocks. Example:

```python
self.move(self.grasp_actor(self.hammer, arm_tag=ArmTag("left")))
```
"""


def _format_actor_details(actor_details: dict) -> str:
    """Render enriched actor metadata (name + functional/contact points) as
    compact JSON so the VLM can pick `functional_point_id` with grounding."""
    if not actor_details:
        return "(no per-actor details available)"
    return json.dumps(actor_details, indent=2, ensure_ascii=False, default=str)


def build_codegen_user_text(
    subtask: SubtaskSpec,
    actor_names: list[str],
    skill_catalog: str,
    *,
    actor_details: dict | None = None,
    full_task_description: str | None = None,
    previous_failure_hint: str | None = None,
    domain_hints: str | None = None,
) -> str:
    """Compose the user-role text that accompanies the current RGB frame."""
    actors = ", ".join(f"self.{n}" for n in actor_names) if actor_names else "(none)"
    sections: list[str] = []

    if full_task_description:
        sections += [
            "## Original task instruction (whole episode)",
            full_task_description.strip(),
            "",
        ]

    sections += [
        f"## Current subtask #{subtask.id}",
        f"instruction:    {subtask.instruction}",
        f"skill_type:     {subtask.skill_type}",
        f"arm_tag:        {subtask.arm_tag}",
        f"target_actor:   {subtask.target_actor}",
        f"success_hint:   {subtask.success_hint}",
        "",
        f"## Available actor names: {actors}",
        "",
        "## Available actor details (functional / contact points)",
        _format_actor_details(actor_details or {}),
        "",
        "## Skill catalog",
        skill_catalog,
    ]

    if previous_failure_hint:
        sections += [
            "",
            "## Previous attempt failed — incorporate this feedback",
            previous_failure_hint,
        ]
        if domain_hints:
            sections += ["", domain_hints]

    sections += [
        "",
        "Generate the Python snippet for THIS subtask only. "
        "Remember: exactly one fenced ```python block.",
    ]
    return "\n".join(sections)
