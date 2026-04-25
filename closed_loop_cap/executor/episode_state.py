"""Per-episode state that crosses subtask boundaries.

Currently we track *which actor each arm is holding* so `place` post-conditions
can verify both (a) that the previously grasped object was released and (b)
that it actually ended up near the destination — neither of which is derivable
from a single subtask snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EpisodeState:
    """Mutable state shared across all subtask judgements in one episode."""

    # arm_tag ("left" / "right") → actor reference ("self.hammer")
    held_by_arm: dict[str, str] = field(default_factory=dict)

    def on_grasp_success(self, arm_tag: str, target_actor: str) -> None:
        self.held_by_arm[arm_tag] = target_actor

    def on_place_success(self, arm_tag: str) -> None:
        self.held_by_arm.pop(arm_tag, None)

    def held_by(self, arm_tag: str) -> str | None:
        return self.held_by_arm.get(arm_tag)

    def snapshot(self) -> dict:
        return {"held_by_arm": dict(self.held_by_arm)}
