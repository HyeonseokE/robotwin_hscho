"""Subtask timeline IO + step→label resolver.

Phase 1 writes a shallow subtask listing to subtask_timeline.json; Phase 2
loads it and resolves which subtask is active at a given replay step.

In the MVP the per-subtask step boundaries are not persisted (see
run_closed_loop.py TODO comment). The resolver therefore distributes steps
evenly across completed subtasks; this is approximate but enough to seed
label injection while a more precise capture lands in P5-followup.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SubtaskEntry:
    subtask_id: int | None
    natural_language: str
    skill_type: str
    target_actor: str
    arm_tag: str
    success_hint: str
    final_ok: bool
    # Populated by SubtaskTimeline.assign_step_ranges():
    step_start: int = 0
    step_end: int = 0


@dataclass
class SubtaskTimeline:
    subtasks: list[SubtaskEntry] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "SubtaskTimeline":
        raw = json.loads(Path(path).read_text())
        entries = [
            SubtaskEntry(
                subtask_id=d.get("subtask_id"),
                natural_language=d.get("natural_language", ""),
                skill_type=d.get("skill_type", ""),
                target_actor=d.get("target_actor", ""),
                arm_tag=d.get("arm_tag", ""),
                success_hint=d.get("success_hint", ""),
                final_ok=bool(d.get("final_ok", False)),
            )
            for d in raw.get("subtasks", [])
        ]
        return cls(subtasks=entries)

    def assign_step_ranges(self, total_steps: int) -> None:
        """Evenly partition `total_steps` across subtasks. MVP approximation."""
        n = len(self.subtasks)
        if n == 0 or total_steps <= 0:
            return
        chunk = max(1, total_steps // n)
        for i, entry in enumerate(self.subtasks):
            entry.step_start = i * chunk
            entry.step_end = (i + 1) * chunk if i < n - 1 else total_steps

    def resolve(self, step_idx: int) -> SubtaskEntry | None:
        for entry in self.subtasks:
            if entry.step_start <= step_idx < entry.step_end:
                return entry
        return self.subtasks[-1] if self.subtasks else None


def write_subtask_timeline(path: Path, subtask_records: list[dict]) -> None:
    """Serialize a list of plain-dict subtask records (from EpisodeReport)."""
    Path(path).write_text(json.dumps({"subtasks": subtask_records}, indent=2, default=str))
