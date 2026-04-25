"""Helpers for reading hand-curated task metadata from code_gen/task_info.py.

We import that module (read-only) and search its module-level UPPERCASE dicts
for one whose 'task_name' matches. The dict carries task_description and
actor_list — both fed into planner / codegen prompts.

If `code_gen.test_gen_code.enrich_actors` is importable, we also pull
functional_points / contact_points from each actor's `points_info.json`. This
mirrors the upstream CaP pipeline so codegen can pick `functional_point_id`
and `contact_point_id` with real geometric grounding instead of guessing.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskMeta:
    task_name: str
    description: str
    actor_names: tuple[str, ...]   # bare names, no "self." prefix
    actor_details: dict = field(default_factory=dict)
    """Enriched actor dict keyed by 'self.<name>'. Each entry mirrors the
    upstream `enrich_actors()` output: name + description + functional_points
    (with id/position/direction/description) + contact_points. Empty when
    points_info.json is unavailable or enrich_actors fails."""


def _strip_self_prefix(key: str) -> str:
    return key[5:] if key.startswith("self.") else key


def _try_enrich(actor_list: dict) -> dict:
    """Call upstream enrich_actors() if it is importable. On any failure
    fall back to the bare actor_list (still usable, just without points)."""
    if not isinstance(actor_list, dict) or not actor_list:
        return {}
    try:
        from code_gen.test_gen_code import enrich_actors  # type: ignore
    except Exception as exc:  # noqa: BLE001
        logger.warning("enrich_actors unavailable (%s); using bare actor_list", exc)
        return dict(actor_list)
    try:
        return dict(enrich_actors(actor_list))
    except Exception as exc:  # noqa: BLE001
        logger.warning("enrich_actors failed (%s); using bare actor_list", exc)
        return dict(actor_list)


def load_task_meta(task_name: str) -> TaskMeta:
    """Look up TASK_NAME_UPPER in code_gen.task_info; fall back to stub.

    The stub keeps the orchestrator usable for tasks not covered in task_info.py
    (planner will just see an empty actor list and rely on the image).
    """
    try:
        module = importlib.import_module("code_gen.task_info")
    except Exception as exc:
        logger.warning("code_gen.task_info not importable (%s); using stub meta", exc)
        return TaskMeta(
            task_name=task_name,
            description=f"Complete the {task_name} task.",
            actor_names=(),
        )

    for attr in dir(module):
        if not attr.isupper():
            continue
        val = getattr(module, attr)
        if not isinstance(val, dict):
            continue
        if val.get("task_name") != task_name:
            continue

        desc = str(val.get("task_description", "")).strip() or f"Complete the {task_name} task."
        actor_list = val.get("actor_list", {})
        if isinstance(actor_list, dict):
            names = tuple(_strip_self_prefix(k) for k in actor_list.keys())
            details = _try_enrich(actor_list)
        elif isinstance(actor_list, (list, tuple)):
            names = tuple(_strip_self_prefix(str(k)) for k in actor_list)
            details = {}
        else:
            names = ()
            details = {}
        return TaskMeta(
            task_name=task_name,
            description=desc,
            actor_names=names,
            actor_details=details,
        )

    logger.warning("task_info.py has no entry for %s; using stub", task_name)
    return TaskMeta(
        task_name=task_name,
        description=f"Complete the {task_name} task.",
        actor_names=(),
    )
