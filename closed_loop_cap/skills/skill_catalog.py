"""Expose the existing RoboTwin skill catalog to the codegen prompt.

We import (read-only) the AVAILABLE_ENV_FUNCTION / FUNCTION_EXAMPLE dicts from
code_gen/prompt.py and render them into a compact Markdown block the VLM can
consume. We never mutate those dicts and never re-implement the skills — this
module is a presentation layer.
"""

from __future__ import annotations

import importlib
import textwrap


def _load_upstream() -> tuple[dict, str]:
    """Import code_gen.prompt lazily so closed_loop_cap has no hard dep at import time."""
    module = importlib.import_module("code_gen.prompt")
    api = getattr(module, "AVAILABLE_ENV_FUNCTION", {}) or {}
    example = getattr(module, "FUNCTION_EXAMPLE", "") or ""
    if not isinstance(api, dict):
        raise TypeError("code_gen.prompt.AVAILABLE_ENV_FUNCTION must be a dict")
    if not isinstance(example, str):
        example = str(example)
    return dict(api), example


def format_skill_catalog() -> str:
    """Render AVAILABLE_ENV_FUNCTION + FUNCTION_EXAMPLE for prompt injection."""
    api, example = _load_upstream()
    lines = ["## Available skills (call via self.move(...) and the helpers below):"]
    for name, desc in api.items():
        cleaned = " ".join(s.strip() for s in str(desc).split())
        lines.append(f"- `{name}` — {cleaned}")
    if example.strip():
        lines.append("")
        lines.append("## Example usage:")
        lines.append("```python")
        lines.append(textwrap.dedent(example).strip())
        lines.append("```")
    return "\n".join(lines)


try:
    SKILL_CATALOG_TEXT: str = format_skill_catalog()
except Exception:  # pragma: no cover — deferred until first use in restricted envs
    SKILL_CATALOG_TEXT = ""
