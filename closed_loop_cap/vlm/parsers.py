"""Tolerant extractors for VLM outputs.

Gemini (and LLMs in general) wrap structured outputs inconsistently: sometimes
in ```json fences, sometimes in ```python, sometimes bare, sometimes with
trailing commentary. These helpers normalize that.

All parse errors raise ParseError so the retry loop can route them as
L1-S1 (JSON) or L2-S1 (code) signals.
"""

from __future__ import annotations

import json
import re

_CODE_FENCE_RE = re.compile(
    r"```(?P<lang>[a-zA-Z0-9_+-]*)\s*\n(?P<body>.*?)```",
    re.DOTALL,
)
_FIRST_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class ParseError(ValueError):
    """Raised when a VLM response cannot be parsed into the expected shape."""


def _extract_fenced_block(raw: str, preferred_langs: tuple[str, ...]) -> str | None:
    """Return the body of the first matching fenced block, or None."""
    matches = list(_CODE_FENCE_RE.finditer(raw))
    if not matches:
        return None
    for m in matches:
        if m.group("lang").lower() in preferred_langs:
            return m.group("body").strip()
    # Fall back to the first fence regardless of language tag.
    return matches[0].group("body").strip()


def parse_json_response(raw: str) -> dict:
    """Extract a JSON object from a possibly fenced, possibly verbose response.

    Priority:
        1. ```json fenced block
        2. Any fenced block
        3. First balanced {...} substring in the raw text

    Raises:
        ParseError: no parseable JSON object found, or JSON is malformed.
    """
    if not raw or not raw.strip():
        raise ParseError("empty response")

    candidate = _extract_fenced_block(raw, preferred_langs=("json",))
    if candidate is None:
        bare = _FIRST_JSON_OBJECT_RE.search(raw)
        if bare is None:
            raise ParseError("no JSON object or fenced block found in response")
        candidate = bare.group(0)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ParseError(f"JSON decode error: {exc.msg} (pos {exc.pos})") from exc


def parse_code_snippet(raw: str) -> str:
    """Extract a Python code snippet from a fenced response.

    Priority:
        1. ```python / ```py fenced block
        2. Any fenced block
        3. Entire response if it has no fence AND looks code-ish
           (contains at least one `def ` or `self.` token).

    Raises:
        ParseError: no code block extractable.
    """
    if not raw or not raw.strip():
        raise ParseError("empty response")

    candidate = _extract_fenced_block(raw, preferred_langs=("python", "py"))
    if candidate is not None:
        return candidate

    stripped = raw.strip()
    if "def " in stripped or "self." in stripped:
        return stripped

    raise ParseError("no fenced code block found and content does not look like code")
