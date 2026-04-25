"""Gemini client wrapper for closed_loop_cap.

Why a wrapper:
    - Keeps SDK import lazy so parsers/schema can be unit-tested without
      google-generativeai installed.
    - Centralizes API-key loading from gemini_api_key.json (user pastes key
      into that file; .gitignore keeps it out of git).
    - Adds uniform retry with exponential backoff for rate-limit / 5xx /
      transient errors (Layer 0 signals in failure_detection_and_recovery.md §9.1).
    - Uniform handling of vision payloads: raw PNG bytes in, text out.

SDK compatibility:
    Tested against `google-generativeai>=0.7`. The import is lazy so the
    presence of the SDK is only required at GeminiClient.__init__ time.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from closed_loop_cap.vlm.schema import VLMRequest, VLMResponse

logger = logging.getLogger(__name__)


class VLMCallError(RuntimeError):
    """Non-retryable error from the VLM client (bad key, unsupported input, etc.)."""


class VLMTransientError(RuntimeError):
    """Retryable error (rate limit, 5xx, network timeout)."""


def load_api_key(path: str | os.PathLike) -> str:
    """Read {"api_key": "..."} from a local JSON file.

    The file is intentionally gitignored. A *.example template ships with the
    repo so the user knows the expected shape.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"API key file not found at {p}. "
            f"Copy {p.name}.example to {p.name} and paste your key."
        )
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    key = payload.get("api_key")
    if not isinstance(key, str) or not key.strip() or key.strip().startswith("PASTE_"):
        raise VLMCallError(
            f"api_key in {p} is missing or still the placeholder. "
            f"Paste a real Gemini API key."
        )
    return key.strip()


def _rgb_to_png_bytes(rgb: np.ndarray, max_side: int | None = None) -> bytes:
    """Encode an HxWx3 uint8 RGB array to PNG bytes, optionally downscaled."""
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    if max_side is not None and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def encode_rgb(rgb: np.ndarray, max_side: int | None = None) -> bytes:
    """Public helper so callers don't import _rgb_to_png_bytes."""
    return _rgb_to_png_bytes(rgb, max_side=max_side)


@dataclass
class _RetryPolicy:
    max_retries: int = 3
    backoff_base_s: float = 2.0

    def sleep_for(self, attempt: int) -> float:
        return self.backoff_base_s * (2 ** attempt)


class GeminiClient:
    """Thin adapter around google.generativeai.

    Usage:
        client = GeminiClient(api_key_path="closed_loop_cap/gemini_api_key.json",
                              model="gemini-2.0-flash",
                              temperature=0.0,
                              max_output_tokens=4096,
                              max_retries=3, backoff_base_s=2.0)
        resp = client.call(VLMRequest(system=..., user_text=..., images=(png_bytes,)))
    """

    def __init__(
        self,
        api_key_path: str | os.PathLike,
        model: str,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        max_retries: int = 3,
        backoff_base_s: float = 2.0,
    ) -> None:
        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover
            raise VLMCallError(
                "google-generativeai is not installed. "
                "Run: pip install google-generativeai"
            ) from exc

        self._genai = genai
        self._api_key = load_api_key(api_key_path)
        genai.configure(api_key=self._api_key)
        self._model_name = model
        self._generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        self._retry = _RetryPolicy(max_retries=max_retries, backoff_base_s=backoff_base_s)

    def _build_parts(self, req: VLMRequest) -> list:
        """Assemble Gemini `Content.parts`: text + inline_data images."""
        parts: list = []
        if req.user_text:
            parts.append(req.user_text)
        for png in req.images:
            parts.append({"mime_type": "image/png", "data": png})
        return parts

    def call(self, req: VLMRequest) -> VLMResponse:
        """Synchronous call with retry on transient errors."""
        model = self._genai.GenerativeModel(
            self._model_name,
            system_instruction=req.system or None,
            generation_config=self._generation_config,
        )
        parts = self._build_parts(req)

        last_err: Exception | None = None
        for attempt in range(self._retry.max_retries + 1):
            try:
                raw = model.generate_content(parts)
                return self._to_response(raw)
            except Exception as exc:  # SDK raises various subclasses
                msg = str(exc).lower()
                transient = any(
                    tag in msg
                    for tag in ("rate", "429", "500", "502", "503", "504", "timeout", "deadline")
                )
                if not transient or attempt >= self._retry.max_retries:
                    logger.error(
                        "VLM call failed (attempt %d, transient=%s): %s",
                        attempt + 1,
                        transient,
                        exc,
                    )
                    raise VLMCallError(str(exc)) from exc
                wait_s = self._retry.sleep_for(attempt)
                logger.warning(
                    "VLM transient error (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1,
                    self._retry.max_retries,
                    exc,
                    wait_s,
                )
                last_err = exc
                time.sleep(wait_s)

        # Should be unreachable; loop either returns or raises.
        raise VLMCallError(f"exhausted retries: {last_err}")

    @staticmethod
    def _to_response(raw) -> VLMResponse:  # type: ignore[no-untyped-def]
        text = getattr(raw, "text", "") or ""
        finish_reason = "UNKNOWN"
        prompt_tokens = 0
        output_tokens = 0
        try:
            cand = raw.candidates[0] if getattr(raw, "candidates", None) else None
            if cand is not None and getattr(cand, "finish_reason", None) is not None:
                finish_reason = str(cand.finish_reason)
        except Exception:
            pass
        try:
            um = getattr(raw, "usage_metadata", None)
            if um is not None:
                prompt_tokens = int(getattr(um, "prompt_token_count", 0) or 0)
                output_tokens = int(getattr(um, "candidates_token_count", 0) or 0)
        except Exception:
            pass
        return VLMResponse(
            raw_text=text,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )
