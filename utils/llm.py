"""Minimal OpenAI wrapper with graceful degradation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.config import AppConfig
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime
    OpenAI = None  # type: ignore[assignment]


@dataclass(slots=True)
class LLMResult:
    """Return value for optional LLM calls."""

    text: str | None
    provider: str
    used_fallback: bool


class OpenAIClient:
    """Compatibility wrapper around the OpenAI Python SDK."""

    def __init__(self, settings: AppConfig) -> None:
        self.settings = settings
        self._client = (
            OpenAI(api_key=settings.openai_api_key)
            if settings.openai_enabled and OpenAI is not None
            else None
        )

    @property
    def available(self) -> bool:
        """Return whether the client can be used."""

        return self._client is not None

    def generate_text(self, instructions: str, user_input: str) -> LLMResult:
        """Generate text with the OpenAI API when configured."""

        if not self._client:
            return LLMResult(text=None, provider="disabled", used_fallback=True)

        try:
            response = self._client.responses.create(
                model=self.settings.openai_model,
                instructions=instructions,
                input=user_input,
            )
            text = getattr(response, "output_text", "") or ""
            if text.strip():
                return LLMResult(text=text.strip(), provider="responses", used_fallback=False)
        except Exception as exc:  # pragma: no cover - network/runtime variability
            logger.warning("Responses API failed, falling back to chat completions: %s", exc)

        try:
            completion = self._client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "developer", "content": instructions},
                    {"role": "user", "content": user_input},
                ],
            )
            message = completion.choices[0].message.content if completion.choices else None
            return LLMResult(
                text=(message or "").strip() or None,
                provider="chat.completions",
                used_fallback=False,
            )
        except Exception as exc:  # pragma: no cover - network/runtime variability
            logger.warning("OpenAI request failed: %s", exc)
            return LLMResult(text=None, provider="failed", used_fallback=True)
