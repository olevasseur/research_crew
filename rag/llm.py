"""LLM generation client via Ollama chat API."""

from __future__ import annotations

import requests

from .config import GenerationConfig


class LLMClient:
    """Thin wrapper around Ollama's /api/chat endpoint.

    Designed to be swappable: replace this class to point at a different
    backend (OpenAI-compatible, Anthropic, etc.) without changing callers.
    """

    def __init__(self, config: GenerationConfig):
        self.model = config.model
        self.base_url = config.base_url.rstrip("/")

    # TODO v2: add temperature, max_tokens, and retry/backoff controls
    def generate(self, prompt: str, system: str = "") -> str:
        """Send a prompt (with optional system message) and return the response text."""
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False},
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
