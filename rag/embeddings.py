"""Ollama embedding client with a ChromaDB-compatible interface."""

from __future__ import annotations

import requests

from .config import EmbeddingConfig


class OllamaEmbedder:
    """Wraps the Ollama embeddings API."""

    MAX_CHARS = 6000  # safe limit for nomic-embed-text (8192-token context)

    def __init__(self, config: EmbeddingConfig):
        self.model = config.model
        self.base_url = config.base_url.rstrip("/")

    def embed(self, text: str) -> list[float]:
        """Embed a single text string, truncating if needed."""
        if len(text) > self.MAX_CHARS:
            text = text[: self.MAX_CHARS]
        resp = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts sequentially."""
        # TODO v2: use /api/embed batch endpoint when stable across Ollama versions
        return [self.embed(t) for t in texts]


class ChromaOllamaEmbedding:
    """Adapter so ChromaDB can call our Ollama embedder transparently.

    Implements the methods ChromaDB >=1.0 expects on embedding functions:
    name(), get_config(), __call__().
    """

    def __init__(self, embedder: OllamaEmbedder):
        self._embedder = embedder

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._embedder.embed_batch(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """ChromaDB 1.x calls this for query-time embedding."""
        return self.__call__(input)

    def name(self) -> str:
        return f"ollama/{self._embedder.model}"

    def get_config(self) -> dict:
        return {
            "name": self.name(),
            "model": self._embedder.model,
            "base_url": self._embedder.base_url,
        }
