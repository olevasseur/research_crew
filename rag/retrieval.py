"""Retrieval layer: wraps the vector store with formatting for LLM prompts."""

from __future__ import annotations

from .config import RAGConfig, RetrievalConfig
from .embeddings import OllamaEmbedder
from .store import VectorStore


class Retrieval:
    """Query interface over the vector store."""

    def __init__(self, config: RAGConfig):
        embedder = OllamaEmbedder(config.embedding)
        self.store = VectorStore(config.vectorstore, embedder)
        self.rc: RetrievalConfig = config.retrieval

    # ------------------------------------------------------------------
    # Book-scoped queries
    # ------------------------------------------------------------------

    def get_book_chapters(self, book_id: str) -> list[str]:
        return self.store.get_chapters(book_id)

    def get_chapter_chunks(self, book_id: str, chapter: str) -> list[dict]:
        return self.store.get_chunks_by_chapter(book_id, chapter)

    def search_book(self, book_id: str, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self.rc.top_k
        results = self.store.query(query, top_k=k, where={"book_id": book_id})
        return self._filter_threshold(results)

    # ------------------------------------------------------------------
    # Cross-book queries
    # ------------------------------------------------------------------

    def search_all(self, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self.rc.top_k
        results = self.store.query(query, top_k=k)
        return self._filter_threshold(results)

    def search_books(self, query: str, book_ids: list[str], top_k: int | None = None) -> list[dict]:
        """Search across a specific set of books."""
        # TODO v2: ChromaDB $in filter when supported; for now, union queries
        all_results = []
        k = top_k or self.rc.top_k
        for bid in book_ids:
            all_results.extend(self.search_book(bid, query, top_k=k))
        all_results.sort(key=lambda r: r.get("distance", 999))
        return all_results[: self.rc.max_context_chunks]

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format_chunks_for_prompt(chunks: list[dict], include_meta: bool = True) -> str:
        """Format retrieved chunks into text suitable for an LLM prompt."""
        parts = []
        for c in chunks:
            header = ""
            if include_meta:
                header = (
                    f"[{c.get('title', '?')} | {c.get('chapter', '?')} | "
                    f"p.{c.get('page_range', '?')} | chunk {c.get('id', '?')}]\n"
                )
            parts.append(header + c.get("text", ""))
        return "\n---\n".join(parts)

    @staticmethod
    def format_citation(chunk: dict) -> str:
        return (
            f"({chunk.get('title', '?')}, {chunk.get('chapter', '?')}, "
            f"p.{chunk.get('page_range', '?')})"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _filter_threshold(self, results: list[dict]) -> list[dict]:
        if self.rc.similarity_threshold <= 0:
            return results
        return [r for r in results if r.get("distance", 999) <= self.rc.similarity_threshold]
