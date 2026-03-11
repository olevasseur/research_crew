"""ChromaDB vector store wrapper with a JSON book registry."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import chromadb

from .chunker import Chunk
from .config import VectorStoreConfig
from .embeddings import ChromaOllamaEmbedding, OllamaEmbedder


class VectorStore:
    """Thin wrapper around ChromaDB.

    Book-level metadata is stored in a separate ``books.json`` file so it's
    easy to inspect without touching the vector DB.
    """

    def __init__(self, vs_config: VectorStoreConfig, embedder: OllamaEmbedder):
        persist_dir = Path(vs_config.persist_directory)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._ef = ChromaOllamaEmbedding(embedder)
        self._collection = self._client.get_or_create_collection(
            name=vs_config.collection_name,
            embedding_function=self._ef,
        )
        self._registry_path = persist_dir / "books.json"
        self._registry: dict = self._load_registry()

    # ------------------------------------------------------------------
    # Book registry (plain JSON — inspectable)
    # ------------------------------------------------------------------

    def register_book(self, book_id: str, meta: dict) -> None:
        meta.setdefault("ingested_at", datetime.now(timezone.utc).isoformat())
        self._registry[book_id] = meta
        self._save_registry()

    def get_book_info(self, book_id: str) -> dict | None:
        return self._registry.get(book_id)

    def list_books(self) -> dict:
        return dict(self._registry)

    def delete_book(self, book_id: str) -> None:
        ids = self._collection.get(where={"book_id": book_id})["ids"]
        if ids:
            self._collection.delete(ids=ids)
        self._registry.pop(book_id, None)
        self._save_registry()

    # ------------------------------------------------------------------
    # Chunk CRUD
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Upsert chunks into ChromaDB."""
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[_chunk_meta(c) for c in chunks],
        )

    def get_chunks_by_book(self, book_id: str) -> list[dict]:
        results = self._collection.get(where={"book_id": book_id}, include=["documents", "metadatas"])
        return _zip_results(results)

    def get_chunks_by_chapter(self, book_id: str, chapter: str) -> list[dict]:
        results = self._collection.get(
            where={"$and": [{"book_id": book_id}, {"chapter": chapter}]},
            include=["documents", "metadatas"],
        )
        return sorted(_zip_results(results), key=lambda c: c["chunk_index"])

    def get_chapters(self, book_id: str) -> list[str]:
        info = self.get_book_info(book_id)
        if info and "chapters" in info:
            return info["chapters"]
        chunks = self.get_chunks_by_book(book_id)
        seen: dict[str, int] = {}
        for c in chunks:
            ch = c.get("chapter", "")
            if ch and ch not in seen:
                seen[ch] = c.get("chunk_index", 0)
        return sorted(seen, key=lambda k: seen[k])

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        """Semantic search. ChromaDB handles embedding the query via our embedding function."""
        kwargs: dict = {
            "query_texts": [query_text],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        results = self._collection.query(**kwargs)
        return _zip_query_results(results)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_registry(self) -> dict:
        if self._registry_path.exists():
            return json.loads(self._registry_path.read_text())
        return {}

    def _save_registry(self) -> None:
        self._registry_path.write_text(json.dumps(self._registry, indent=2))


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _chunk_meta(c: Chunk) -> dict:
    return {
        "book_id": c.book_id,
        "title": c.title,
        "author": c.author,
        "chapter": c.chapter,
        "section": c.section,
        "source_path": c.source_path,
        "chunk_index": c.chunk_index,
        "parent_section_id": c.parent_section_id,
        "page_range": c.page_range,
        "section_type": c.section_type,
    }


def _zip_results(raw: dict) -> list[dict]:
    out = []
    ids = raw.get("ids", [])
    docs = raw.get("documents", [])
    metas = raw.get("metadatas", [])
    for i, cid in enumerate(ids):
        entry = dict(metas[i]) if i < len(metas) else {}
        entry["id"] = cid
        entry["text"] = docs[i] if i < len(docs) else ""
        out.append(entry)
    return out


def _zip_query_results(raw: dict) -> list[dict]:
    out = []
    for i, cid in enumerate(raw["ids"][0]):
        entry = dict(raw["metadatas"][0][i])
        entry["id"] = cid
        entry["text"] = raw["documents"][0][i]
        entry["distance"] = raw["distances"][0][i]
        out.append(entry)
    return out
