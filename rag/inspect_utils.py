"""Inspection and debugging utilities for the RAG pipeline."""

from __future__ import annotations

from pathlib import Path

from .config import RAGConfig
from .embeddings import OllamaEmbedder
from .retrieval import Retrieval
from .store import VectorStore


def inspect_books(config: RAGConfig) -> None:
    """Print all ingested books."""
    embedder = OllamaEmbedder(config.embedding)
    store = VectorStore(config.vectorstore, embedder)
    books = store.list_books()
    if not books:
        print("No books ingested yet.")
        return
    for bid, info in books.items():
        print(f"\n  {bid}")
        print(f"    Title:    {info.get('title', '?')}")
        print(f"    Author:   {info.get('author', '?')}")
        print(f"    Pages:    {info.get('total_pages', '?')}")
        print(f"    Chunks:   {info.get('total_chunks', '?')}")
        print(f"    Chapters: {', '.join(info.get('chapters', []))}")
        print(f"    Source:   {info.get('source_path', '?')}")
        print(f"    Ingested: {info.get('ingested_at', '?')}")


def inspect_chunks(book_id: str, config: RAGConfig, chapter: str | None = None) -> None:
    """Print chunks for a book, optionally filtered by chapter."""
    retrieval = Retrieval(config)
    if chapter:
        chunks = retrieval.get_chapter_chunks(book_id, chapter)
    else:
        chunks = retrieval.store.get_chunks_by_book(book_id)

    if not chunks:
        print(f"No chunks found for book_id='{book_id}'" +
              (f", chapter='{chapter}'" if chapter else ""))
        return

    chunks.sort(key=lambda c: c.get("chunk_index", 0))
    for c in chunks:
        print(f"\n{'=' * 60}")
        print(f"ID:      {c.get('id', '?')}")
        print(f"Chapter: {c.get('chapter', '?')}")
        print(f"Pages:   {c.get('page_range', '?')}")
        print(f"Index:   {c.get('chunk_index', '?')}")
        print(f"{'- ' * 30}")
        text = c.get("text", "")
        if len(text) > 500:
            print(text[:500] + "\n... (truncated)")
        else:
            print(text)


def inspect_summary(book_id: str, config: RAGConfig) -> None:
    """Print a book's summary if it exists."""
    results_dir = Path(config.storage.results_directory) / book_id
    for name in ["book_summary.md", "chapter_insights.md", "verification.md"]:
        fpath = results_dir / name
        if fpath.exists():
            print(f"\n{'=' * 60}")
            print(f"  {name}")
            print(f"{'=' * 60}")
            print(fpath.read_text())
        else:
            print(f"  {name}: not yet generated")


def inspect_retrieval(query: str, config: RAGConfig, book_id: str | None = None) -> None:
    """Run a retrieval query and print the results with metadata."""
    retrieval = Retrieval(config)
    if book_id:
        results = retrieval.search_book(book_id, query)
    else:
        results = retrieval.search_all(query)

    if not results:
        print("No results found.")
        return

    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (distance: {r.get('distance', '?'):.4f}) ---")
        print(f"Book:    {r.get('title', '?')} ({r.get('book_id', '?')})")
        print(f"Chapter: {r.get('chapter', '?')}")
        print(f"Pages:   {r.get('page_range', '?')}")
        print(f"ID:      {r.get('id', '?')}")
        text = r.get("text", "")
        if len(text) > 400:
            print(text[:400] + "\n... (truncated)")
        else:
            print(text)
