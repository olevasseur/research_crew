"""Inspection and debugging utilities for the RAG pipeline."""

from __future__ import annotations

import json
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
        sections = info.get("sections", [])
        if sections:
            types = {}
            for s in sections:
                t = s.get("section_type", "?")
                types[t] = types.get(t, 0) + 1
            print(f"    Sections: {len(sections)} ({', '.join(f'{v} {k}' for k, v in types.items())})")
        else:
            print(f"    Chapters: {', '.join(info.get('chapters', []))}")
        print(f"    Source:   {info.get('source_path', '?')}")
        print(f"    Ingested: {info.get('ingested_at', '?')}")


def inspect_chunks(book_id: str, config: RAGConfig, chapter: str | None = None) -> None:
    """Print chunks for a book, optionally filtered by section name."""
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

    # Summary table first
    current_section = None
    for c in chunks:
        sec = c.get("chapter", c.get("section_label", "?"))
        if sec != current_section:
            current_section = sec
            sec_chunks = [x for x in chunks if (x.get("chapter") or x.get("section_label")) == sec]
            print(f"\n  {sec}  ({len(sec_chunks)} chunks)")
        stype = c.get("section_type", "?")
        print(f"    {c.get('id', '?'):>30s}  "
              f"p.{c.get('page_range', '?'):>7s}  "
              f"type={stype:<14s}  "
              f"len={len(c.get('text', '')):>5d} chars")

    if chapter:
        print(f"\n{'=' * 60}")
        print(f"Full content for section: {chapter}")
        print(f"{'=' * 60}")
        for c in chunks:
            print(f"\n--- {c.get('id', '?')} (p.{c.get('page_range', '?')}) ---")
            text = c.get("text", "")
            if len(text) > 800:
                print(text[:800] + f"\n... ({len(text) - 800} more chars)")
            else:
                print(text)


def inspect_structure(book_id: str, config: RAGConfig) -> None:
    """Print the detected section structure for an ingested book."""
    embedder = OllamaEmbedder(config.embedding)
    store = VectorStore(config.vectorstore, embedder)
    info = store.get_book_info(book_id)
    if not info:
        print(f"Book '{book_id}' not found. Ingest it first.")
        return
    sections = info.get("sections", [])
    if not sections:
        print(f"No section metadata stored for '{book_id}'. Re-ingest to populate.")
        return

    # Also get chunk counts per section
    all_chunks = store.get_chunks_by_book(book_id)
    chunk_counts: dict[str, int] = {}
    for c in all_chunks:
        sec = c.get("chapter", "")
        chunk_counts[sec] = chunk_counts.get(sec, 0) + 1

    title = info.get("title", "?")
    author = info.get("author", "?")
    print(f"\nStructure of \"{title}\" by {author}")
    print(f"{'=' * 80}")
    fmt = "{idx:>3}  {label:<40s}  {stype:<16s}  p.{start:>3d}-{end:<3d}  {count:>3d}pp  {chunks:>2d} chunks"
    header = f"{'#':>3}  {'Label':<40s}  {'Type':<16s}  {'Pages':>11s}  {'Span':>5s}  {'Chunks':>8s}"
    print(header)
    print("-" * 80)
    for i, s in enumerate(sections, 1):
        parent = s.get("parent", "")
        label = s.get("name", "?")
        if parent:
            label = f"  └─ {label}"
        start = s.get("start_page", 0)
        end = s.get("end_page", 0)
        n_chunks = chunk_counts.get(s.get("name", ""), 0)
        print(fmt.format(
            idx=i,
            label=label,
            stype=s.get("section_type", "unknown"),
            start=start,
            end=end,
            count=end - start + 1,
            chunks=n_chunks,
        ))
        conf = s.get("confidence", 0)
        reason = s.get("detection_reason", "")
        if reason:
            print(f"     {'':40s}  conf={conf:.2f}  {reason}")
    print()


def inspect_subchunks(book_id: str, section: str, config: RAGConfig) -> None:
    """Print detailed subchunk info for a single section."""
    retrieval = Retrieval(config)
    chunks = retrieval.get_chapter_chunks(book_id, section)
    if not chunks:
        print(f"No chunks found for section '{section}' in '{book_id}'")
        return
    chunks.sort(key=lambda c: c.get("chunk_index", 0))
    print(f"\nSubchunks for: {section}")
    print(f"{'=' * 60}")
    for c in chunks:
        print(f"\n  ID:          {c.get('id', '?')}")
        print(f"  Pages:       {c.get('page_range', '?')}")
        print(f"  Section idx: {c.get('section_chunk_index', '?')}")
        print(f"  Type:        {c.get('section_type', '?')}")
        print(f"  Parent:      {c.get('parent_part', c.get('parent_section_id', '?'))}")
        print(f"  Length:      {len(c.get('text', ''))} chars")
        text = c.get("text", "")
        preview = text[:300].replace("\n", " ")
        print(f"  Preview:     {preview}{'…' if len(text) > 300 else ''}")
    print(f"\n  Total: {len(chunks)} subchunks")


def inspect_summary(book_id: str, config: RAGConfig) -> None:
    """Print a book's summary and related files if they exist."""
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

    # Show subchunk summary counts if available
    sc_path = results_dir / "subchunk_summaries.json"
    if sc_path.exists():
        data = json.loads(sc_path.read_text())
        print(f"\n{'=' * 60}")
        print(f"  Subchunk summaries")
        print(f"{'=' * 60}")
        for section, summaries in data.items():
            print(f"  {section}: {len(summaries)} subchunk summaries")


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
        print(f"Section: {r.get('chapter', r.get('section_label', '?'))}")
        print(f"Type:    {r.get('section_type', '?')}")
        print(f"Pages:   {r.get('page_range', '?')}")
        print(f"ID:      {r.get('id', '?')}")
        text = r.get("text", "")
        if len(text) > 500:
            print(text[:500] + f"\n... ({len(text) - 500} more chars)")
        else:
            print(text)
