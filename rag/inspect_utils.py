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
            types: dict[str, int] = {}
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
            idx=i, label=label, stype=s.get("section_type", "unknown"),
            start=start, end=end, count=end - start + 1, chunks=n_chunks,
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


def inspect_windows(book_id: str, config: RAGConfig, section: str | None = None) -> None:
    """Print summary window info from the last summarization run."""
    results_dir = Path(config.storage.results_directory) / book_id
    ws_path = results_dir / "window_summaries.json"
    meta_path = results_dir / "summary_meta.json"

    if not ws_path.exists():
        print(f"No window summaries found for '{book_id}'. Run summarize first.")
        return

    window_data = json.loads(ws_path.read_text())
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    meta_sections = {s["name"]: s for s in meta.get("sections", [])}

    sections_to_show = [section] if section else list(window_data.keys())

    for sec_name in sections_to_show:
        if sec_name not in window_data:
            print(f"  Section '{sec_name}' not found in window data.")
            continue

        windows = window_data[sec_name]
        sec_meta = meta_sections.get(sec_name, {})
        total_w = sec_meta.get("windows_total", "?")
        selected_w = sec_meta.get("windows_selected", len(windows))
        selected_indices = sec_meta.get("selected_window_indices", [])
        selected_labels = sec_meta.get("selected_labels", [])

        print(f"\n  {sec_name}")
        print(f"  {'─' * 60}")
        print(f"  Total windows: {total_w}  |  Selected: {selected_w}  |  Indices: {selected_indices}")
        print()
        for j, ws in enumerate(windows):
            wi = ws.get("window", "?")
            score = ws.get("score", 0)
            labels = ws.get("labels", [])
            cached = "CACHED" if ws.get("cached") else "COMPUTED"
            n_chunks = len(ws.get("chunk_ids", []))
            summary_preview = ws.get("summary", "")[:200].replace("\n", " ")
            print(f"    Window {wi+1 if isinstance(wi, int) else wi}  "
                  f"score={score:.3f}  [{'+'.join(labels) or 'general'}]  "
                  f"{cached}  ({n_chunks} chunks)")
            print(f"      {summary_preview}{'…' if len(ws.get('summary', '')) > 200 else ''}")
            print()


def inspect_selection(book_id: str, config: RAGConfig, section: str | None = None) -> None:
    """Print detailed window selection info: all candidates, scores, reasons."""
    results_dir = Path(config.storage.results_directory) / book_id
    sel_path = results_dir / "selection_detail.json"

    if not sel_path.exists():
        print(f"No selection detail for '{book_id}'. Run summarize first.")
        return

    sel_data = json.loads(sel_path.read_text())
    sections_to_show = [section] if section else list(sel_data.keys())

    for sec_name in sections_to_show:
        if sec_name not in sel_data:
            print(f"  Section '{sec_name}' not found in selection data.")
            continue

        windows = sel_data[sec_name]
        n_selected = sum(1 for w in windows if w.get("selected"))
        print(f"\n  {sec_name}  ({n_selected}/{len(windows)} selected)")
        print(f"  {'─' * 80}")
        print(f"  {'W#':>3s}  {'Sel':>3s}  {'Composite':>9s}  {'Content':>7s}  "
              f"{'Density':>7s}  {'Specific':>8s}  {'TitleOv':>7s}  {'Pos':>5s}  "
              f"{'Labels':<20s}  Reason")
        print(f"  {'─' * 80}")

        for w in windows:
            idx = w.get("index", "?")
            sel = "YES" if w.get("selected") else " - "
            scores = w.get("scores", {})
            labels = "+".join(w.get("content_labels", [])) or "general"
            reason = w.get("reason", "")
            print(f"  {idx+1 if isinstance(idx, int) else idx:>3}  {sel:>3s}  "
                  f"{scores.get('composite', 0):>9.4f}  "
                  f"{scores.get('content_type', 0):>7.3f}  "
                  f"{scores.get('concept_density', 0):>7.3f}  "
                  f"{scores.get('specificity', 0):>8.3f}  "
                  f"{scores.get('title_overlap', 0):>7.3f}  "
                  f"{scores.get('position', 0):>5.2f}  "
                  f"{labels:<20s}  {reason}")
        print()


def inspect_summary_meta(book_id: str, config: RAGConfig) -> None:
    """Print summary metadata from the last summarization run."""
    results_dir = Path(config.storage.results_directory) / book_id
    meta_path = results_dir / "summary_meta.json"
    if not meta_path.exists():
        print(f"No summary metadata for '{book_id}'. Run summarize first.")
        return

    meta = json.loads(meta_path.read_text())
    print(f"\nSummary metadata for: {book_id}")
    print(f"{'=' * 60}")
    print(f"  Quality:       {meta.get('quality', '?')}")
    print(f"  Mode:          {meta.get('mode', '?')}")
    print(f"  Budget:        {meta.get('budget_per_section', '?')} windows/section")
    print(f"  Strategy:      {meta.get('strategy', '?')}")
    print(f"  MMR lambda:    {meta.get('mmr_lambda', '?')}")
    print(f"  Model:         {meta.get('model', '?')}")
    print(f"  Timestamp:     {meta.get('timestamp', '?')}")
    print(f"  LLM calls:     {meta.get('total_llm_calls', '?')}")
    print(f"  Cache hits:    {meta.get('cache_hits', '?')}")
    print(f"  Cache misses:  {meta.get('cache_misses', '?')}")

    sections = meta.get("sections", [])
    if sections:
        print(f"\n  Per-section breakdown:")
        total_selected = 0
        total_total = 0
        for s in sections:
            sel = s.get("windows_selected", "?")
            tot = s.get("windows_total", "?")
            if isinstance(sel, int):
                total_selected += sel
            if isinstance(tot, int):
                total_total += tot
            indices = s.get("selected_window_indices", [])
            labels = s.get("selected_labels", [])
            labels_str = ", ".join(
                "+".join(l) if l else "gen" for l in labels
            ) if labels else ""
            print(f"    {s.get('name', '?'):<40s}  "
                  f"{sel}/{tot} windows  chunks={s.get('chunks', '?')}")
            if labels_str:
                print(f"      labels: {labels_str}")
        if isinstance(total_selected, int) and isinstance(total_total, int):
            print(f"\n  Totals: {total_selected}/{total_total} windows summarized "
                  f"across {len(sections)} sections")
    print()


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
