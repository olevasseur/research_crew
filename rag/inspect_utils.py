"""Inspection and debugging utilities for the RAG pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .config import RAGConfig


def _new_retrieval(config: RAGConfig):
    from .retrieval import Retrieval

    return Retrieval(config)


def _new_vector_store(config: RAGConfig):
    from .embeddings import OllamaEmbedder
    from .store import VectorStore

    embedder = OllamaEmbedder(config.embedding)
    return VectorStore(config.vectorstore, embedder)


def inspect_books(config: RAGConfig) -> None:
    """Print all ingested books."""
    store = _new_vector_store(config)
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
    retrieval = _new_retrieval(config)
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
    store = _new_vector_store(config)
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
    retrieval = _new_retrieval(config)
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


def _find_leading_overlap(prev_text: str, curr_text: str) -> int:
    """Return the number of leading chars in curr_text that repeat the tail of prev_text.

    Scans suffixes of prev_text (up to 400 chars back) looking for one that is
    a prefix of curr_text.  Returns 0 if no overlap >= 20 chars is found.
    """
    if not prev_text or not curr_text:
        return 0
    search_start = max(0, len(prev_text) - 400)
    for start in range(search_start, len(prev_text) - 20):
        suffix = prev_text[start:]
        if curr_text.startswith(suffix):
            return len(suffix)
    return 0


def inspect_window(
    book_id: str,
    window_id: int,
    config: RAGConfig,
    section: str | None = None,
) -> None:
    """Show the full content of one window: summary + original chunks + context.

    window_id is 1-based (as displayed to the user).  Internally windows are
    stored 0-based in window_summaries.json under the ``window`` key.
    """
    import re

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.lower().strip())

    results_dir = Path(config.storage.results_directory) / book_id
    ws_path = results_dir / "window_summaries.json"
    meta_path = results_dir / "summary_meta.json"
    sel_path = results_dir / "selection_detail.json"

    if not ws_path.exists():
        print(f"No window summaries found for '{book_id}'. Run summarize first.")
        return

    window_data = json.loads(ws_path.read_text())
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    sel_data = json.loads(sel_path.read_text()) if sel_path.exists() else {}

    target_idx = window_id - 1  # stored 0-based

    # --- Narrow to one section if requested ---
    if section:
        norm = _norm(section)
        matched = [s for s in window_data if norm in _norm(s)]
        if not matched:
            print(f"Section '{section}' not found in summaries for '{book_id}'.")
            print("\nAvailable sections:")
            for s in window_data:
                print(f"  - {s}")
            return
        if len(matched) > 1:
            print(f"Ambiguous --section '{section}' matches multiple sections:")
            for s in matched:
                print(f"  - {s}")
            print("Use a more specific --section value.")
            return
        sections_to_search: dict = {matched[0]: window_data[matched[0]]}
    else:
        sections_to_search = window_data

    # --- Find the window ---
    candidates: list[tuple[str, dict]] = []
    for sec_name, windows in sections_to_search.items():
        for ws in windows:
            if ws.get("window") == target_idx:
                candidates.append((sec_name, ws))

    is_summarised = True
    if not candidates:
        # Fallback: look in selection_detail.json for non-selected windows
        if section:
            sel_sections = {s: sel_data[s] for s in sel_data if norm in _norm(s)}
        else:
            sel_sections = sel_data
        for sec_name, sel_windows in sel_sections.items():
            for sel_w in sel_windows:
                if sel_w.get("index") == target_idx:
                    candidates.append((sec_name, {
                        "window": target_idx,
                        "chunk_ids": sel_w.get("chunk_ids", []),
                        "score": sel_w.get("scores", {}).get("composite", 0),
                        "labels": sel_w.get("content_labels", []),
                        "summary": "",
                    }))
        is_summarised = False

    if not candidates:
        print(f"Window {window_id} not found in '{book_id}'.")
        print("\nAvailable window indices per section:")
        for sec_name, windows in sections_to_search.items():
            ids = [
                str(ws["window"] + 1) if isinstance(ws.get("window"), int) else str(ws.get("window", "?"))
                for ws in windows
            ]
            print(f"  {sec_name}: {', '.join(ids)}")
        return

    if len(candidates) > 1:
        print(f"Window {window_id} appears in multiple sections:")
        for sec, _ in candidates:
            print(f"  - {sec}")
        print("Use --section to disambiguate.")
        return

    found_sec, found_ws = candidates[0]

    # --- Selection detail entry for this window ---
    sel_entry = next(
        (d for d in sel_data.get(found_sec, []) if d.get("index") == target_idx),
        None,
    )

    # --- Load chunk text from vectorstore ---
    retrieval = _new_retrieval(config)
    chunk_ids: list[str] = found_ws.get("chunk_ids", [])
    chapter_chunks = retrieval.get_chapter_chunks(book_id, found_sec)
    chunk_map = {c["id"]: c for c in chapter_chunks}

    # --- Page range: span across all chunks in the window ---
    starts: list[int] = []
    ends: list[int] = []
    for cid in chunk_ids:
        c = chunk_map.get(cid)
        if c:
            if c.get("page_start") is not None:
                starts.append(c["page_start"])
            if c.get("page_end") is not None:
                ends.append(c["page_end"])
    if starts and ends:
        pages = f"{min(starts)}\u2013{max(ends)}"
    else:
        ranges = [chunk_map[cid].get("page_range", "") for cid in chunk_ids if cid in chunk_map]
        pages = ", ".join(r for r in ranges if r) or "?"

    wi_display = target_idx + 1
    labels = found_ws.get("labels", [])
    labels_str = " + ".join(labels) if labels else "general"
    score = found_ws.get("score", 0)

    sec_meta_map = {s["name"]: s for s in meta.get("sections", [])}
    total_windows = sec_meta_map.get(found_sec, {}).get("windows_total", "?")

    selected = sel_entry.get("selected", True) if sel_entry else True
    sel_str = "yes" if selected else "no"
    rank_str = f"{wi_display} of {total_windows}"

    # --- Header ---
    print(f"\n{'=' * 70}")
    print(f"  Window {wi_display}  ({found_sec})")
    print(f"{'=' * 70}")
    print(f"  Pages: {pages}")
    print(f"  Tags:  {labels_str}")
    print(f"  Score: {score:.3f}")

    # --- Find previous window's last chunk for inter-window overlap dedup ---
    prev_last_text = ""
    if target_idx > 0:
        prev_idx = target_idx - 1
        prev_chunk_ids: list[str] = []
        for sec_ws in window_data.get(found_sec, []):
            if sec_ws.get("window") == prev_idx:
                prev_chunk_ids = sec_ws.get("chunk_ids", [])
                break
        if not prev_chunk_ids:
            for sel_w in sel_data.get(found_sec, []):
                if sel_w.get("index") == prev_idx:
                    prev_chunk_ids = sel_w.get("chunk_ids", [])
                    break
        if prev_chunk_ids:
            last_prev = chunk_map.get(prev_chunk_ids[-1])
            if last_prev:
                prev_last_text = last_prev.get("text", "")

    # --- Summary ---
    print(f"\n{'─' * 70}")
    if is_summarised and found_ws.get("summary", "").strip():
        print(f"\n  [Summary]")
        for line in found_ws["summary"].strip().split("\n"):
            print(f"  {line}")
    else:
        print(f"\n  [Full text — not summarised]")

    # --- Full text ---
    print(f"\n{'─' * 70}")
    print(f"\n  [Full Text]")
    prev_display_text = prev_last_text
    for i, cid in enumerate(chunk_ids):
        chunk = chunk_map.get(cid)
        if chunk:
            text = chunk.get("text", "")
            skip = _find_leading_overlap(prev_display_text, text) if prev_display_text else 0
            print(f"\n  --- {cid}  (p.{chunk.get('page_range', '?')}) ---\n")
            if i == 0 and skip > 0 and prev_last_text:
                print(f"  ↑ continued from window {target_idx}  ({skip} chars overlap skipped)\n")
            display_text = text[skip:].lstrip("\n") if skip else text
            for line in display_text.split("\n"):
                print(f"  {line}")
            prev_display_text = text
        else:
            print(f"\n  --- {cid}  (not found in vectorstore) ---")
            prev_display_text = ""

    # --- Context ---
    print(f"\n{'─' * 70}")
    print(f"\n  [Context]")
    print(f"  Section:  {found_sec}")
    print(f"  Selected: {sel_str}")
    print(f"  Rank:     {rank_str}")
    if sel_entry and sel_entry.get("reason"):
        print(f"  Reason:   {sel_entry['reason']}")
    print()


_IDEA_HEADING_MAP: dict[str, str] = {
    "key claims": "key_idea",
    "key claim": "key_idea",
    "evidence & examples": "example",
    "evidence and examples": "example",
    "frameworks/concepts": "framework",
    "frameworks / concepts": "framework",
    "frameworks": "framework",
    "concepts": "framework",
}


def extract_book_ideas(book_id: str, config: RAGConfig) -> list[dict]:
    """Extract structured idea items from existing window summaries.

    Parses ``**Heading**`` sections and numbered items from each window
    summary.  Returns a flat list of ``{type, text, section, window}``
    dicts.  Skips the ``actionable points`` category to keep the list
    focused on ideas, evidence, and frameworks.
    """
    results_dir = Path(config.storage.results_directory) / book_id
    ws_path = results_dir / "window_summaries.json"
    if not ws_path.exists():
        return []

    window_data: dict = json.loads(ws_path.read_text())
    ideas: list[dict] = []

    for section_name, windows in window_data.items():
        for ws in windows:
            summary = ws.get("summary", "")
            if not summary:
                continue
            win_num = ws.get("window", 0) + 1
            parts = re.split(r"\*\*([^*]+)\*\*", summary)
            for i in range(1, len(parts), 2):
                heading = parts[i].strip().rstrip(":").lower()
                idea_type = _IDEA_HEADING_MAP.get(heading)
                if not idea_type:
                    continue
                body = parts[i + 1] if i + 1 < len(parts) else ""
                for m in re.finditer(r"^\d+\.\s+(.+)", body, re.MULTILINE):
                    text = m.group(1).strip()
                    if text and len(text) > 10:
                        ideas.append({
                            "type": idea_type,
                            "text": text,
                            "section": section_name,
                            "window": win_num,
                        })
    return ideas


def score_book_ideas(
    ideas: list[dict],
    book_id: str,
    config: RAGConfig,
) -> None:
    """Attach ``rank_score`` to each idea in-place for deterministic ranking.

    Signals (generated ideas only — user-authored always get 1.0):
      - Window content score from summarization  (weight 0.40)
      - Text quality: length + specificity cues  (weight 0.35)
      - User-note bonus                          (+0.15)
      - Near-duplicate penalty within type        (−0.10)
    """
    results_dir = Path(config.storage.results_directory) / book_id
    ws_path = results_dir / "window_summaries.json"
    win_scores: dict[tuple[str, int], float] = {}
    if ws_path.exists():
        for sec, windows in json.loads(ws_path.read_text()).items():
            for w in windows:
                win_scores[(sec, w.get("window", 0) + 1)] = w.get("score", 0.5)

    for idea in ideas:
        if idea.get("source") == "user":
            idea["rank_score"] = 1.0
            continue

        s = 0.0

        # --- window content score (0.40) ---
        ws = win_scores.get(
            (idea.get("section", ""), idea.get("window", 0)), 0.5
        )
        s += 0.40 * ws

        # --- text quality (0.35) ---
        text = idea.get("text", "")
        tlen = len(text)
        if tlen < 30:
            lq = 0.2
        elif tlen < 60:
            lq = 0.5
        elif tlen <= 200:
            lq = 1.0
        else:
            lq = 0.85
        spec = 0.0
        if re.search(r"\[p\.\s*\d+\]", text):
            spec += 0.3
        if re.search(r"[A-Z][a-z]+\s+[A-Z][a-z]+", text):
            spec += 0.2
        if re.search(r"\d{2,}", text):
            spec += 0.1
        if '"' in text or "\u2018" in text or "\u201c" in text:
            spec += 0.1
        spec = min(spec, 0.5)
        tq = 0.6 * lq + 0.4 * (spec / 0.5 if spec else 0.0)
        s += 0.35 * tq

        # --- note bonus (0.15) ---
        if idea.get("note"):
            s += 0.15

        s += 0.05  # base
        idea["rank_score"] = round(s, 4)

    # --- near-duplicate suppression within each type ---
    by_type: dict[str, list[dict]] = {}
    for idea in ideas:
        if idea.get("source") == "user":
            continue
        by_type.setdefault(idea["type"], []).append(idea)

    def _content_tokens(text: str) -> set[str]:
        return {
            w for w in re.sub(r"[^\w\s]", " ", text.lower()).split()
            if w not in _STOP and len(w) > 2
        }

    for group in by_type.values():
        group.sort(key=lambda x: x.get("rank_score", 0), reverse=True)
        seen: list[set[str]] = []
        for idea in group:
            words = _content_tokens(idea.get("text", ""))
            if len(words) < 4:
                continue
            for prev_words in seen:
                overlap = len(words & prev_words) / max(len(words | prev_words), 1)
                if overlap >= 0.50:
                    idea["rank_score"] = round(
                        idea.get("rank_score", 0) - 0.10, 4
                    )
                    idea["near_dup"] = True
                    break
            seen.append(words)


# ── Stopwords for query relevance (kept minimal) ────────────────────────
_STOP = frozenset(
    "a an the is are was were be been being have has had do does did will "
    "would shall should may might can could of in to for on with at by from "
    "and or but not no nor so yet this that these those it its i me my we "
    "our you your he she they them their what which who whom how about into "
    "than too very also just as".split()
)


def rank_ideas_for_query(
    ideas: list[dict],
    query: str,
    *,
    top_k: int = 10,
    include_hidden: bool = False,
) -> list[dict]:
    """Return the *top_k* ideas most relevant to *query*.

    Scoring combines token overlap with the existing ``rank_score``
    (from :func:`score_book_ideas`) as a quality prior.

    Signals:
      - Token overlap between query and idea text      (weight 0.55)
      - Token overlap between query and section name   (weight 0.15)
      - Existing rank_score quality prior               (weight 0.20)
      - User-authored / has-note bonus                  (weight 0.10)
      - Pinned curation boost                           (additive 0.25)
      - Hidden curation penalty                         (additive -0.5 when
        ``include_hidden=True``; hidden ideas are excluded entirely by default)
      - Near-duplicate penalty                          (additive -0.15 so the
        canonical idea in a near-duplicate pair wins the retrieval slot)

    Returns a new list sorted by combined relevance, trimmed to *top_k*.
    Each returned idea gets a ``query_score`` field attached.
    """
    # Default policy: hidden ideas are excluded from the candidate pool.
    # When include_hidden=True (explicit opt-in), hidden ideas stay but get a
    # strong additive penalty so only overwhelmingly relevant ones survive.
    if not include_hidden:
        ideas = [i for i in ideas if not i.get("hidden")]

    query_tokens = {
        w for w in re.sub(r"[^\w\s]", " ", query.lower()).split()
        if w not in _STOP and len(w) > 1
    }
    if not query_tokens:
        # No meaningful tokens — fall back to rank_score order (pinned still wins ties)
        ranked = sorted(
            ideas,
            key=lambda x: (bool(x.get("pinned")), x.get("rank_score", 0)),
            reverse=True,
        )
        for it in ranked:
            it["query_score"] = it.get("rank_score", 0) + (0.25 if it.get("pinned") else 0.0)
        return ranked[:top_k]

    scored: list[tuple[float, dict]] = []
    for idea in ideas:
        # Include note text in matching so annotated ideas surface better
        idea_text = idea.get("text", "")
        note_text = idea.get("note", "")
        full_text = (idea_text + " " + note_text) if note_text else idea_text
        idea_tokens = {
            w for w in re.sub(r"[^\w\s]", " ", full_text.lower()).split()
            if w not in _STOP and len(w) > 1
        }
        sec_tokens = {
            w for w in re.sub(r"[^\w\s]", " ", idea.get("section", "").lower()).split()
            if w not in _STOP and len(w) > 1
        }

        # Text overlap (0.55)
        if idea_tokens:
            text_overlap = len(query_tokens & idea_tokens) / len(query_tokens)
        else:
            text_overlap = 0.0

        # Section overlap (0.15)
        if sec_tokens:
            sec_overlap = len(query_tokens & sec_tokens) / len(query_tokens)
        else:
            sec_overlap = 0.0

        # Rank score prior (0.20)
        rank_prior = idea.get("rank_score", 0.3)

        # User/note bonus (0.10)
        bonus = 0.0
        if idea.get("source") == "user":
            bonus = 1.0
        elif idea.get("note"):
            bonus = 0.7

        # Pinned boost (additive; pinned items rise meaningfully without
        # overwhelming strong query matches).
        pinned_boost = 0.25 if idea.get("pinned") else 0.0

        # Hidden penalty — only reachable when include_hidden=True (hidden
        # ideas are pre-filtered otherwise). Strong enough that a hidden idea
        # needs near-perfect query match to beat a pinned or high-relevance one.
        hidden_penalty = -0.5 if idea.get("hidden") else 0.0

        # Near-duplicate penalty — honor either signal:
        #  * 'near_duplicate' from curation enrichment (strict shingle match),
        #  * 'near_dup' from score_book_ideas (looser token-overlap match that
        #    catches paraphrased restatements of the same idea).
        # Whichever fires, we penalize the weaker twin so the canonical idea
        # wins its retrieval slot and duplicates don't crowd selected evidence.
        is_near_dup = bool(idea.get("near_duplicate") or idea.get("near_dup"))
        near_dup_penalty = -0.15 if is_near_dup else 0.0

        combined = (
            0.55 * text_overlap
            + 0.15 * sec_overlap
            + 0.20 * rank_prior
            + 0.10 * bonus
            + pinned_boost
            + hidden_penalty
            + near_dup_penalty
        )
        idea["query_score"] = round(combined, 4)
        scored.append((combined, idea))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Greedy dedup in two passes:
    # 1. Skip items already flagged as near_duplicate/near_dup by the
    #    upstream curation/scoring signals (their canonical twin wins).
    # 2. Additionally skip items whose normalized-token Jaccard against an
    #    already-selected item exceeds 0.60 — this catches cross-type
    #    restatements (e.g. the same definition surfaced as both a
    #    'key_idea' and a 'framework') that per-type near_dup missed.
    # Any item skipped by either check is pushed to a backfill queue and
    #  only used if we still have room after non-duplicate candidates.
    def _tokset(text: str) -> set[str]:
        return {
            w for w in re.sub(r"[^\w\s]", " ", (text or "").lower()).split()
            if w not in _STOP and len(w) > 2
        }

    def _jac(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    selected: list[dict] = []
    selected_tokens: list[set[str]] = []
    dup_backfill: list[dict] = []
    for _, idea in scored:
        if len(selected) >= top_k:
            break
        if idea.get("near_duplicate") or idea.get("near_dup"):
            dup_backfill.append(idea)
            continue
        toks = _tokset(idea.get("text", ""))
        if any(_jac(toks, prev) >= 0.60 for prev in selected_tokens):
            dup_backfill.append(idea)
            continue
        selected.append(idea)
        selected_tokens.append(toks)
    for idea in dup_backfill:
        if len(selected) >= top_k:
            break
        selected.append(idea)
    return selected


def rank_passages_for_query(
    passages: list[dict],
    query: str,
    *,
    top_k: int = 3,
) -> list[dict]:
    """Return the *top_k* saved passages most relevant to *query*.

    Each passage has ``preview`` (text) and ``section``.  Scoring uses
    token overlap with the preview and section name.
    """
    if not passages:
        return []
    query_tokens = {
        w for w in re.sub(r"[^\w\s]", " ", query.lower()).split()
        if w not in _STOP and len(w) > 1
    }
    if not query_tokens:
        return passages[:top_k]

    scored: list[tuple[float, dict]] = []
    for p in passages:
        preview_tokens = {
            w for w in re.sub(r"[^\w\s]", " ", p.get("preview", "").lower()).split()
            if w not in _STOP and len(w) > 1
        }
        sec_tokens = {
            w for w in re.sub(r"[^\w\s]", " ", p.get("section", "").lower()).split()
            if w not in _STOP and len(w) > 1
        }
        text_ov = len(query_tokens & preview_tokens) / len(query_tokens) if preview_tokens else 0.0
        sec_ov = len(query_tokens & sec_tokens) / len(query_tokens) if sec_tokens else 0.0
        combined = 0.70 * text_ov + 0.30 * sec_ov
        p["query_score"] = round(combined, 4)
        scored.append((combined, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k]]


def read_section_paragraphs(
    book_id: str, section: str, config: RAGConfig
) -> dict:
    """Return the full deduplicated text of a section as a list of paragraphs.

    Each paragraph is a non-empty line from the concatenated, overlap-stripped
    chunk text.  Returns ``{"section", "pages", "paragraphs": [str, ...]}``.
    """
    retrieval = _new_retrieval(config)
    chunks = retrieval.get_chapter_chunks(book_id, section)
    if not chunks:
        return {"section": section, "pages": "", "paragraphs": []}

    chunks.sort(key=lambda c: c.get("chunk_index", 0))

    starts = [c["page_start"] for c in chunks if c.get("page_start") is not None]
    ends = [c["page_end"] for c in chunks if c.get("page_end") is not None]
    pages = f"pp. {min(starts)}\u2013{max(ends)}" if starts and ends else ""

    parts: list[str] = []
    prev_text = ""
    for chunk in chunks:
        text = chunk.get("text", "")
        if not text.strip():
            continue
        skip = _find_leading_overlap(prev_text, text) if prev_text else 0
        display = text[skip:].lstrip("\n") if skip else text
        if display.strip():
            parts.append(display)
        prev_text = text

    full = "\n".join(parts)
    paragraphs = [p.strip() for p in full.split("\n") if p.strip()]
    return {"section": section, "pages": pages, "paragraphs": paragraphs}


def read_section(book_id: str, section: str, config: RAGConfig) -> None:
    """Print the full continuous text of a section with overlap deduplicated.

    Retrieves all chunks for the section, sorts by chunk_index, strips the
    200-char overlap between consecutive chunks, and prints the result as
    one continuous reading passage.
    """
    retrieval = _new_retrieval(config)
    chunks = retrieval.get_chapter_chunks(book_id, section)
    if not chunks:
        print(f"No text found for section '{section}' in '{book_id}'.")
        return

    chunks.sort(key=lambda c: c.get("chunk_index", 0))

    # --- Page range across all chunks ---
    starts = [c["page_start"] for c in chunks if c.get("page_start") is not None]
    ends = [c["page_end"] for c in chunks if c.get("page_end") is not None]
    pages = f"pp. {min(starts)}\u2013{max(ends)}" if starts and ends else ""

    print(f"\n{'=' * 70}")
    print(f"  {section}")
    if pages:
        print(f"  {pages}  ·  {len(chunks)} chunks")
    else:
        print(f"  {len(chunks)} chunks")
    print(f"{'=' * 70}\n")

    prev_text = ""
    for chunk in chunks:
        text = chunk.get("text", "")
        if not text.strip():
            continue
        skip = _find_leading_overlap(prev_text, text) if prev_text else 0
        display = text[skip:].lstrip("\n") if skip else text
        if display.strip():
            print(display)
        prev_text = text

    print()


def inspect_retrieval(query: str, config: RAGConfig, book_id: str | None = None) -> None:
    """Run a retrieval query and print the results with metadata."""
    retrieval = _new_retrieval(config)
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
