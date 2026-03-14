"""Book analysis: windowed hierarchical summarization with caching.

Pipeline:
  1. Group each section's subchunks into summary windows (~8000 chars each).
  2. Summarize each window (cached).
  3. Synthesize section summary from window summaries (cached).
  4. Synthesize book summary from section summaries (cached).

Retrieval granularity stays fine (small chunks in vector store).
Summarization granularity is coarse (few large windows per section).
"""

from __future__ import annotations

import json
from pathlib import Path

from .cache import SummaryCache
from .chunker import SECTION_TYPES_SUMMARIZABLE, SECTION_TYPES_SKIPPABLE
from .config import RAGConfig, SummarizationConfig
from .llm import LLMClient
from .retrieval import Retrieval

# ---------------------------------------------------------------------------
# Section type filters
# ---------------------------------------------------------------------------

SUMMARIZE_MODES = {
    "default": SECTION_TYPES_SUMMARIZABLE,
    "body-only": {"chapter"},
    "chapter-only": {"chapter"},
    "full": SECTION_TYPES_SUMMARIZABLE | SECTION_TYPES_SKIPPABLE,
    "back-matter": {"acknowledgments", "notes", "index", "about_author"},
}

# Quality presets override window sizing
QUALITY_PRESETS: dict[str, dict] = {
    "fast":     {"window_size": 12000, "max_windows_per_section": 8},
    "default":  {},  # use config as-is
    "thorough": {"window_size": 5000,  "max_windows_per_section": 25},
}


def resolve_section_filter(
    mode: str = "default",
    include_types: list[str] | None = None,
    exclude_types: list[str] | None = None,
) -> set[str]:
    allowed = set(SUMMARIZE_MODES.get(mode, SUMMARIZE_MODES["default"]))
    if include_types:
        allowed |= set(include_types)
    if exclude_types:
        allowed -= set(exclude_types)
    return allowed


# ---------------------------------------------------------------------------
# Windowing: group adjacent subchunks into larger summary windows
# ---------------------------------------------------------------------------

def build_windows(
    chunks: list[dict],
    window_size: int,
    window_overlap: int,
    max_windows: int,
) -> list[list[dict]]:
    """Group sorted chunks into windows of approximately window_size characters.

    Each window is a list of chunk dicts. Adjacent windows overlap by
    window_overlap characters worth of chunks.
    """
    if not chunks:
        return []

    windows: list[list[dict]] = []
    current_window: list[dict] = []
    current_size = 0

    for c in chunks:
        clen = len(c.get("text", ""))
        if current_size + clen > window_size and current_window:
            windows.append(current_window)
            if len(windows) >= max_windows:
                break
            # Overlap: keep trailing chunks that fit within window_overlap chars
            overlap_chunks: list[dict] = []
            overlap_size = 0
            for oc in reversed(current_window):
                oc_len = len(oc.get("text", ""))
                if overlap_size + oc_len > window_overlap:
                    break
                overlap_chunks.insert(0, oc)
                overlap_size += oc_len
            current_window = list(overlap_chunks)
            current_size = overlap_size

        current_window.append(c)
        current_size += clen

    if current_window and len(windows) < max_windows:
        windows.append(current_window)

    return windows


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

WINDOW_SYSTEM = """\
You are a close-reading analyst. Extract key content from a book excerpt.
Rules:
- Only state what is explicitly in the text. No interpretation beyond what is written.
- Use specific names, numbers, examples from the text.
- Cite page numbers as [p.N] when present in the source.
- Be concise but thorough: cover all major points in the excerpt."""

WINDOW_PROMPT = """\
Excerpt from "{title}" by {author}
Section: {section_label} ({section_type})
Pages: {page_range}
Window {window_idx} of {total_windows} | Chunks: {chunk_ids}

--- TEXT ---
{text}
--- END TEXT ---

Extract from this excerpt ONLY:

**Key claims**: The specific arguments or assertions made. State the claim, not the topic.

**Evidence & examples**: Named people, studies, statistics, anecdotes with concrete details.

**Frameworks/concepts**: Any named or described framework, model, or defined term.

**Actionable points**: Specific advice or prescriptions the author gives.

Do NOT add generic commentary. Only report what this specific excerpt contains."""

SECTION_SYSTEM = """\
You are a book analysis expert. Synthesize window summaries into a coherent section summary.
Rules:
- Preserve specific details: names, numbers, examples, page references.
- Deduplicate overlapping content (windows may overlap).
- Flag the section's core argument distinctly from supporting details.
- No generic filler. Every sentence should carry information."""

SECTION_PROMPT = """\
Synthesize these window summaries into a summary of section "{section_label}" from "{title}" by {author}.
Section type: {section_type} | Pages: {page_range} | Based on {n_windows} windows covering {n_chunks} chunks.

--- WINDOW SUMMARIES ---
{window_summaries}
--- END ---

Produce:

### Core Argument
One paragraph: what is the main claim or purpose of this section? Be specific.

### Key Supporting Ideas
3–5 bullet points. Each should state a distinct idea, not repeat the core argument.

### Strongest Examples
The most compelling examples, case studies, or stories. Include who/what/outcome.
If none, write "No concrete examples in this section."

### Frameworks & Mental Models
Named or described frameworks. For each: name and one-sentence description.
If none, write "None introduced in this section."

### Actionable Takeaways
Concrete actions a reader could take based on this section.
If none, write "No specific actions prescribed."

### Notable Quotes
1–2 verbatim quotes with page numbers, only if clearly present in the window summaries.
If none, write "No direct quotes extracted."

### Source Chunks
{chunk_ids}"""

BOOK_SYSTEM = """\
You are a book summary synthesiser. Produce a polished, structured summary
from section analyses. Every sentence must carry information. No padding or
vague references to "balance" or "intentionality" unless they are central
and specifically defined in the text."""

BOOK_PROMPT = """\
Synthesise the section analyses below into a structured summary of
"{title}" by {author}.

Section Analyses:
{section_summaries}

Output these sections in order:

## Book Metadata
- **Title**: {title}
- **Author**: {author}
- **Domain/Category**: (infer from content)
- **Central thesis**: One paragraph. State the author's core argument with specificity.

## Section-by-Section Summary
For each analysed section, one focused paragraph containing:
- The section's core argument (not a topic label, but the actual claim)
- The strongest example or piece of evidence
- Any framework introduced
- One actionable takeaway if present
Each section MUST contribute at least one idea not found in other sections.

## Cross-Cutting Themes
3–7 themes that recur across sections. For each theme:
- **Theme**: name
- **How it appears**: specific instances across sections (not generic descriptions)

## Frameworks & Mental Models (Quick Reference)
For each: **Name** — one-sentence description with the section it comes from.

## Top Actionable Items
5–10 most impactful DISTINCT actions ranked by impact.
Each must be specific enough to act on without re-reading the book.

## Connections & Building Blocks
Ideas from this book that could combine with concepts from other domains:
- **Idea from this book**: brief description
- **Could pair with**: specific domain/concept
- **Why**: what the combination could produce
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse_book(
    book_id: str,
    config: RAGConfig,
    mode: str = "default",
    quality: str | None = None,
    include_types: list[str] | None = None,
    exclude_types: list[str] | None = None,
    force: bool = False,
    verify: bool | None = None,
) -> dict:
    """Windowed hierarchical book analysis with caching and resume.

    Args:
        quality: "fast", "default", or "thorough" — overrides config
        force: if True, ignore cache and recompute everything
        verify: if True, run verification; if None, use config default
    """
    retrieval = Retrieval(config)
    llm = LLMClient(config.generation)
    summ_cfg = config.summarization

    quality = quality or summ_cfg.quality
    preset = QUALITY_PRESETS.get(quality, {})
    window_size = preset.get("window_size", summ_cfg.window_size)
    window_overlap = summ_cfg.window_overlap
    max_windows = preset.get("max_windows_per_section", summ_cfg.max_windows_per_section)

    book_info = retrieval.store.get_book_info(book_id)
    if not book_info:
        raise ValueError(f"Book '{book_id}' not found. Ingest it first.")

    title = book_info["title"]
    author = book_info["author"]
    sections_meta = book_info.get("sections", [])
    chapters = retrieval.get_book_chapters(book_id)

    allowed_types = resolve_section_filter(mode, include_types, exclude_types)
    meta_by_name: dict[str, dict] = {s["name"]: s for s in sections_meta}

    cache = SummaryCache(summ_cfg.cache_dir, book_id, config.generation.model)
    if force:
        cache.clear()
        print("  Cache cleared (--force)")

    section_results: list[dict] = []
    total_llm_calls = 0
    section_keys: list[str] = []

    for ch in chapters:
        meta = meta_by_name.get(ch, {})
        stype = meta.get("section_type", "unknown")
        start_page = meta.get("start_page", "?")
        end_page = meta.get("end_page", "?")
        page_range = f"{start_page}-{end_page}"
        parent = meta.get("parent", "")

        if stype not in allowed_types:
            print(f"  ⊘ SKIP  {ch}  (type={stype})")
            continue

        chunks = retrieval.get_chapter_chunks(book_id, ch)
        if not chunks:
            print(f"  ⊘ SKIP  {ch}  (no chunks)")
            continue

        chunks.sort(key=lambda c: c.get("chunk_index", 0))
        chunk_ids = [c["id"] for c in chunks]

        # --- Build windows ---
        windows = build_windows(chunks, window_size, window_overlap, max_windows)

        print(f"  ▶ {ch}  ({len(chunks)} chunks → {len(windows)} windows, pages {page_range})")

        # --- Phase 1: Summarize each window (cached) ---
        window_summaries: list[dict] = []
        window_keys: list[str] = []
        for wi, window_chunks in enumerate(windows):
            w_chunk_ids = [c["id"] for c in window_chunks]
            w_key = cache.window_key(w_chunk_ids)
            window_keys.append(w_key)

            cached = cache.get_window(w_key)
            if cached is not None:
                window_summaries.append({
                    "window_idx": wi,
                    "chunk_ids": w_chunk_ids,
                    "summary": cached,
                })
                w_pages = f"{window_chunks[0].get('page_range', '?').split('-')[0]}-{window_chunks[-1].get('page_range', '?').split('-')[-1]}"
                print(f"    ✓ window {wi+1}/{len(windows)}  (CACHED, {len(window_chunks)} chunks, p.{w_pages})")
                continue

            w_text = "\n\n".join(c.get("text", "") for c in window_chunks)
            w_pages = f"{window_chunks[0].get('page_range', '?').split('-')[0]}-{window_chunks[-1].get('page_range', '?').split('-')[-1]}"

            prompt = WINDOW_PROMPT.format(
                title=title,
                author=author,
                section_label=ch,
                section_type=stype,
                page_range=w_pages,
                window_idx=wi + 1,
                total_windows=len(windows),
                chunk_ids=", ".join(w_chunk_ids),
                text=w_text,
            )
            summary = llm.generate(prompt, system=WINDOW_SYSTEM)
            total_llm_calls += 1

            cache.put_window(w_key, summary, w_chunk_ids)
            window_summaries.append({
                "window_idx": wi,
                "chunk_ids": w_chunk_ids,
                "summary": summary,
            })
            print(f"    ✓ window {wi+1}/{len(windows)}  ({len(window_chunks)} chunks, p.{w_pages})")

        # --- Phase 2: Synthesize section summary (cached) ---
        s_key = cache.section_key(ch, window_keys)
        section_keys.append(s_key)

        cached_section = cache.get_section(s_key)
        if cached_section is not None:
            print(f"    ✓ section synthesis  (CACHED)")
            section_summary = cached_section
        else:
            combined_windows = "\n\n---\n\n".join(
                f"[Window {ws['window_idx']+1}, chunks: {', '.join(ws['chunk_ids'])}]\n{ws['summary']}"
                for ws in window_summaries
            )
            section_prompt = SECTION_PROMPT.format(
                title=title,
                author=author,
                section_label=ch,
                section_type=stype,
                page_range=page_range,
                n_windows=len(windows),
                n_chunks=len(chunks),
                window_summaries=combined_windows,
                chunk_ids=", ".join(chunk_ids),
            )
            section_summary = llm.generate(section_prompt, system=SECTION_SYSTEM)
            total_llm_calls += 1
            cache.put_section(s_key, section_summary, ch, window_keys)
            print(f"    ✓ section synthesis done")

        section_results.append({
            "section": ch,
            "section_type": stype,
            "page_range": page_range,
            "parent": parent,
            "n_chunks": len(chunks),
            "n_windows": len(windows),
            "chunk_ids": chunk_ids,
            "window_summaries": window_summaries,
            "section_summary": section_summary,
        })

    if not section_results:
        print("\nNo sections matched the filter. Try --mode full.")
        return {"section_results": [], "book_summary": ""}

    # --- Phase 3: Book-level synthesis (cached) ---
    b_key = cache.book_key(section_keys)
    cached_book = cache.get_book(b_key)

    if cached_book is not None:
        print(f"\n  ▶ Book summary  (CACHED)")
        book_summary = cached_book
    else:
        print(f"\n  ▶ Synthesizing book summary from {len(section_results)} sections …")
        all_section_text = "\n\n---\n\n".join(
            f"### {sr['section']} ({sr['section_type']}, p.{sr['page_range']})\n{sr['section_summary']}"
            for sr in section_results
        )
        book_summary = llm.generate(
            BOOK_PROMPT.format(
                title=title, author=author, section_summaries=all_section_text,
            ),
            system=BOOK_SYSTEM,
        )
        total_llm_calls += 1
        cache.put_book(b_key, book_summary)

    # --- Save results ---
    results_dir = Path(config.storage.results_directory) / book_id
    results_dir.mkdir(parents=True, exist_ok=True)

    insights_parts: list[str] = []
    for sr in section_results:
        parent_str = f" (in {sr['parent']})" if sr.get("parent") else ""
        header = (
            f"## {sr['section']}{parent_str}\n"
            f"**Type:** {sr['section_type']} | "
            f"**Pages:** {sr['page_range']} | "
            f"**Chunks:** {sr['n_chunks']} | "
            f"**Windows:** {sr['n_windows']}\n"
            f"**Chunk IDs:** {', '.join(sr['chunk_ids'])}\n"
        )
        insights_parts.append(header + "\n" + sr["section_summary"])
    (results_dir / "chapter_insights.md").write_text("\n\n---\n\n".join(insights_parts))

    window_data: dict[str, list[dict]] = {}
    for sr in section_results:
        window_data[sr["section"]] = [
            {"window": ws["window_idx"], "chunk_ids": ws["chunk_ids"], "summary": ws["summary"]}
            for ws in sr["window_summaries"]
        ]
    (results_dir / "window_summaries.json").write_text(json.dumps(window_data, indent=2))

    (results_dir / "book_summary.md").write_text(book_summary)
    (results_dir / "chunk_map.json").write_text(json.dumps(
        {sr["section"]: sr["chunk_ids"] for sr in section_results}, indent=2
    ))

    stats = cache.stats()
    print(f"\n  ✓ Done. {total_llm_calls} LLM calls, "
          f"{stats['hits']} cache hits, {stats['misses']} cache misses. "
          f"Results → {results_dir}/")

    # --- Optional verification ---
    do_verify = verify if verify is not None else summ_cfg.verify_by_default
    if do_verify:
        from .critic import verify_book_summary
        print("\n  Running verification …")
        verify_book_summary(book_id, config)

    return {"section_results": section_results, "book_summary": book_summary}
