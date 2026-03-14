"""Book analysis: selective windowed summarization with caching.

Pipeline:
  1. Group each section's subchunks into summary windows.
  2. Select the most informative windows (budget-based, not exhaustive).
  3. Summarize only selected windows (cached).
  4. Synthesize section summary from selected window summaries (cached).
  5. Synthesize book summary from section summaries (cached).

Retrieval granularity stays fine (small chunks in vector store).
Summarization is budget-controlled: fast=2, default=4, thorough=all windows/section.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
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
# Windowing
# ---------------------------------------------------------------------------

def build_windows(
    chunks: list[dict],
    window_size: int,
    window_overlap: int,
) -> list[list[dict]]:
    """Group sorted chunks into windows of approximately window_size characters."""
    if not chunks:
        return []

    windows: list[list[dict]] = []
    current_window: list[dict] = []
    current_size = 0

    for c in chunks:
        clen = len(c.get("text", ""))
        if current_size + clen > window_size and current_window:
            windows.append(current_window)
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

    if current_window:
        windows.append(current_window)

    return windows


# ---------------------------------------------------------------------------
# Window selection — lightweight heuristic ranking
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "the a an and or but in on at to for of is it this that with as by from be "
    "are was were been have has had do does did will would could should may might "
    "can shall not no so if than they he she we you i my our his her its their".split()
)


def _keyword_score(text: str, section_title: str) -> float:
    """Score a window's keyword richness + overlap with section title."""
    words = re.findall(r"[a-z]{3,}", text.lower())
    if not words:
        return 0.0
    content_words = [w for w in words if w not in _STOPWORDS]
    richness = len(set(content_words)) / max(len(content_words), 1)

    title_words = set(re.findall(r"[a-z]{3,}", section_title.lower())) - _STOPWORDS
    if title_words:
        overlap = len(title_words & set(content_words)) / len(title_words)
    else:
        overlap = 0.0

    return richness * 0.6 + overlap * 0.4


def _position_score(idx: int, total: int) -> float:
    """Favour opening and closing windows. Middle windows get lower scores."""
    if total <= 1:
        return 1.0
    if idx == 0:
        return 1.0
    if idx == total - 1:
        return 0.9
    if idx == 1:
        return 0.7
    # Linear decay in the middle
    return max(0.2, 0.6 - 0.4 * abs(idx - total / 2) / (total / 2))


def select_windows(
    windows: list[list[dict]],
    budget: int,
    section_title: str,
    always_first: bool = True,
    always_last: bool = True,
    strategy: str = "position_keyword",
) -> list[tuple[int, list[dict], float]]:
    """Select the top-budget windows by heuristic score.

    Returns list of (window_index, window_chunks, score), sorted by window_index.
    """
    if not windows:
        return []
    if budget >= len(windows) or strategy == "all":
        return [(i, w, 1.0) for i, w in enumerate(windows)]

    scores: list[tuple[int, float]] = []
    for i, w in enumerate(windows):
        w_text = " ".join(c.get("text", "")[:500] for c in w)
        ks = _keyword_score(w_text, section_title)
        ps = _position_score(i, len(windows))
        scores.append((i, ks * 0.5 + ps * 0.5))

    # Reserve slots for first/last if configured
    forced: set[int] = set()
    if always_first:
        forced.add(0)
    if always_last and len(windows) > 1:
        forced.add(len(windows) - 1)

    remaining_budget = budget - len(forced)

    # Sort non-forced windows by score, pick top remaining
    candidates = [(i, s) for i, s in scores if i not in forced]
    candidates.sort(key=lambda x: x[1], reverse=True)

    selected_indices = set(forced)
    for i, _ in candidates:
        if len(selected_indices) >= budget:
            break
        selected_indices.add(i)
        remaining_budget -= 1

    # Return sorted by position, with scores
    score_map = dict(scores)
    result = [
        (i, windows[i], score_map[i])
        for i in sorted(selected_indices)
    ]
    return result


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
Window {window_idx} of {total_windows} (selected for summarization) | Chunks: {chunk_ids}

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
You are a book analysis expert. You synthesize window summaries into a precise section summary.
Rules:
- Every sentence must carry specific information. No filler.
- Preserve names, numbers, page references, and concrete details.
- Deduplicate overlapping content from adjacent windows.
- State the section's core argument as a specific claim, not a topic label.
- Do not use vague phrases like "explores the concept of" or "discusses the importance of."
  Instead, state what the author argues and why."""

SECTION_PROMPT = """\
Synthesize these window summaries into a summary of section "{section_label}" from "{title}" by {author}.
Section type: {section_type} | Pages: {page_range}
Based on {n_selected} selected windows out of {n_total} total (covering {n_chunks} retrieval chunks).

--- WINDOW SUMMARIES ---
{window_summaries}
--- END ---

Produce:

### Core Argument
One paragraph stating the section's central claim. Be specific: what does the author argue, and what is the key evidence? Do not say "the author explores X" — say what the author's position on X actually is.

### Key Supporting Ideas
3–5 bullet points. Each states a distinct supporting argument or insight, not a restatement of the core argument.

### Strongest Examples
The most compelling named examples, case studies, or stories. For each: who, what happened, and what it demonstrates. If none, write "No concrete examples in this section."

### Frameworks & Mental Models
Named frameworks with one-sentence descriptions. If none, write "None."

### Actionable Takeaways
Specific actions a reader could take, grounded in the text. If none, write "None prescribed."

### Notable Quotes
1–2 verbatim quotes with page numbers if present. If none, write "None extracted."

### Source
Selected windows: {selected_window_ids} | All chunk IDs: {chunk_ids}"""

BOOK_SYSTEM = """\
You are a book summary synthesiser. Produce a polished, high-signal summary.
Rules:
- Every sentence must carry specific information from the section analyses.
- Do NOT use vague thematic labels ("balance", "intentionality", "mindfulness")
  unless you can cite a specific definition, framework, or argument from the text.
- Cross-cutting themes must reference specific sections and examples, not be generic platitudes.
- If two sections make similar points, merge them rather than repeating."""

BOOK_PROMPT = """\
Synthesise the section analyses below into a structured summary of
"{title}" by {author}.

Section Analyses:
{section_summaries}

Output these sections:

## Book Metadata
- **Title**: {title}
- **Author**: {author}
- **Domain/Category**: (infer)
- **Central thesis**: One paragraph. State the author's core argument as a specific claim with the key supporting logic. Avoid vague phrasing.

## Section-by-Section Summary
For each analysed section, one focused paragraph:
- The section's core argument (the actual claim, not "this section discusses X")
- The strongest specific example
- Any framework introduced (by name)
- One actionable takeaway if present
Each section MUST contribute at least one unique idea.

## Cross-Cutting Themes
3–5 themes. For each:
- **Theme**: specific name (not generic words like "balance")
- **Evidence**: cite specific examples or arguments from at least 2 sections

## Frameworks & Mental Models
For each: **Name** — one-sentence definition with the section it appears in.

## Top Actionable Items
5–8 distinct, specific actions. Each actionable without re-reading the book.

## Connections & Building Blocks
Ideas that could combine with other domains:
- **Idea**: brief description
- **Pairs with**: specific domain/concept
- **Why**: what the combination produces
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse_book(
    book_id: str,
    config: RAGConfig,
    mode: str = "default",
    quality: str | None = None,
    section_filter: str | None = None,
    include_types: list[str] | None = None,
    exclude_types: list[str] | None = None,
    force: bool = False,
    verify: bool | None = None,
) -> dict:
    """Selective windowed book analysis with caching and resume.

    Args:
        quality: "fast", "default", or "thorough"
        section_filter: if set, only summarize this one section name
        force: clear cache and recompute
        verify: run verification; None = use config default
    """
    retrieval = Retrieval(config)
    llm = LLMClient(config.generation)
    summ_cfg = config.summarization

    quality = quality or "default"
    budget = summ_cfg.budget_for(quality)
    window_size = summ_cfg.window_size
    window_overlap = summ_cfg.window_overlap
    strategy = summ_cfg.selection_strategy
    always_first = summ_cfg.always_include_first
    always_last = summ_cfg.always_include_last

    book_info = retrieval.store.get_book_info(book_id)
    if not book_info:
        raise ValueError(f"Book '{book_id}' not found. Ingest it first.")

    title = book_info["title"]
    author = book_info["author"]
    sections_meta = book_info.get("sections", [])
    chapters = retrieval.get_book_chapters(book_id)

    allowed_types = resolve_section_filter(mode, include_types, exclude_types)
    meta_by_name: dict[str, dict] = {s["name"]: s for s in sections_meta}

    cache = SummaryCache(summ_cfg.cache_dir, book_id, config.generation.model, quality)
    if force:
        cache.clear()
        print("  Cache cleared (--force)")

    print(f"  Quality={quality}, budget={budget} windows/section, strategy={strategy}")

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

        if section_filter and ch != section_filter:
            print(f"  ⊘ SKIP  {ch}  (not target section)")
            continue

        chunks = retrieval.get_chapter_chunks(book_id, ch)
        if not chunks:
            print(f"  ⊘ SKIP  {ch}  (no chunks)")
            continue

        chunks.sort(key=lambda c: c.get("chunk_index", 0))
        chunk_ids = [c["id"] for c in chunks]

        # --- Build all windows, then select ---
        all_windows = build_windows(chunks, window_size, window_overlap)
        selected = select_windows(
            all_windows, budget, ch,
            always_first=always_first, always_last=always_last,
            strategy=strategy,
        )

        skipped = len(all_windows) - len(selected)
        print(f"  ▶ {ch}  ({len(chunks)} chunks → {len(all_windows)} windows → {len(selected)} selected, {skipped} skipped)")

        # --- Phase 1: Summarize selected windows (cached) ---
        window_summaries: list[dict] = []
        window_keys: list[str] = []
        for wi, window_chunks, score in selected:
            w_chunk_ids = [c["id"] for c in window_chunks]
            w_key = cache.window_key(w_chunk_ids)
            window_keys.append(w_key)

            cached = cache.get_window(w_key)
            if cached is not None:
                window_summaries.append({
                    "window_idx": wi, "chunk_ids": w_chunk_ids,
                    "summary": cached, "score": score, "cached": True,
                })
                w_pages = f"{window_chunks[0].get('page_range', '?').split('-')[0]}-{window_chunks[-1].get('page_range', '?').split('-')[-1]}"
                print(f"    ✓ window {wi+1}/{len(all_windows)}  (CACHED, score={score:.2f}, p.{w_pages})")
                continue

            w_text = "\n\n".join(c.get("text", "") for c in window_chunks)
            w_pages = f"{window_chunks[0].get('page_range', '?').split('-')[0]}-{window_chunks[-1].get('page_range', '?').split('-')[-1]}"

            prompt = WINDOW_PROMPT.format(
                title=title, author=author,
                section_label=ch, section_type=stype,
                page_range=w_pages,
                window_idx=wi + 1, total_windows=len(all_windows),
                chunk_ids=", ".join(w_chunk_ids),
                text=w_text,
            )
            summary = llm.generate(prompt, system=WINDOW_SYSTEM)
            total_llm_calls += 1

            cache.put_window(w_key, summary, w_chunk_ids)
            window_summaries.append({
                "window_idx": wi, "chunk_ids": w_chunk_ids,
                "summary": summary, "score": score, "cached": False,
            })
            print(f"    ✓ window {wi+1}/{len(all_windows)}  (score={score:.2f}, p.{w_pages})")

        # --- Phase 2: Synthesize section summary (cached) ---
        s_key = cache.section_key(ch, window_keys)
        section_keys.append(s_key)

        cached_section = cache.get_section(s_key)
        if cached_section is not None:
            print(f"    ✓ section synthesis  (CACHED)")
            section_summary = cached_section
        else:
            combined_windows = "\n\n---\n\n".join(
                f"[Window {ws['window_idx']+1}, score={ws['score']:.2f}, chunks: {', '.join(ws['chunk_ids'])}]\n{ws['summary']}"
                for ws in window_summaries
            )
            section_prompt = SECTION_PROMPT.format(
                title=title, author=author,
                section_label=ch, section_type=stype,
                page_range=page_range,
                n_selected=len(selected), n_total=len(all_windows), n_chunks=len(chunks),
                window_summaries=combined_windows,
                selected_window_ids=", ".join(str(ws["window_idx"]+1) for ws in window_summaries),
                chunk_ids=", ".join(chunk_ids),
            )
            section_summary = llm.generate(section_prompt, system=SECTION_SYSTEM)
            total_llm_calls += 1
            cache.put_section(s_key, section_summary, ch, window_keys, meta={
                "quality": quality, "budget": budget,
                "n_windows_total": len(all_windows), "n_windows_selected": len(selected),
                "n_chunks": len(chunks), "strategy": strategy,
            })
            print(f"    ✓ section synthesis done")

        section_results.append({
            "section": ch,
            "section_type": stype,
            "page_range": page_range,
            "parent": parent,
            "n_chunks": len(chunks),
            "n_windows_total": len(all_windows),
            "n_windows_selected": len(selected),
            "chunk_ids": chunk_ids,
            "selected_windows": [
                {"window_idx": ws["window_idx"], "score": ws["score"],
                 "chunk_ids": ws["chunk_ids"], "cached": ws["cached"]}
                for ws in window_summaries
            ],
            "window_summaries": window_summaries,
            "section_summary": section_summary,
        })

    if not section_results:
        print("\nNo sections matched the filter. Try --mode full or check --section name.")
        return {"section_results": [], "book_summary": ""}

    # --- Phase 3: Book-level synthesis (cached, skip if single-section) ---
    book_summary = ""
    if section_filter:
        print(f"\n  ⊘ Skipping book synthesis (single-section mode)")
    else:
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
            cache.put_book(b_key, book_summary, meta={
                "quality": quality, "n_sections": len(section_results),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

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
            f"**Windows:** {sr['n_windows_selected']}/{sr['n_windows_total']} selected\n"
            f"**Selected windows:** {', '.join(str(w['window_idx']+1) for w in sr['selected_windows'])}\n"
        )
        insights_parts.append(header + "\n" + sr["section_summary"])
    (results_dir / "chapter_insights.md").write_text("\n\n---\n\n".join(insights_parts))

    window_data: dict[str, list[dict]] = {}
    for sr in section_results:
        window_data[sr["section"]] = [
            {"window": ws["window_idx"], "chunk_ids": ws["chunk_ids"],
             "score": ws["score"], "cached": ws["cached"], "summary": ws["summary"]}
            for ws in sr["window_summaries"]
        ]
    (results_dir / "window_summaries.json").write_text(json.dumps(window_data, indent=2))

    if book_summary:
        (results_dir / "book_summary.md").write_text(book_summary)

    (results_dir / "chunk_map.json").write_text(json.dumps(
        {sr["section"]: sr["chunk_ids"] for sr in section_results}, indent=2
    ))

    # Save summary metadata
    summary_meta = {
        "book_id": book_id,
        "quality": quality,
        "mode": mode,
        "budget_per_section": budget,
        "strategy": strategy,
        "model": config.generation.model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_llm_calls": total_llm_calls,
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "sections": [
            {
                "name": sr["section"],
                "type": sr["section_type"],
                "pages": sr["page_range"],
                "chunks": sr["n_chunks"],
                "windows_total": sr["n_windows_total"],
                "windows_selected": sr["n_windows_selected"],
                "selected_window_indices": [w["window_idx"] for w in sr["selected_windows"]],
            }
            for sr in section_results
        ],
    }
    (results_dir / "summary_meta.json").write_text(json.dumps(summary_meta, indent=2))

    stats = cache.stats()
    print(f"\n  ✓ Done. {total_llm_calls} LLM calls, "
          f"{stats['hits']} cache hits, {stats['misses']} misses. "
          f"Results → {results_dir}/")

    # --- Optional verification ---
    do_verify = verify if verify is not None else summ_cfg.verify_by_default
    if do_verify:
        from .critic import verify_book_summary
        print("\n  Running verification …")
        verify_book_summary(book_id, config)

    return {"section_results": section_results, "book_summary": book_summary}
