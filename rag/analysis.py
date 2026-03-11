"""Book analysis: per-section insights and full-book summary.

Uses deterministic retrieval + direct LLM calls for each section, then
synthesises a full book summary.  Every claim is grounded in source chunks.

Structure-aware: only meaningful body content is summarised by default.
"""

from __future__ import annotations

import json
from pathlib import Path

from .chunker import SECTION_TYPES_SUMMARIZABLE, SECTION_TYPES_SKIPPABLE
from .config import RAGConfig
from .llm import LLMClient
from .retrieval import Retrieval

# ---------------------------------------------------------------------------
# Summarize modes
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
    """Build the set of section_types to include based on mode + overrides."""
    allowed = set(SUMMARIZE_MODES.get(mode, SUMMARIZE_MODES["default"]))
    if include_types:
        allowed |= set(include_types)
    if exclude_types:
        allowed -= set(exclude_types)
    return allowed


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

ANALYST_SYSTEM = """\
You are a book analysis expert. You extract structured insights from book chapters.
Rules:
- Only make claims supported by the source text you are given.
- Be specific: use names, numbers, concrete details from the text.
- Cite page numbers when shown as [p.N] in the source.
- If the chapter has no notable content, say so briefly."""

CHAPTER_PROMPT = """\
Analyse the following section from "{title}" by {author}.

Section: {chapter}
Section type: {section_type}
Pages: {page_range}

Source text (from the book — these are verbatim excerpts):
{chunks}

Produce the following sections. Be thorough but dense; no filler.

### Core Ideas
2–5 key ideas. State the argument or claim, not just the topic.

### Key Examples
The most important examples, case studies, or stories with context (who, what, outcome).
If none, write "No named example in this section."

### Frameworks & Mental Models
Named or implied frameworks with a short description of how each works.

### Actionable Takeaways
Concrete, specific actions a reader could take.

### Notable Quotes
1–3 verbatim quotes with page numbers. If none, write "No notable quote extracted."
"""

BOOK_SYNTHESIS_SYSTEM = """\
You are a book summary synthesiser. You produce polished, structured summaries
from chapter analyses. Keep it dense and high-signal. No padding."""

BOOK_SYNTHESIS_PROMPT = """\
Synthesise the section analyses below into a structured summary of
"{title}" by {author}.

Section Analyses:
{chapter_summaries}

Output these sections in order:

## Book Metadata
- **Title**: {title}
- **Author**: {author}
- **Domain/Category**: (infer from content)
- **One-paragraph summary**: central thesis of the book

## Chapter-by-Chapter Summary
For each section, a focused paragraph with distinct content (core idea, top example,
top actionable item, any frameworks). Each chapter MUST add at least one idea not
restated from another chapter.

## Cross-Cutting Themes
3–7 themes that recur across chapters.

## Frameworks & Mental Models (Quick Reference)
Consolidated list. For each: **Name** — one-line description.

## Top Actionable Items
5–10 most impactful DISTINCT actions ranked by impact. Do not pad with restatements.

## Connections & Building Blocks
Ideas that could pair with concepts from other domains (software engineering,
business strategy, investing, entrepreneurship, productivity). For each:
- **Idea from this book**: brief description
- **Could pair with**: type of concept/domain
- **Why**: what the combination could produce
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse_book(
    book_id: str,
    config: RAGConfig,
    mode: str = "default",
    include_types: list[str] | None = None,
    exclude_types: list[str] | None = None,
) -> dict:
    """Run full book analysis. Returns dict with chapter_insights and book_summary.

    Args:
        mode: one of "default", "body-only", "chapter-only", "full", "back-matter"
        include_types: extra section_types to include
        exclude_types: section_types to exclude
    """
    retrieval = Retrieval(config)
    llm = LLMClient(config.generation)

    book_info = retrieval.store.get_book_info(book_id)
    if not book_info:
        raise ValueError(f"Book '{book_id}' not found. Ingest it first.")

    title = book_info["title"]
    author = book_info["author"]
    sections_meta = book_info.get("sections", [])
    chapters = retrieval.get_book_chapters(book_id)

    allowed_types = resolve_section_filter(mode, include_types, exclude_types)

    # Build a lookup: section name → metadata
    meta_by_name: dict[str, dict] = {}
    for s in sections_meta:
        meta_by_name[s["name"]] = s

    # -- per-section analysis --
    chapter_insights: list[dict] = []
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
            chapter_insights.append({
                "chapter": ch, "section_type": stype,
                "page_range": page_range, "parent": parent,
                "summary": "(no content)", "chunk_ids": [],
            })
            print(f"  ⊘ SKIP  {ch}  (no chunks)")
            continue

        chunk_text = retrieval.format_chunks_for_prompt(chunks)
        prompt = CHAPTER_PROMPT.format(
            title=title, author=author, chapter=ch,
            section_type=stype, page_range=page_range,
            chunks=chunk_text,
        )
        summary = llm.generate(prompt, system=ANALYST_SYSTEM)
        chapter_insights.append({
            "chapter": ch,
            "section_type": stype,
            "page_range": page_range,
            "parent": parent,
            "summary": summary,
            "chunk_ids": [c["id"] for c in chunks],
        })
        print(f"  ✓ {ch}  (type={stype}, pages={page_range}, chunks={len(chunks)})")

    if not chapter_insights:
        print("\nNo sections matched the filter. Try --mode full.")
        return {"chapter_insights": [], "book_summary": ""}

    # -- book-level synthesis --
    all_chapter_text = "\n\n---\n\n".join(
        f"### {ci['chapter']} ({ci['section_type']}, p.{ci['page_range']})\n{ci['summary']}"
        for ci in chapter_insights
    )
    prompt = BOOK_SYNTHESIS_PROMPT.format(
        title=title, author=author, chapter_summaries=all_chapter_text,
    )
    book_summary = llm.generate(prompt, system=BOOK_SYNTHESIS_SYSTEM)

    # -- save results --
    results_dir = Path(config.storage.results_directory) / book_id
    results_dir.mkdir(parents=True, exist_ok=True)

    chapter_insights_md = []
    for ci in chapter_insights:
        parent_str = f" (in {ci['parent']})" if ci.get("parent") else ""
        header = (
            f"## {ci['chapter']}{parent_str}\n"
            f"**Type:** {ci['section_type']} | "
            f"**Pages:** {ci['page_range']} | "
            f"**Chunks:** {', '.join(ci['chunk_ids'])}\n"
        )
        chapter_insights_md.append(header + "\n" + ci["summary"])

    (results_dir / "chapter_insights.md").write_text(
        "\n\n---\n\n".join(chapter_insights_md)
    )
    (results_dir / "book_summary.md").write_text(book_summary)
    (results_dir / "chunk_map.json").write_text(json.dumps(
        {ci["chapter"]: ci["chunk_ids"] for ci in chapter_insights}, indent=2
    ))

    print(f"\nResults saved to {results_dir}/")
    return {"chapter_insights": chapter_insights, "book_summary": book_summary}
