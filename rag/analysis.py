"""Book analysis: per-chapter insights and full-book summary.

Uses deterministic retrieval + direct LLM calls for each chapter, then
synthesises a full book summary.  Every claim is grounded in source chunks.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import RAGConfig
from .llm import LLMClient
from .retrieval import Retrieval

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
Analyse the following chapter from "{title}" by {author}.

Chapter: {chapter}

Source text (from the book — these are verbatim excerpts):
{chunks}

Produce the following sections. Be thorough but dense; no filler.

### Core Ideas
2–5 key ideas. State the argument or claim, not just the topic.

### Key Examples
The most important examples, case studies, or stories with context (who, what, outcome).
If none, write "No named example in this chapter."

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
Synthesise the chapter analyses below into a structured summary of
"{title}" by {author}.

Chapter Analyses:
{chapter_summaries}

Output these sections in order:

## Book Metadata
- **Title**: {title}
- **Author**: {author}
- **Domain/Category**: (infer from content)
- **One-paragraph summary**: central thesis of the book

## Chapter-by-Chapter Summary
For each chapter, a focused paragraph with distinct content (core idea, top example,
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

def analyse_book(book_id: str, config: RAGConfig) -> dict:
    """Run full book analysis. Returns dict with chapter_insights and book_summary."""
    retrieval = Retrieval(config)
    llm = LLMClient(config.generation)

    book_info = retrieval.store.get_book_info(book_id)
    if not book_info:
        raise ValueError(f"Book '{book_id}' not found. Ingest it first.")

    title = book_info["title"]
    author = book_info["author"]
    chapters = retrieval.get_book_chapters(book_id)

    # -- per-chapter analysis --
    chapter_insights: list[dict] = []
    for ch in chapters:
        chunks = retrieval.get_chapter_chunks(book_id, ch)
        if not chunks:
            chapter_insights.append({"chapter": ch, "summary": "(no content)", "chunk_ids": []})
            continue

        chunk_text = retrieval.format_chunks_for_prompt(chunks)
        prompt = CHAPTER_PROMPT.format(
            title=title, author=author, chapter=ch, chunks=chunk_text,
        )
        summary = llm.generate(prompt, system=ANALYST_SYSTEM)
        chapter_insights.append({
            "chapter": ch,
            "summary": summary,
            "chunk_ids": [c["id"] for c in chunks],
        })
        print(f"  ✓ {ch}")

    # -- book-level synthesis --
    all_chapter_text = "\n\n---\n\n".join(
        f"### {ci['chapter']}\n{ci['summary']}" for ci in chapter_insights
    )
    prompt = BOOK_SYNTHESIS_PROMPT.format(
        title=title, author=author, chapter_summaries=all_chapter_text,
    )
    book_summary = llm.generate(prompt, system=BOOK_SYNTHESIS_SYSTEM)

    # -- save results --
    results_dir = Path(config.storage.results_directory) / book_id
    results_dir.mkdir(parents=True, exist_ok=True)

    (results_dir / "chapter_insights.md").write_text(
        "\n\n---\n\n".join(f"## {ci['chapter']}\n\n{ci['summary']}" for ci in chapter_insights)
    )
    (results_dir / "book_summary.md").write_text(book_summary)

    # Save chunk IDs per chapter for traceability
    (results_dir / "chunk_map.json").write_text(json.dumps(
        {ci["chapter"]: ci["chunk_ids"] for ci in chapter_insights}, indent=2
    ))

    print(f"\nResults saved to {results_dir}/")
    return {"chapter_insights": chapter_insights, "book_summary": book_summary}
