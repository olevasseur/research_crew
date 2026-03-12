"""Book analysis: hierarchical summarization with subchunk granularity.

Pipeline:
  1. For each section, summarize each subchunk individually (small, faithful).
  2. Synthesize per-section summary from subchunk summaries.
  3. Synthesize full-book summary from section summaries.

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
    allowed = set(SUMMARIZE_MODES.get(mode, SUMMARIZE_MODES["default"]))
    if include_types:
        allowed |= set(include_types)
    if exclude_types:
        allowed -= set(exclude_types)
    return allowed


# ---------------------------------------------------------------------------
# Prompts — subchunk level
# ---------------------------------------------------------------------------

SUBCHUNK_SYSTEM = """\
You are a close-reading analyst. Extract key content from a book excerpt.
Rules:
- Only state what is explicitly in the text. No interpretation beyond what is written.
- Use specific names, numbers, examples from the text.
- Cite page numbers as [p.N] when present in the source.
- Be concise: aim for 150-300 words per subchunk summary."""

SUBCHUNK_PROMPT = """\
Excerpt from "{title}" by {author}
Section: {section_label} ({section_type})
Pages: {page_range}
Chunk ID: {chunk_id}

--- TEXT ---
{text}
--- END TEXT ---

Extract from this excerpt ONLY:

**Key claims**: The specific arguments or assertions made. State the claim, not the topic.

**Evidence & examples**: Named people, studies, statistics, anecdotes with concrete details (who, what, outcome). If none, say "None in this excerpt."

**Frameworks/concepts**: Any named or described framework, model, or defined term. If none, say "None."

**Actionable points**: Specific advice or prescriptions the author gives. If none, say "None."

Do NOT add generic commentary. Only report what this specific excerpt contains."""

# ---------------------------------------------------------------------------
# Prompts — section level
# ---------------------------------------------------------------------------

SECTION_SYSTEM = """\
You are a book analysis expert. Synthesize subchunk summaries into a coherent section summary.
Rules:
- Preserve specific details: names, numbers, examples, page references.
- Deduplicate overlapping content (chunks may overlap).
- Flag the section's core argument distinctly from supporting details.
- No generic filler. Every sentence should carry information."""

SECTION_PROMPT = """\
Synthesize these subchunk summaries into a summary of section "{section_label}" from "{title}" by {author}.
Section type: {section_type} | Pages: {page_range} | Based on {n_chunks} subchunks.

--- SUBCHUNK SUMMARIES ---
{subchunk_summaries}
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
1–2 verbatim quotes with page numbers, only if clearly present in the subchunk summaries.
If none, write "No direct quotes extracted."

### Source Chunks
List the chunk IDs used: {chunk_ids}"""

# ---------------------------------------------------------------------------
# Prompts — book level
# ---------------------------------------------------------------------------

BOOK_SYNTHESIS_SYSTEM = """\
You are a book summary synthesiser. Produce a polished, structured summary
from section analyses. Every sentence must carry information. No padding or
vague references to "balance" or "intentionality" unless they are central
and specifically defined in the text."""

BOOK_SYNTHESIS_PROMPT = """\
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
    include_types: list[str] | None = None,
    exclude_types: list[str] | None = None,
) -> dict:
    """Hierarchical book analysis: subchunks → sections → book."""
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

    meta_by_name: dict[str, dict] = {s["name"]: s for s in sections_meta}

    section_results: list[dict] = []
    total_llm_calls = 0

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

        # --- Phase 1: Summarize each subchunk ---
        print(f"  ▶ {ch}  ({len(chunks)} chunks, pages {page_range})")
        subchunk_summaries: list[dict] = []
        for c in chunks:
            prompt = SUBCHUNK_PROMPT.format(
                title=title,
                author=author,
                section_label=ch,
                section_type=stype,
                page_range=c.get("page_range", "?"),
                chunk_id=c["id"],
                text=c["text"],
            )
            summary = llm.generate(prompt, system=SUBCHUNK_SYSTEM)
            total_llm_calls += 1
            subchunk_summaries.append({
                "chunk_id": c["id"],
                "page_range": c.get("page_range", "?"),
                "summary": summary,
            })
            print(f"    ✓ chunk {c['id']}  (p.{c.get('page_range', '?')})")

        # --- Phase 2: Synthesize section summary from subchunk summaries ---
        combined_subchunks = "\n\n---\n\n".join(
            f"[Chunk {sc['chunk_id']}, p.{sc['page_range']}]\n{sc['summary']}"
            for sc in subchunk_summaries
        )
        section_prompt = SECTION_PROMPT.format(
            title=title,
            author=author,
            section_label=ch,
            section_type=stype,
            page_range=page_range,
            n_chunks=len(chunks),
            subchunk_summaries=combined_subchunks,
            chunk_ids=", ".join(chunk_ids),
        )
        section_summary = llm.generate(section_prompt, system=SECTION_SYSTEM)
        total_llm_calls += 1
        print(f"    ✓ section synthesis done")

        section_results.append({
            "section": ch,
            "section_type": stype,
            "page_range": page_range,
            "parent": parent,
            "n_chunks": len(chunks),
            "chunk_ids": chunk_ids,
            "subchunk_summaries": subchunk_summaries,
            "section_summary": section_summary,
        })

    if not section_results:
        print("\nNo sections matched the filter. Try --mode full.")
        return {"section_results": [], "book_summary": ""}

    # --- Phase 3: Book-level synthesis ---
    print(f"\n  ▶ Synthesizing book summary from {len(section_results)} sections …")
    all_section_text = "\n\n---\n\n".join(
        f"### {sr['section']} ({sr['section_type']}, p.{sr['page_range']})\n{sr['section_summary']}"
        for sr in section_results
    )
    book_summary = llm.generate(
        BOOK_SYNTHESIS_PROMPT.format(
            title=title, author=author, section_summaries=all_section_text,
        ),
        system=BOOK_SYNTHESIS_SYSTEM,
    )
    total_llm_calls += 1

    # --- Save results ---
    results_dir = Path(config.storage.results_directory) / book_id
    results_dir.mkdir(parents=True, exist_ok=True)

    # Per-section insights with full provenance
    insights_parts: list[str] = []
    for sr in section_results:
        parent_str = f" (in {sr['parent']})" if sr.get("parent") else ""
        header = (
            f"## {sr['section']}{parent_str}\n"
            f"**Type:** {sr['section_type']} | "
            f"**Pages:** {sr['page_range']} | "
            f"**Chunks:** {sr['n_chunks']}\n"
            f"**Chunk IDs:** {', '.join(sr['chunk_ids'])}\n"
        )
        insights_parts.append(header + "\n" + sr["section_summary"])
    (results_dir / "chapter_insights.md").write_text("\n\n---\n\n".join(insights_parts))

    # Subchunk-level summaries for inspection
    subchunk_data: dict[str, list[dict]] = {}
    for sr in section_results:
        subchunk_data[sr["section"]] = sr["subchunk_summaries"]
    (results_dir / "subchunk_summaries.json").write_text(json.dumps(subchunk_data, indent=2))

    (results_dir / "book_summary.md").write_text(book_summary)
    (results_dir / "chunk_map.json").write_text(json.dumps(
        {sr["section"]: sr["chunk_ids"] for sr in section_results}, indent=2
    ))

    print(f"\n  ✓ Done. {total_llm_calls} LLM calls. Results → {results_dir}/")
    return {"section_results": section_results, "book_summary": book_summary}
