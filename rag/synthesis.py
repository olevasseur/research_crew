"""Cross-book synthesis: compare books, find connections, generate novel ideas."""

from __future__ import annotations

from pathlib import Path

from .config import RAGConfig
from .llm import LLMClient
from .retrieval import Retrieval

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CONCEPT_SYSTEM = """\
You are a concept extraction agent. Given a book summary, extract the 5–10 most
important concepts, frameworks, or arguments as short phrases (one line each)."""

CONCEPT_PROMPT = """\
Extract the key concepts from this book summary. Output one concept per line,
no numbering, no bullets.

{summary}"""

CROSS_SYSTEM = """\
You are an idea cross-pollinator. You find connections across books and generate
novel insights. Every claim must reference a specific book and concept.
Keep it concrete and actionable."""

CROSS_PROMPT = """\
You are comparing these books:
{book_list}

Key concepts per book:
{concepts}

Related excerpts found across books:
{cross_chunks}

Produce:

## Similarities
Shared ideas across the books with citations.

## Contradictions
Ideas where the books disagree, with citations.

## Novel Synthesised Ideas
New ideas formed by combining concepts from different books. For each:
- **Idea**: description
- **Sources**: which book concepts it combines
- **Potential application**: startup / product / personal insight

## Startup & Product Insights
Concrete product or business ideas inspired by the cross-pollination.
"""

ASK_SYSTEM = """\
You are a research assistant with access to multiple book excerpts.
Answer the question using only the provided excerpts. Cite each claim
with the book title, chapter, and page range."""

ASK_PROMPT = """\
Question: {question}

Relevant excerpts from your library:
{chunks}

Answer the question using only the excerpts above. For every claim,
include a citation like (Book Title, Chapter X, p.N-M).
If the excerpts don't contain enough information, say so."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare_books(book_ids: list[str], config: RAGConfig) -> str:
    """Cross-pollinate ideas across the given books."""
    retrieval = Retrieval(config)
    llm = LLMClient(config.generation)

    # 1. Load summaries (must be analysed first)
    summaries: dict[str, str] = {}
    for bid in book_ids:
        summary_path = Path(config.storage.results_directory) / bid / "book_summary.md"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"No summary for '{bid}'. Run `summarize {bid}` first."
            )
        summaries[bid] = summary_path.read_text()

    # 2. Extract key concepts per book
    concepts: dict[str, str] = {}
    for bid, summary in summaries.items():
        concepts[bid] = llm.generate(
            CONCEPT_PROMPT.format(summary=summary), system=CONCEPT_SYSTEM,
        )
        print(f"  Extracted concepts for {bid}")

    # 3. Cross-retrieve: for each concept in book A, find related chunks in book B
    cross_chunks: list[dict] = []
    for bid, concept_text in concepts.items():
        other_ids = [b for b in book_ids if b != bid]
        for line in concept_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            results = retrieval.search_books(line, other_ids, top_k=3)
            cross_chunks.extend(results)

    # Deduplicate by chunk id
    seen = set()
    unique = []
    for c in cross_chunks:
        if c["id"] not in seen:
            seen.add(c["id"])
            unique.append(c)
    cross_chunks = unique[: config.retrieval.max_context_chunks]

    # 4. Synthesise
    book_list = "\n".join(
        f"- {bid}: {retrieval.store.get_book_info(bid).get('title', bid)}"
        for bid in book_ids
    )
    concepts_text = "\n\n".join(
        f"### {bid}\n{text}" for bid, text in concepts.items()
    )
    chunk_text = retrieval.format_chunks_for_prompt(cross_chunks)

    result = llm.generate(
        CROSS_PROMPT.format(
            book_list=book_list, concepts=concepts_text, cross_chunks=chunk_text,
        ),
        system=CROSS_SYSTEM,
    )

    # 5. Save
    results_dir = Path(config.storage.results_directory) / "cross"
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = "_vs_".join(sorted(book_ids))
    (results_dir / f"{tag}.md").write_text(result)
    print(f"\nCross-synthesis saved to {results_dir / tag}.md")
    return result


def ask_question(question: str, config: RAGConfig, book_ids: list[str] | None = None) -> str:
    """Answer a cross-pollination question grounded in the library."""
    retrieval = Retrieval(config)
    llm = LLMClient(config.generation)

    if book_ids:
        chunks = retrieval.search_books(question, book_ids, top_k=config.retrieval.top_k)
    else:
        chunks = retrieval.search_all(question, top_k=config.retrieval.top_k)

    if not chunks:
        return "No relevant excerpts found in the library."

    chunk_text = retrieval.format_chunks_for_prompt(chunks)
    result = llm.generate(
        ASK_PROMPT.format(question=question, chunks=chunk_text),
        system=ASK_SYSTEM,
    )
    return result
