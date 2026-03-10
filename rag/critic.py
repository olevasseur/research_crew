"""Critic / verifier agent: checks that claims are grounded in source chunks."""

from __future__ import annotations

from pathlib import Path

from .config import RAGConfig
from .llm import LLMClient
from .retrieval import Retrieval

CRITIC_SYSTEM = """\
You are a verification agent. Your job is to check that every claim in a summary
is supported by the source excerpts. Be precise and skeptical."""

CRITIC_PROMPT = """\
Below is a summary followed by the source excerpts it should be grounded in.

=== SUMMARY ===
{summary}

=== SOURCE EXCERPTS ===
{sources}

Review the summary and:
1. For each major claim, check if it is supported by the source excerpts.
2. Mark unsupported claims with [UNSUPPORTED: reason].
3. Note significant omissions from the sources with [MISSING: topic].
4. Rate overall grounding: HIGH / MEDIUM / LOW.

Output the verified summary with inline flags, followed by:

## Verification Assessment
- **Grounding**: HIGH / MEDIUM / LOW
- **Unsupported claims**: count
- **Missing topics**: count
- **Notes**: brief commentary
"""


def verify_book_summary(book_id: str, config: RAGConfig) -> str:
    """Verify a book summary against its source chunks."""
    retrieval = Retrieval(config)
    llm = LLMClient(config.generation)

    # Load the summary
    summary_path = Path(config.storage.results_directory) / book_id / "book_summary.md"
    if not summary_path.exists():
        raise FileNotFoundError(f"No summary for '{book_id}'. Run summarize first.")
    summary = summary_path.read_text()

    # Retrieve source chunks — sample across chapters for coverage
    chapters = retrieval.get_book_chapters(book_id)
    source_chunks: list[dict] = []
    per_chapter = max(3, config.retrieval.max_context_chunks // len(chapters)) if chapters else 5
    for ch in chapters:
        chunks = retrieval.get_chapter_chunks(book_id, ch)
        source_chunks.extend(chunks[:per_chapter])

    source_text = retrieval.format_chunks_for_prompt(
        source_chunks[: config.retrieval.max_context_chunks]
    )

    result = llm.generate(
        CRITIC_PROMPT.format(summary=summary, sources=source_text),
        system=CRITIC_SYSTEM,
    )

    # Save
    results_dir = Path(config.storage.results_directory) / book_id
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "verification.md").write_text(result)
    print(f"Verification saved to {results_dir / 'verification.md'}")
    return result


def verify_text(text: str, book_id: str, config: RAGConfig) -> str:
    """Verify arbitrary text against a book's chunks (used for cross-synthesis)."""
    retrieval = Retrieval(config)
    llm = LLMClient(config.generation)

    # Retrieve chunks relevant to the claims in the text
    chunks = retrieval.search_book(book_id, text, top_k=config.retrieval.max_context_chunks)
    source_text = retrieval.format_chunks_for_prompt(chunks)

    return llm.generate(
        CRITIC_PROMPT.format(summary=text, sources=source_text),
        system=CRITIC_SYSTEM,
    )
