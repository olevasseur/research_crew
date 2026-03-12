"""Critic / verifier: checks that claims are grounded in source subchunks."""

from __future__ import annotations

import json
from pathlib import Path

from .config import RAGConfig
from .llm import LLMClient
from .retrieval import Retrieval

CRITIC_SYSTEM = """\
You are a verification agent. Your job is to check that every specific claim
in a summary is supported by the source excerpts provided.
Rules:
- Be precise: verify individual claims, not vague impressions.
- A claim is SUPPORTED if a source excerpt contains the same fact, name, or argument.
- A claim is UNSUPPORTED if no source excerpt contains evidence for it.
- A claim is PARTIALLY SUPPORTED if the gist is right but specific details are missing.
- Cite the chunk IDs that support each claim.
- Do not generate generic criticism that would apply to any book summary."""

CRITIC_PROMPT = """\
Below is a book summary, followed by numbered source excerpts from the actual book.

=== SUMMARY ===
{summary}

=== SOURCE EXCERPTS ===
{sources}

For each major factual claim in the summary:
1. State the claim briefly.
2. Verdict: SUPPORTED / PARTIALLY SUPPORTED / UNSUPPORTED
3. Supporting chunk ID(s) if applicable, or reason if unsupported.

After checking individual claims, provide:

## Verification Summary
- **Total claims checked**: N
- **Supported**: N (with chunk IDs)
- **Partially supported**: N
- **Unsupported**: N
- **Grounding score**: HIGH (>80% supported) / MEDIUM (50-80%) / LOW (<50%)
- **Most reliable sections**: list sections with strongest grounding
- **Weakest sections**: list sections with most unsupported claims
"""


def verify_book_summary(book_id: str, config: RAGConfig) -> str:
    """Verify a book summary against its source subchunks."""
    retrieval = Retrieval(config)
    llm = LLMClient(config.generation)

    summary_path = Path(config.storage.results_directory) / book_id / "book_summary.md"
    if not summary_path.exists():
        raise FileNotFoundError(f"No summary for '{book_id}'. Run summarize first.")
    summary = summary_path.read_text()

    # Retrieve source chunks — use semantic search against the summary's claims
    # to find the most relevant subchunks, rather than just grabbing first N per section
    print("  Retrieving relevant source chunks …")
    chapters = retrieval.get_book_chapters(book_id)
    source_chunks: list[dict] = []

    # Semantic retrieval: search for chunks relevant to the summary
    top_results = retrieval.search_book(
        book_id, summary[:2000], top_k=config.retrieval.max_context_chunks
    )
    source_chunks.extend(top_results)

    # Also ensure per-section coverage
    seen_ids = {c["id"] for c in source_chunks}
    for ch in chapters:
        ch_chunks = retrieval.get_chapter_chunks(book_id, ch)
        for c in ch_chunks[:2]:
            if c["id"] not in seen_ids:
                source_chunks.append(c)
                seen_ids.add(c["id"])

    source_chunks = source_chunks[:config.retrieval.max_context_chunks]
    print(f"  Using {len(source_chunks)} source chunks for verification")

    source_text = retrieval.format_chunks_for_prompt(source_chunks)

    result = llm.generate(
        CRITIC_PROMPT.format(summary=summary, sources=source_text),
        system=CRITIC_SYSTEM,
    )

    results_dir = Path(config.storage.results_directory) / book_id
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "verification.md").write_text(result)

    # Also save which chunk IDs were used for verification
    verification_meta = {
        "chunk_ids_used": [c["id"] for c in source_chunks],
        "total_chunks": len(source_chunks),
    }
    (results_dir / "verification_meta.json").write_text(json.dumps(verification_meta, indent=2))

    print(f"  Verification saved to {results_dir / 'verification.md'}")
    return result


def verify_text(text: str, book_id: str, config: RAGConfig) -> str:
    """Verify arbitrary text against a book's subchunks."""
    retrieval = Retrieval(config)
    llm = LLMClient(config.generation)

    chunks = retrieval.search_book(book_id, text, top_k=config.retrieval.max_context_chunks)
    source_text = retrieval.format_chunks_for_prompt(chunks)

    return llm.generate(
        CRITIC_PROMPT.format(summary=text, sources=source_text),
        system=CRITIC_SYSTEM,
    )
