# Project Memory

Stable facts about this project. Edit manually when fundamentals change.
The orchestrator reads this on every planning call.

## Purpose

Local book analysis tool. Two pipelines:
1. **RAG pipeline** (`rag_cli.py` + `rag/`) — primary, active development.
   Ingest → chunk → embed → select windows → summarize → evaluate → navigate → cross-synthesize.
2. **CrewAI pipeline** (`main.py` + `crews/book_summary/`) — simpler, multi-agent, may be legacy.

## Architecture

| Module | Role |
|---|---|
| `rag/ingest.py` | Load PDF/txt/md, clean text, detect structure, chunk, embed, register |
| `rag/chunker.py` | Page-level state machine → sections; overlapping chunk generation |
| `rag/analysis.py` | Windowed summarization with MMR content-diversity selection |
| `rag/cache.py` | File-backed content-addressed cache; keyed by PROMPT_VERSION + model + quality |
| `rag/retrieval.py` | ChromaDB vector search |
| `rag/evaluate.py` | Quality assessment of summaries against source chunks |
| `rag/navigation.py` | trace (idea → sections) and explore (section detail) |
| `rag/synthesis.py` | Cross-book comparison and Q&A |
| `rag_cli.py` | Main CLI entry point |
| `rag_config.yaml` | All tunable settings |

## Key invariants

- Fully local: Ollama for LLM + embeddings; ChromaDB for vectors. No external API calls.
- Data layout: `./data/cache/<book_id>/`, `./data/results/<book_id>/`, `./data/vectorstore/`
- Cache invalidation: bump `PROMPT_VERSION` in `rag/cache.py` (currently `"v4"`).
- Section types summarizable: `{introduction, chapter, conclusion, epilogue, appendix}`
- Section types skippable: `{front_matter, toc, acknowledgments, notes, index, about_author, unknown}`
- Config defaults live in `rag/config.py` dataclasses; overrides in `rag_config.yaml`.

## Constraints

- Ollama must be running (`ollama serve`) before any ingest/summarize/ask commands.
- Embedding model: `nomic-embed-text`. Generation model: `llama3.1`.
- Python 3.13, venv at `.venv/`. No package manager — plain pip.
- Currently ingested: `digital-minimalism` (test book).

## Key file paths

- `rag_config.yaml` — tuning knobs (chunking, window size, budget, MMR lambda)
- `rag/cache.py:PROMPT_VERSION` — bump when prompts change to invalidate stale cache
- `data/results/<book_id>/book_summary.md` — final output per book
- `data/results/<book_id>/evaluations/` — per-section and book-level eval reports
- `scripts/validate_rag.sh` — smoke-test commands
