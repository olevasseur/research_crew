# Working Memory

Rolling context. Updated after each iteration. Refresh when it grows stale.

## Current goal

Develop and harden the RAG pipeline. The CrewAI pipeline is not the active focus.

## Recent progress (iterations 8–11)

- **iter 8**: Major `analysis.py` refactor — content-diversity window selection via MMR,
  `mmr_lambda` config knob, `selection_strategy` setting.
- **iter 9**: Added `rag/evaluate.py` — evaluates summary quality against source chunks;
  `rag_cli.py evaluate` command; results land in `data/results/<book_id>/evaluations/`.
- **iter 10**: Added `rag/navigation.py` with `trace` (idea → matching sections) and
  `explore` (single section detail); added `scripts/validate_rag.sh`.
- **iter 11**: Tested and fixed `navigation.py` end-to-end on `digital-minimalism`.
  Three bugs fixed: (1) `_compact_preview` produced empty `...` when first paragraph
  exceeded 400 chars; (2) `trace` snippets exposed raw `## heading` / `**Type:**`
  metadata lines; (3) `explore` Source section dumped all 34 chunk IDs on one line.
  Both commands produce clean, readable output. No regressions.

Test book throughout: `digital-minimalism`. Summary + evaluations exist for it.

## Active assumptions

- RAG pipeline is primary; CrewAI pipeline (`main.py` / `crews/`) is not being extended.
- `llama3.1` via Ollama is good enough for summarization quality at this scale.
- Summary quality has not been formally assessed beyond the one eval in `evaluations/`.
- Cache is at `PROMPT_VERSION = "v4"` — if prompts change next iteration, bump it.

## Open questions

- Is summary quality actually good on `digital-minimalism`? Eval exists but hasn't been reviewed.
- Is `navigation.py` (`trace`/`explore`) working end-to-end? ✓ Resolved in iter 11.
- Is the CrewAI pipeline being kept at all, or is it dead code?

## Next likely steps

- Review eval output for `digital-minimalism` and decide if quality is acceptable.
- ~~Test `trace` and `explore` commands end-to-end.~~ Done in iter 11.
- Add a second book to test multi-book `compare` and `ask` commands.
- Or: improve summary prompts based on eval feedback (would require PROMPT_VERSION bump).
