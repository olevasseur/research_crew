# CLAUDE.md — research_crew

## What this is

Local-first book navigation and idea cross-pollination tool. EPUB/PDF ingestion → chunk → summarize → navigate/read via CLI and web UI. No cloud, no auth, no build step.

**Main components:**
- `rag_cli.py` — CLI for ingest, summarize, trace, explore, fiction, inspect
- `rag_api.py` — FastAPI server serving the HTML UI + REST endpoints
- `rag_ui.html` — single-file HTML/CSS/JS UI (no framework, no build)
- `rag/` — core pipeline modules (chunker, analysis, navigation, inspect_utils, etc.)
- `rag_config.yaml` — all pipeline config (embedding, chunking, summarization, retrieval)

## Environment

```bash
# Virtual environment — always use this Python
/Users/aiagent/venv/bin/python

# Short alias for commands
PYTHON=/Users/aiagent/venv/bin/python

# Ollama must be running for LLM/embedding calls
# Models: llama3.1 (generation), nomic-embed-text (embedding)
```

**pytest is NOT installed** in the venv. Tests exist in `tests/` but cannot currently be run via `pytest`. Validate with direct Python scripts and curl instead.

## Running the app

```bash
# Start the server (default: 127.0.0.1:8000)
/Users/aiagent/venv/bin/python rag_api.py

# For Tailscale/mobile access
/Users/aiagent/venv/bin/python rag_api.py --host 0.0.0.0 --port 8000

# Dev mode with auto-reload
/Users/aiagent/venv/bin/python rag_api.py --reload
```

## Critical gotcha: HTML is loaded at import time

```python
_UI_HTML = (Path(__file__).parent / "rag_ui.html").read_text()  # line 25
```

**The server reads `rag_ui.html` once at startup.** After editing the HTML, you MUST restart the server to see changes. Browser cache can also serve stale content — hard refresh with Cmd+Shift+R.

## Before making changes — checklist

1. **Kill stale servers** before any UI/API validation:
   ```bash
   kill $(lsof -ti:8000) 2>/dev/null
   ```
   Multiple processes CAN bind the same port (127.0.0.1 vs 0.0.0.0) — always check with `lsof -i :8000 -P`.

2. **Use the correct Python**: `/Users/aiagent/venv/bin/python` — system python3 does not have uvicorn/fastapi.

3. **Verify CLI flags** before running `rag_cli.py` commands:
   ```bash
   /Users/aiagent/venv/bin/python rag_cli.py <subcommand> --help
   ```

4. **Syntax check** Python files after editing:
   ```bash
   python3 -c "import ast; ast.parse(open('rag_api.py').read()); print('OK')"
   ```

5. **Write validation output** to `/tmp/<feature>_*.txt` files, not inline.

6. **Restart server and hard-refresh browser** before declaring UI changes work.

## Key directories and files

```
rag_api.py              → FastAPI app, all endpoints
rag_ui.html             → entire UI (single file, no build)
rag_cli.py              → CLI entry point
rag_config.yaml         → pipeline configuration
rag/
  inspect_utils.py      → inspect, read_section, extract_book_ideas, etc.
  analysis.py           → window scoring, MMR selection, build_windows
  chunker.py            → text splitting with sentence-aware boundaries
  navigation.py         → trace_idea, explore_section
  fiction.py            → fiction-whois, fiction-diff
  config.py             → RAGConfig dataclass, load_config()
  ingest.py             → EPUB/PDF ingestion
data/
  vectorstore/          → ChromaDB + books.json registry
  cache/<book-id>/      → cached window/section summaries
  results/<book-id>/    → window_summaries.json, selection_detail.json,
                          summary_meta.json, chapter_insights.md, etc.
  reading_state.json    → bookmarks, saved items, section progress
```

## API endpoints (port 8000)

| Path | Method | Purpose |
|------|--------|---------|
| `/` | GET | HTML UI |
| `/health` | GET | Liveness check |
| `/books` | GET | List ingested books |
| `/books/{id}/sections` | GET | Section list |
| `/books/{id}/section-windows` | GET | Window indices (`?all=true` for non-selected) |
| `/books/{id}/trace` | GET | Trace idea through summaries |
| `/books/{id}/explore` | GET | Explore a section |
| `/books/{id}/inspect-window` | GET | Full window detail |
| `/books/{id}/ideas` | GET | Generated ideas from summaries |
| `/books/{id}/read-section` | GET | Full section text (plain) |
| `/books/{id}/read-section-paragraphs` | GET | Section as paragraph list |
| `/books/{id}/section-progress` | GET/POST | Paragraph read progress |
| `/books/{id}/reading-state` | GET | Read windows, bookmarks, saved items |
| `/books/{id}/reading-state/mark` | POST | Mark window read |
| `/books/{id}/reading-state/bookmark` | POST | Toggle bookmark |
| `/books/{id}/reading-state/save` | POST | Toggle saved item |
| `/saved-items` | GET | Cross-book saved items |

## Common CLI commands

```bash
PYTHON=/Users/aiagent/venv/bin/python

# Ingest a book
$PYTHON rag_cli.py ingest path/to/book.epub --book-id my-book

# Summarize
$PYTHON rag_cli.py summarize digital-minimalism --quality default

# Navigation
$PYTHON rag_cli.py trace digital-minimalism --idea "solitude"
$PYTHON rag_cli.py explore digital-minimalism --section "Chapter 4: Spend Time Alone"
$PYTHON rag_cli.py inspect-window digital-minimalism --window 3 --section "Introduction"

# Full validation suite (runs summarize + evaluate + inspect)
bash scripts/validate_rag.sh digital-minimalism
```

## Ingested books

- `digital-minimalism` — Digital Minimalism by Cal Newport (richest summaries)
- `pale-criminal` — The Pale Criminal by Philip Kerr (fiction)
- `the-clockwork-garden` — The Clockwork Garden by Ada Marchetti
- `thinking-in-systems` — Thinking in Systems by Donella H. Meadows

## State persistence

All user state is in `data/reading_state.json`:
```json
{
  "<book_id>": {
    "read_windows": [1, 2, 3],
    "last_read_window": 3,
    "last_read_section": "Chapter 1: ...",
    "bookmarks": [{"window": 5, "section": "..."}],
    "saved_items": [{"window": 7, "section": "...", "preview": "..."}],
    "section_progress": {"Chapter 2: ...": 42}
  }
}
```

Clean up test entries after validation runs — don't leave `test-*` book IDs in the file.

## Common pitfalls

1. **Stale server process**: The #1 source of "my changes aren't showing." Always `lsof -i :8000 -P` and kill ALL matching processes before restarting. Two processes can coexist on the same port if one binds `127.0.0.1` and the other `0.0.0.0`.

2. **HTML read at import time**: `_UI_HTML` is set once at module load. No amount of browser refreshing helps if the server is still running with old code. Must restart the Python process.

3. **Wrong Python**: System `python3` (3.14) has no project deps. Always use `/Users/aiagent/venv/bin/python` (3.12).

4. **Section names are exact-match**: The `section-windows` and `section-progress` endpoints do exact key matching against `selection_detail.json` / `window_summaries.json`. Partial names return empty results silently.

5. **Chunk overlap**: Adjacent chunks share ~200 chars of text. Windows share ~500 chars of chunks. Use `_find_leading_overlap()` in `inspect_utils.py` for display-time dedup. Never change stored artifacts.

6. **Config version**: If summarization prompts change, bump `PROMPT_VERSION` (currently `v4`) in the relevant module to invalidate cache.

## Incremental workflow for UI+backend changes

1. Edit `rag_api.py` / `rag/*.py` / `rag_ui.html`
2. Syntax check: `python3 -c "import ast; ast.parse(open('file.py').read())"`
3. Kill server: `kill $(lsof -ti:8000) 2>/dev/null`
4. Start fresh: `/Users/aiagent/venv/bin/python rag_api.py --host 0.0.0.0 --port 8000 &`
5. Wait 2 seconds, then validate with curl
6. Write validation to `/tmp/<feature>_checks.txt`
7. Clean up test data from `reading_state.json` if you added any
