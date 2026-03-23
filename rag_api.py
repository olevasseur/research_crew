#!/usr/bin/env python3
"""Local HTTP API for RAG book navigation.

Thin wrapper over the existing navigation primitives.
All endpoints are read-only and make no LLM calls.

Run with:
    python rag_api.py
    python rag_api.py --port 8001
    uvicorn rag_api:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from rag.config import load_config

# ---------------------------------------------------------------------------
# App + config
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Book Navigation",
    description="Local navigation API: trace ideas, explore sections, inspect windows.",
    version="0.1.0",
)

_config = load_config("rag_config.yaml")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _results_dir(book_id: str) -> Path:
    return Path(_config.storage.results_directory) / book_id


def _require_book(book_id: str) -> None:
    """Raise HTTP 404 when a book has no summary artifacts yet."""
    if not (_results_dir(book_id) / "chapter_insights.md").exists():
        raise HTTPException(
            status_code=404,
            detail=f"Book '{book_id}' not found. Run 'summarize' first.",
        )


def _capture(fn, *args, **kwargs) -> str:
    """Call fn and return whatever it printed to stdout as a string.

    Translates ValueError (raised by ambiguous section matching) into HTTP 400.
    """
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            fn(*args, **kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
def health():
    """Liveness check."""
    return {"ok": True}


@app.get("/books", tags=["meta"])
def list_books():
    """Return all ingested books with light metadata."""
    registry_path = Path(_config.vectorstore.persist_directory) / "books.json"
    if not registry_path.exists():
        return {"books": []}

    registry: dict = json.loads(registry_path.read_text())
    books = []
    for book_id, meta in registry.items():
        books.append({
            "book_id": book_id,
            "title": meta.get("title"),
            "author": meta.get("author"),
            "total_pages": meta.get("total_pages"),
            "total_chunks": meta.get("total_chunks"),
        })
    return {"books": books}


@app.get("/books/{book_id}/trace", tags=["navigation"])
def trace(
    book_id: str,
    idea: str = Query(..., description="Idea or concept to trace through summaries"),
    limit: int = Query(20, description="Max matching sections to return"),
    show: str = Query("both", description="'both', 'sections', or 'windows'"),
):
    """Trace an idea through a book's summaries."""
    _require_book(book_id)
    from rag.navigation import trace_idea
    output = _capture(trace_idea, book_id, idea, _config, limit=limit, show=show)
    return {"book_id": book_id, "idea": idea, "output": output}


@app.get("/books/{book_id}/explore", tags=["navigation"])
def explore(
    book_id: str,
    section: str = Query(..., description="Section name (exact, normalized, or unambiguous partial)"),
    windows: int = Query(3, description="Max selected windows to display (0 = all)"),
    show: str = Query("all", description="'all', 'summary', or 'windows'"),
):
    """Explore a section: metadata, core argument, examples, frameworks, top windows."""
    _require_book(book_id)
    from rag.navigation import explore_section
    output = _capture(explore_section, book_id, section, _config,
                      show=show, show_windows=windows)
    return {"book_id": book_id, "section": section, "output": output}


@app.get("/books/{book_id}/inspect-window", tags=["navigation"])
def inspect_window(
    book_id: str,
    window: int = Query(..., description="1-based window index"),
    section: str | None = Query(None, description="Section name to disambiguate"),
):
    """Zoom into a specific window: full summary + original passage text."""
    _require_book(book_id)
    from rag import inspect_utils
    output = _capture(inspect_utils.inspect_window, book_id, window, _config,
                      section=section)
    return {"book_id": book_id, "window": window, "section": section, "output": output}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="RAG navigation API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true",
                        help="Auto-reload on code changes (dev mode)")
    args = parser.parse_args()

    uvicorn.run("rag_api:app", host=args.host, port=args.port, reload=args.reload)
