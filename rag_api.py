#!/usr/bin/env python3
"""Local HTTP API for RAG book navigation.

Thin wrapper over the existing navigation primitives.
Navigation endpoints are read-only and make no LLM calls.
Reading-state endpoints (GET/POST) write a small local JSON file.

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

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

_UI_HTML = (Path(__file__).parent / "rag_ui.html").read_text()

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

@app.get("/", include_in_schema=False)
def ui():
    """Mobile-friendly navigation UI."""
    return HTMLResponse(_UI_HTML)


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
    fmt: str = Query("text", description="'text' for plain output, 'structured' for JSON cards"),
):
    """Trace an idea through a book's summaries."""
    _require_book(book_id)
    if fmt == "structured":
        from rag.navigation import trace_idea_data
        return trace_idea_data(book_id, idea, _config, limit=limit)
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


@app.get("/books/{book_id}/ideas", tags=["navigation"])
def book_ideas(book_id: str):
    """Return generated + user-authored idea items for a book."""
    _require_book(book_id)
    from rag.inspect_utils import extract_book_ideas
    generated = extract_book_ideas(book_id, _config)
    for g in generated:
        g["source"] = "generated"
    user = _load_all_reading_state().get(book_id, {}).get("user_ideas", [])
    for u in user:
        u["source"] = "user"
    ideas = generated + user
    return {"book_id": book_id, "ideas": ideas, "total": len(ideas)}


@app.post("/books/{book_id}/ideas", tags=["navigation"])
def add_user_idea(
    book_id: str,
    type: str = Query(..., description="key_idea, example, or framework"),
    text: str = Query(..., description="Idea text"),
    section: str = Query("", description="Source section"),
):
    """Add a user-authored idea for a book."""
    if type not in ("key_idea", "example", "framework"):
        raise HTTPException(status_code=400, detail="type must be key_idea, example, or framework")
    all_state = _load_all_reading_state()
    state = all_state.setdefault(book_id, {})
    ideas = state.setdefault("user_ideas", [])
    next_id = max((i.get("id", -1) for i in ideas), default=-1) + 1
    idea = {"type": type, "text": text, "section": section, "id": next_id}
    ideas.append(idea)
    _save_all_reading_state(all_state)
    return {"book_id": book_id, "idea": idea, "total_user_ideas": len(ideas)}


@app.delete("/books/{book_id}/ideas/{idea_id}", tags=["navigation"])
def delete_user_idea(book_id: str, idea_id: int):
    """Delete a user-authored idea by its id."""
    all_state = _load_all_reading_state()
    state = all_state.get(book_id, {})
    ideas = state.get("user_ideas", [])
    idx = next((i for i, idea in enumerate(ideas) if idea.get("id") == idea_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="User idea not found")
    ideas.pop(idx)
    _save_all_reading_state(all_state)
    return {"book_id": book_id, "deleted": idea_id}


@app.get("/ideas/all", tags=["navigation"])
def all_book_ideas():
    """Return generated + user-authored ideas across all books."""
    from rag.inspect_utils import extract_book_ideas
    registry_path = Path(_config.vectorstore.persist_directory) / "books.json"
    if not registry_path.exists():
        return {"ideas": [], "total": 0}
    registry: dict = json.loads(registry_path.read_text())
    all_state = _load_all_reading_state()
    ideas: list[dict] = []
    for book_id, meta in registry.items():
        title = meta.get("title", book_id)
        results_dir = _results_dir(book_id)
        if not (results_dir / "chapter_insights.md").exists():
            continue
        generated = extract_book_ideas(book_id, _config)
        for g in generated:
            g.update({"source": "generated", "book_id": book_id, "title": title})
        user = all_state.get(book_id, {}).get("user_ideas", [])
        for u in user:
            u.update({"source": "user", "book_id": book_id, "title": title})
        ideas.extend(generated)
        ideas.extend(user)
    return {"ideas": ideas, "total": len(ideas)}


@app.post("/ideas/synthesize", tags=["navigation"])
async def synthesize_ideas(request: Request):
    """Synthesize connections between selected idea items using the LLM."""
    body = await request.json()
    ideas = body.get("ideas", [])
    if len(ideas) < 2:
        raise HTTPException(status_code=400, detail="Select at least 2 ideas to synthesize")
    if len(ideas) > 20:
        raise HTTPException(status_code=400, detail="Select at most 20 ideas")

    idea_lines = []
    for i, idea in enumerate(ideas, 1):
        src = f" [source: {idea.get('source', '?')}]"
        book = f" [book: {idea.get('title', idea.get('book_id', '?'))}]" if idea.get("book_id") else ""
        sec = f" (section: {idea.get('section', '?')})" if idea.get("section") else ""
        idea_lines.append(f"{i}. [{idea.get('type', '?')}]{book}{sec}{src}: {idea.get('text', '')}")
    ideas_block = "\n".join(idea_lines)

    prompt = f"""You are analyzing a curated set of ideas from one or more books. The ideas below were selected by the reader as important.

SELECTED IDEAS:
{ideas_block}

Provide a focused synthesis with these sections:

**Connections**: What key connections, themes, or patterns link these ideas? Be specific and cite the idea numbers.

**Tensions**: Are there any contradictions or tensions between the ideas? If none, say so briefly.

**Opportunities**: List 2-5 concrete product ideas, project opportunities, or actionable experiments that emerge from combining these ideas. Each should reference which ideas it draws from.

**Sources used**: List the idea numbers, types, and sections used in this synthesis.

Keep the output concise and grounded in the selected ideas. Do not introduce unrelated concepts."""

    from rag.config import load_config
    from rag.llm import LLMClient
    config = load_config("rag_config.yaml")
    llm = LLMClient(config.generation)

    try:
        result = llm.generate(prompt, system="You are a concise idea synthesizer. Ground every claim in the provided ideas.")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}")

    return {
        "synthesis": result,
        "sources_used": [
            {"type": idea.get("type"), "text": idea.get("text", "")[:100],
             "section": idea.get("section", ""), "source": idea.get("source", "?"),
             "book_id": idea.get("book_id", ""), "title": idea.get("title", "")}
            for idea in ideas
        ],
        "total_ideas": len(ideas),
    }


@app.get("/books/{book_id}/read-section-paragraphs", tags=["navigation"])
def read_section_paragraphs(
    book_id: str,
    section: str = Query(..., description="Section name"),
):
    """Return the section text as a list of paragraphs with overlap deduplicated."""
    _require_book(book_id)
    from rag.inspect_utils import read_section_paragraphs as _rsp
    return _rsp(book_id, section, _config)


@app.get("/books/{book_id}/section-progress", tags=["reading"])
def get_section_progress(
    book_id: str,
    section: str = Query(..., description="Section name"),
):
    """Return the last-read paragraph index for a section (0-based)."""
    state = _load_all_reading_state().get(book_id, {})
    progress = state.get("section_progress", {})
    return {"book_id": book_id, "section": section,
            "last_read_paragraph": progress.get(section, -1)}


@app.post("/books/{book_id}/section-progress", tags=["reading"])
def set_section_progress(
    book_id: str,
    section: str = Query(..., description="Section name"),
    paragraph: int = Query(..., description="0-based paragraph index"),
):
    """Set the last-read paragraph index for a section. Only advances forward."""
    all_state = _load_all_reading_state()
    state = all_state.setdefault(book_id, {})
    progress = state.setdefault("section_progress", {})
    current = progress.get(section, -1)
    if paragraph > current:
        progress[section] = paragraph
    state["last_reader_section"] = section
    _save_all_reading_state(all_state)
    return {"book_id": book_id, "section": section,
            "last_read_paragraph": progress[section]}


@app.get("/books/{book_id}/read-section", tags=["navigation"])
def read_section(
    book_id: str,
    section: str = Query(..., description="Section name"),
):
    """Return the full continuous text of a section with overlap deduplicated."""
    _require_book(book_id)
    from rag import inspect_utils
    output = _capture(inspect_utils.read_section, book_id, section, _config)
    return {"book_id": book_id, "section": section, "output": output}


@app.get("/books/{book_id}/sections", tags=["navigation"])
def list_sections(book_id: str):
    """List all summarised sections with lightweight metadata."""
    _require_book(book_id)
    meta_path = _results_dir(book_id) / "summary_meta.json"
    if not meta_path.exists():
        return {"book_id": book_id, "sections": []}
    meta = json.loads(meta_path.read_text())
    return {
        "book_id": book_id,
        "sections": [
            {"name": s["name"], "type": s.get("type", "?"), "pages": s.get("pages", "?")}
            for s in meta.get("sections", [])
        ],
    }


@app.get("/books/{book_id}/section-windows", tags=["navigation"])
def section_windows(
    book_id: str,
    section: str = Query(..., description="Section name"),
    all: bool = Query(False, description="If true, return all windows including non-selected ones"),
):
    """Return sorted 1-based window indices available for a section.

    By default returns only selected (summarised) windows. Pass all=true to
    include every window the section was split into, including those that were
    below the summarisation budget.
    """
    _require_book(book_id)
    if all:
        sel_path = _results_dir(book_id) / "selection_detail.json"
        if not sel_path.exists():
            return {"book_id": book_id, "section": section, "windows": [], "all": True}
        sel_data = json.loads(sel_path.read_text())
        raw = sel_data.get(section, [])
        ids = sorted(w["index"] + 1 for w in raw if isinstance(w.get("index"), int))
        return {"book_id": book_id, "section": section, "windows": ids, "all": True}
    windows_path = _results_dir(book_id) / "window_summaries.json"
    if not windows_path.exists():
        return {"book_id": book_id, "section": section, "windows": [], "all": False}
    window_data = json.loads(windows_path.read_text())
    raw = window_data.get(section, [])
    ids = sorted(w["window"] + 1 for w in raw if isinstance(w.get("window"), int))
    return {"book_id": book_id, "section": section, "windows": ids, "all": False}


# ---------------------------------------------------------------------------
# Reading state  (persisted to data/reading_state.json)
# ---------------------------------------------------------------------------

def _reading_state_path() -> Path:
    return Path(_config.storage.results_directory).parent / "reading_state.json"


def _load_all_reading_state() -> dict:
    p = _reading_state_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _save_all_reading_state(state: dict) -> None:
    p = _reading_state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


@app.get("/books/{book_id}/reading-state", tags=["reading"])
def get_reading_state(book_id: str):
    """Return reading progress for a book: read windows, last-read position, bookmarks, and saved items."""
    state = _load_all_reading_state().get(book_id, {})
    return {
        "book_id": book_id,
        "read_windows": state.get("read_windows", []),
        "last_read_window": state.get("last_read_window"),
        "last_read_section": state.get("last_read_section"),
        "last_reader_section": state.get("last_reader_section"),
        "bookmarks": state.get("bookmarks", []),
        "saved_items": state.get("saved_items", []),
    }


@app.post("/books/{book_id}/reading-state/mark", tags=["reading"])
def mark_window_read(
    book_id: str,
    window: int = Query(..., description="1-based window number"),
    section: str | None = Query(None, description="Section name"),
):
    """Mark a window as read and update last-read position. Returns updated state."""
    all_state = _load_all_reading_state()
    state = all_state.setdefault(
        book_id,
        {"read_windows": [], "last_read_window": None, "last_read_section": None},
    )
    if window not in state["read_windows"]:
        state["read_windows"].append(window)
        state["read_windows"].sort()
    state["last_read_window"] = window
    if section:
        state["last_read_section"] = section
    _save_all_reading_state(all_state)
    return {
        "book_id": book_id,
        "read_windows": state["read_windows"],
        "last_read_window": state["last_read_window"],
        "last_read_section": state.get("last_read_section"),
        "bookmarks": state.get("bookmarks", []),
    }


@app.post("/books/{book_id}/reading-state/bookmark", tags=["reading"])
def toggle_bookmark(
    book_id: str,
    window: int = Query(..., description="1-based window number"),
    section: str | None = Query(None, description="Section name"),
):
    """Toggle a bookmark for a window. Adds if absent, removes if present."""
    all_state = _load_all_reading_state()
    state = all_state.setdefault(
        book_id,
        {"read_windows": [], "last_read_window": None, "last_read_section": None, "bookmarks": []},
    )
    if "bookmarks" not in state:
        state["bookmarks"] = []

    existing = next((i for i, b in enumerate(state["bookmarks"]) if b["window"] == window), None)
    if existing is not None:
        state["bookmarks"].pop(existing)
        action = "removed"
    else:
        bm: dict = {"window": window}
        if section:
            bm["section"] = section
        state["bookmarks"].append(bm)
        action = "added"

    _save_all_reading_state(all_state)
    return {
        "book_id": book_id,
        "action": action,
        "bookmarks": state["bookmarks"],
        "read_windows": state.get("read_windows", []),
        "last_read_window": state.get("last_read_window"),
        "last_read_section": state.get("last_read_section"),
    }


@app.post("/books/{book_id}/reading-state/save", tags=["reading"])
def toggle_saved_item(
    book_id: str,
    window: int = Query(..., description="1-based window number"),
    section: str | None = Query(None, description="Section name"),
    preview: str | None = Query(None, description="Short text snippet from the window"),
):
    """Toggle a saved item for a window. Adds if absent, removes if present."""
    all_state = _load_all_reading_state()
    state = all_state.setdefault(
        book_id,
        {"read_windows": [], "last_read_window": None, "last_read_section": None,
         "bookmarks": [], "saved_items": []},
    )
    if "saved_items" not in state:
        state["saved_items"] = []

    existing = next(
        (i for i, s in enumerate(state["saved_items"]) if s["window"] == window), None
    )
    if existing is not None:
        state["saved_items"].pop(existing)
        action = "removed"
    else:
        item: dict = {"window": window}
        if section:
            item["section"] = section
        if preview:
            item["preview"] = preview[:200]
        registry_path = Path(_config.vectorstore.persist_directory) / "books.json"
        if registry_path.exists():
            try:
                registry = json.loads(registry_path.read_text())
                title = registry.get(book_id, {}).get("title")
                if title:
                    item["title"] = title
            except Exception:
                pass
        state["saved_items"].append(item)
        action = "added"

    _save_all_reading_state(all_state)
    return {
        "book_id": book_id,
        "action": action,
        "saved_items": state["saved_items"],
        "read_windows": state.get("read_windows", []),
        "last_read_window": state.get("last_read_window"),
        "last_read_section": state.get("last_read_section"),
        "bookmarks": state.get("bookmarks", []),
    }


@app.get("/saved-items", tags=["reading"])
def all_saved_items():
    """Return all saved items across all books, for cross-book idea collection."""
    all_state = _load_all_reading_state()
    result = []
    for bid, state in all_state.items():
        for item in state.get("saved_items", []):
            result.append({"book_id": bid, **item})
    return {"saved_items": result, "total": len(result)}


@app.get("/books/{book_id}/fiction/available", tags=["fiction"])
def fiction_available(book_id: str):
    """Return whether fiction extraction artifacts exist for this book."""
    state_path = _results_dir(book_id) / "fiction_state.json"
    if not state_path.exists():
        return {"book_id": book_id, "available": False}
    import json as _json
    try:
        state = _json.loads(state_path.read_text())
        available = len(state) >= 10
    except Exception:
        available = False
    return {"book_id": book_id, "available": available}


@app.get("/books/{book_id}/fiction/whois", tags=["fiction"])
def fiction_whois(
    book_id: str,
    chapter: int = Query(..., description="Chapter number (1-based)"),
    character: str = Query(..., description="Character name"),
):
    """Character profile up to chapter N (spoiler-safe)."""
    _require_book(book_id)
    from rag.fiction import fiction_whois as _whois
    output = _capture(_whois, book_id, _config, chapter_number=chapter, character_name=character)
    return {"book_id": book_id, "output": output}


@app.get("/books/{book_id}/fiction/diff", tags=["fiction"])
def fiction_diff(
    book_id: str,
    from_chapter: int = Query(..., description="Start chapter (1-based)"),
    to_chapter: int = Query(..., description="End chapter (1-based)"),
):
    """What changed between two reading points (spoiler-safe)."""
    _require_book(book_id)
    from rag.fiction import fiction_diff as _diff
    output = _capture(_diff, book_id, _config, from_chapter=from_chapter, to_chapter=to_chapter)
    return {"book_id": book_id, "output": output}


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
