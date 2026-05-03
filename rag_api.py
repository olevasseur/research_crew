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


def _idea_note_key(idea: dict) -> str:
    """Compute a stable key for attaching notes to an idea."""
    if idea.get("source") == "user" and idea.get("id") is not None:
        return f"user:{idea['id']}"
    import hashlib
    h = hashlib.md5(idea.get("text", "").encode()).hexdigest()[:8]
    return f"gen:{idea.get('type', '')}:{idea.get('section', '')}:{h}"


def _enrich_ideas_with_notes(ideas: list[dict], book_id: str) -> None:
    """Stamp note_key and note (if any) onto each idea in-place."""
    notes = _load_all_reading_state().get(book_id, {}).get("idea_notes", {})
    for idea in ideas:
        key = _idea_note_key(idea)
        idea["note_key"] = key
        note = notes.get(key, "")
        if note:
            idea["note"] = note


def _enrich_ideas_with_curation(ideas: list[dict], book_id: str) -> None:
    """Stamp pinned/hidden/near_duplicate flags onto each idea from curation state."""
    from rag.curation import enrich_idea_data_with_curation_signals
    for idea in ideas:
        if not idea.get("note_key"):
            idea["note_key"] = _idea_note_key(idea)
    enrich_idea_data_with_curation_signals(ideas, book_id)


_LINKED_EXAMPLE_FIELDS = ("text", "section", "window", "type", "source", "note_key", "match_score")


def _enrich_ideas_with_linked_examples(ideas: list[dict]) -> None:
    """Attach a small list of matched examples to each non-example idea."""
    from rag.idea_linker import link_ideas_to_examples, idea_key
    targets = [i for i in ideas if i.get("type") != "example"]
    examples = [i for i in ideas if i.get("type") == "example"]
    links = link_ideas_to_examples(targets, examples, top_k=3)
    for idea in targets:
        matched = links.get(idea_key(idea), [])
        idea["linked_examples"] = [
            {k: ex.get(k) for k in _LINKED_EXAMPLE_FIELDS if k in ex}
            for ex in matched
        ]


def _load_book_ideas(book_id: str) -> list[dict]:
    """Return generated + user-authored ideas with shared ranking signals."""
    from rag.inspect_utils import extract_book_ideas, score_book_ideas
    generated = extract_book_ideas(book_id, _config)
    for g in generated:
        g["source"] = "generated"
    user = _load_all_reading_state().get(book_id, {}).get("user_ideas", [])
    for u in user:
        u["source"] = "user"
    ideas = generated + user
    _enrich_ideas_with_notes(ideas, book_id)
    _enrich_ideas_with_curation(ideas, book_id)
    score_book_ideas(ideas, book_id, _config)
    return ideas


@app.get("/books/{book_id}/ideas", tags=["navigation"])
def book_ideas(book_id: str):
    """Return generated + user-authored idea items for a book, ranked."""
    _require_book(book_id)
    ideas = _load_book_ideas(book_id)
    _enrich_ideas_with_linked_examples(ideas)
    return {"book_id": book_id, "ideas": ideas, "total": len(ideas)}


@app.get("/books/{book_id}/examples", tags=["navigation"])
def book_examples(book_id: str):
    """Return examples ranked by the strength of associated ideas."""
    _require_book(book_id)
    from rag.idea_linker import rank_examples_by_idea_links

    ideas = _load_book_ideas(book_id)
    ranked = rank_examples_by_idea_links(
        [i for i in ideas if i.get("type") != "example"],
        [i for i in ideas if i.get("type") == "example"],
    )
    examples = []
    for ex in ranked:
        examples.append({
            "example_key": ex.get("example_key"),
            "text": ex.get("text"),
            "section": ex.get("section"),
            "window": ex.get("window"),
            "rank_score": ex.get("example_score"),
            "example_score": ex.get("example_score"),
            "example_score_parts": ex.get("example_score_parts"),
            "associated_ideas": ex.get("associated_ideas", []),
            "source": ex.get("source"),
            "source_fields": {
                "type": ex.get("type"),
                "note_key": ex.get("note_key"),
                "pinned": ex.get("pinned"),
                "hidden": ex.get("hidden"),
                "near_duplicate": ex.get("near_duplicate"),
            },
        })
    return {"book_id": book_id, "examples": examples, "total": len(examples)}


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


@app.put("/books/{book_id}/idea-notes/{note_key:path}", tags=["navigation"])
def put_idea_note(
    book_id: str,
    note_key: str,
    text: str = Query(..., description="Note text (empty to clear)"),
):
    """Set or clear a note on an idea."""
    all_state = _load_all_reading_state()
    state = all_state.setdefault(book_id, {})
    notes = state.setdefault("idea_notes", {})
    if text.strip():
        notes[note_key] = text.strip()
    else:
        notes.pop(note_key, None)
    _save_all_reading_state(all_state)
    return {"book_id": book_id, "note_key": note_key, "note": notes.get(note_key, "")}


@app.get("/books/{book_id}/curation", tags=["navigation"])
def get_book_curation(book_id: str):
    """Return the curation state (pinned/hidden flags per idea) for a book."""
    from rag.curation import load_book_curation
    return {"book_id": book_id, "curation": load_book_curation(book_id)}


@app.post("/books/{book_id}/curation", tags=["navigation"])
def post_book_curation(
    book_id: str,
    idea_id: str = Query(..., description="Stable idea identifier (note_key)"),
    action: str = Query(..., description="pin, unpin, hide, or unhide"),
):
    """Apply a curation action (pin/unpin/hide/unhide) to an idea."""
    from rag.curation import store_curation_state
    try:
        updated = store_curation_state(book_id, idea_id, action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "book_id": book_id,
        "idea_id": idea_id,
        "action": action,
        "curation": updated,
    }


@app.post("/books/{book_id}/ask", tags=["navigation"])
async def ask_book(book_id: str, request: Request):
    """Answer a question about a book using relevant ideas, notes, and saved passages."""
    _require_book(book_id)
    body = await request.json()
    question = (body.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    from rag.inspect_utils import (
        extract_book_ideas, score_book_ideas,
        rank_ideas_for_query, rank_passages_for_query,
    )

    # Build scored idea pool (includes note text in matching)
    book_state = _load_all_reading_state().get(book_id, {})
    generated = extract_book_ideas(book_id, _config)
    for g in generated:
        g["source"] = "generated"
    user = book_state.get("user_ideas", [])
    for u in user:
        u["source"] = "user"
    ideas = generated + user
    _enrich_ideas_with_notes(ideas, book_id)
    _enrich_ideas_with_curation(ideas, book_id)
    score_book_ideas(ideas, book_id, _config)

    # Retrieve top relevant ideas. Hidden ideas are excluded by default;
    # callers can explicitly opt in via include_hidden=true on the request body.
    top_k = min(int(body.get("top_k", 10)), 20)
    include_hidden = bool(body.get("include_hidden", False))
    selected_ideas = rank_ideas_for_query(
        ideas, question, top_k=top_k, include_hidden=include_hidden
    )

    # Retrieve relevant saved passages
    passages = book_state.get("saved_passages", [])
    selected_passages = rank_passages_for_query(passages, question, top_k=3)

    # Format mixed evidence for prompt
    evidence_lines = []
    idx = 1
    for idea in selected_ideas:
        sec = f" (section: {idea.get('section', '?')})" if idea.get("section") else ""
        typ = idea.get("type", "?")
        src = "your idea" if idea.get("source") == "user" else "idea"
        note_suffix = f" [your note: {idea['note']}]" if idea.get("note") else ""
        evidence_lines.append(f"{idx}. [{src}/{typ}]{sec}: {idea.get('text', '')}{note_suffix}")
        idx += 1
    for p in selected_passages:
        sec = f" (section: {p.get('section', '?')})" if p.get("section") else ""
        evidence_lines.append(f"{idx}. [saved passage]{sec}: {p.get('preview', '')}")
        idx += 1
    evidence_block = "\n".join(evidence_lines)

    total_evidence = len(selected_ideas) + len(selected_passages)
    total_candidates = len(ideas) + len(passages)

    prompt = f"""You are answering a question about a book using only the evidence below. The evidence includes book ideas (some with the reader's personal notes) and saved passages that the reader highlighted.

QUESTION:
{question}

EVIDENCE ({total_evidence} items, ranked by relevance):
{evidence_block}

Instructions:
- Answer the question using only the evidence provided above.
- Cite source numbers (e.g. [1], [3]) when drawing on specific items.
- Items marked [your idea] or [your note] or [saved passage] reflect the reader's own annotations — give them appropriate weight.
- If the evidence is insufficient to fully answer the question, say so clearly.
- Be concise and direct.
- End with a "Sources used:" line listing the source numbers you referenced."""

    from rag.llm import LLMClient
    llm = LLMClient(_config.generation)
    try:
        answer = llm.generate(
            prompt,
            system="You are a concise book analyst. Answer only from the provided evidence. Cite sources.",
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}")

    # Build sources_used with provenance
    sources_used = []
    for idea in selected_ideas:
        sources_used.append({
            "type": idea.get("type"),
            "text": idea.get("text", "")[:120],
            "section": idea.get("section", ""),
            "source": idea.get("source", "?"),
            "has_note": bool(idea.get("note")),
            "pinned": bool(idea.get("pinned")),
            "query_score": idea.get("query_score", 0),
        })
    for p in selected_passages:
        sources_used.append({
            "type": "passage",
            "text": p.get("preview", "")[:120],
            "section": p.get("section", ""),
            "source": "saved_passage",
            "has_note": False,
            "query_score": p.get("query_score", 0),
        })

    return {
        "book_id": book_id,
        "question": question,
        "answer": answer,
        "sources_used": sources_used,
        "total_candidates": total_candidates,
        "total_sources_used": total_evidence,
    }


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
        book_ideas = generated + user
        _enrich_ideas_with_notes(book_ideas, book_id)
        ideas.extend(book_ideas)
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
        "saved_passages": state.get("saved_passages", []),
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


@app.post("/books/{book_id}/reading-state/save-passage", tags=["reading"])
def toggle_saved_passage(
    book_id: str,
    section: str = Query(..., description="Section name"),
    paragraph: int = Query(..., description="0-based paragraph index"),
    preview: str = Query("", description="Short text from the paragraph"),
):
    """Toggle a saved passage. Adds if absent, removes if present."""
    all_state = _load_all_reading_state()
    state = all_state.setdefault(book_id, {})
    passages = state.setdefault("saved_passages", [])

    existing = next(
        (i for i, p in enumerate(passages)
         if p["section"] == section and p["paragraph_index"] == paragraph),
        None,
    )
    if existing is not None:
        passages.pop(existing)
        action = "removed"
    else:
        entry: dict = {
            "section": section,
            "paragraph_index": paragraph,
            "preview": preview[:200] if preview else "",
        }
        registry_path = Path(_config.vectorstore.persist_directory) / "books.json"
        if registry_path.exists():
            try:
                registry = json.loads(registry_path.read_text())
                title = registry.get(book_id, {}).get("title")
                if title:
                    entry["title"] = title
            except Exception:
                pass
        passages.append(entry)
        action = "added"

    _save_all_reading_state(all_state)
    return {"book_id": book_id, "action": action, "saved_passages": passages}


@app.get("/saved-passages", tags=["reading"])
def all_saved_passages():
    """Return all saved passages across all books."""
    all_state = _load_all_reading_state()
    result = []
    for bid, state in all_state.items():
        for p in state.get("saved_passages", []):
            result.append({"book_id": bid, **p})
    return {"saved_passages": result, "total": len(result)}


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
