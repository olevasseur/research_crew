"""Curation state persistence for idea pin/hide actions."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from threading import Lock
from typing import Literal

CurationAction = Literal["pin", "unpin", "hide", "unhide"]

_STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "curation_state.json"
_LOCK = Lock()


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(path)


def load_curation_state(path: Path | None = None) -> dict:
    """Return the full curation state mapping book_id -> idea_id -> flags."""
    with _LOCK:
        return _load(path or _STATE_PATH)


def load_book_curation(book_id: str, path: Path | None = None) -> dict:
    """Return the per-idea curation mapping for a single book."""
    return load_curation_state(path).get(book_id, {})


def store_curation_state(
    book_id: str,
    idea_id: str,
    action: CurationAction,
    *,
    path: Path | None = None,
) -> dict:
    """Update curation state for (book_id, idea_id) with action and persist.

    Returns the updated per-book curation mapping.
    """
    if not book_id or not idea_id:
        raise ValueError("book_id and idea_id are required")
    if action not in ("pin", "unpin", "hide", "unhide"):
        raise ValueError(f"invalid action: {action}")

    target = path or _STATE_PATH
    with _LOCK:
        state = _load(target)
        book_state = state.setdefault(book_id, {})
        entry = book_state.setdefault(idea_id, {"pinned": False, "hidden": False})

        if action == "pin":
            entry["pinned"] = True
        elif action == "unpin":
            entry["pinned"] = False
        elif action == "hide":
            entry["hidden"] = True
        elif action == "unhide":
            entry["hidden"] = False

        if not entry["pinned"] and not entry["hidden"]:
            del book_state[idea_id]
            if not book_state:
                del state[book_id]

        _save(target, state)
        return state.get(book_id, {})


def _default_idea_key(idea: dict) -> str:
    """Compute a stable curation key for an idea when none is present."""
    if idea.get("source") == "user" and idea.get("id") is not None:
        return f"user:{idea['id']}"
    h = hashlib.md5(idea.get("text", "").encode()).hexdigest()[:8]
    return f"gen:{idea.get('type', '')}:{idea.get('section', '')}:{h}"


_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalize_text(text: str) -> str:
    return " ".join(_WORD_RE.findall((text or "").lower()))


def _shingles(text: str, k: int = 4) -> set[str]:
    tokens = _WORD_RE.findall((text or "").lower())
    if len(tokens) <= k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def enrich_idea_data_with_curation_signals(
    ideas: list[dict],
    book_id: str,
    *,
    path: Path | None = None,
    dup_threshold: float = 0.8,
) -> list[dict]:
    """Enrich ideas in-place with 'pinned', 'hidden', and 'near_duplicate' signals.

    - 'pinned' / 'hidden' come from persisted curation state for the book.
    - 'near_duplicate' is True when another idea with a stronger signal
      (pinned > earlier-in-list) shares substantially similar text.

    Returns the same list for chaining so it can be used inline during retrieval.
    """
    if not ideas:
        return ideas

    curation = load_book_curation(book_id, path)

    for idea in ideas:
        key = idea.get("note_key") or _default_idea_key(idea)
        entry = curation.get(key) or {}
        idea["note_key"] = key
        idea["pinned"] = bool(entry.get("pinned"))
        idea["hidden"] = bool(entry.get("hidden"))
        idea["near_duplicate"] = False

    normalized = [_normalize_text(i.get("text", "")) for i in ideas]
    shingles = [_shingles(n) for n in normalized]

    for i, idea in enumerate(ideas):
        if not normalized[i]:
            continue
        for j in range(len(ideas)):
            if i == j:
                continue
            other = ideas[j]
            if other.get("hidden"):
                continue
            if normalized[i] == normalized[j] or _jaccard(shingles[i], shingles[j]) >= dup_threshold:
                pinned_i = idea.get("pinned", False)
                pinned_j = other.get("pinned", False)
                if pinned_j and not pinned_i:
                    idea["near_duplicate"] = True
                    break
                if pinned_i == pinned_j and j < i:
                    idea["near_duplicate"] = True
                    break

    return ideas
