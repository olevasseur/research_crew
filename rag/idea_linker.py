"""Deterministic linking and ranking for ideas and supporting examples.

Ideas and examples are already extracted by rag/inspect_utils.py as dicts with
at least `text`, `section`, and `window` fields (examples are ideas with
`type == "example"`). This module scores candidate examples per idea using
token overlap plus same-section / same-window priors, and returns a small
ranked list per idea keyed by a stable idea key. The same links can also be
reversed to rank examples by the strength of the ideas they support.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

_STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at by for with from as is are was were be been being
    this that these those it its it's their them they we you your our i me my he she his her him
    not no so too very can could would should may might will shall do does did done doing have has
    had having into over under about than which who whom whose where when why how what
    one two three some any all each every most more less much many few several such also however
    because while during against between through across up down out off over under again further
    here there once only own same other
    """.split()
)


def _tokens(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 2 and t.lower() not in _STOPWORDS]


def _token_set(text: str) -> set:
    return set(_tokens(text))


def idea_key(idea: Dict[str, Any]) -> str:
    """Stable key for an idea across requests; mirrors rag_api note_key shape."""
    section = str(idea.get("section") or "")
    window = idea.get("window")
    text = str(idea.get("text") or "")
    return f"{section}::{window}::{text[:80]}"


def example_key(example: Dict[str, Any]) -> str:
    """Stable key for an example across requests."""
    section = str(example.get("section") or "")
    window = example.get("window")
    text = str(example.get("text") or "")
    return f"{section}::{window}::{text[:80]}"


def _score_pair(idea: Dict[str, Any], example: Dict[str, Any]) -> float:
    idea_tokens = _token_set(idea.get("text", ""))
    ex_tokens = _token_set(example.get("text", ""))
    if not idea_tokens or not ex_tokens:
        overlap = 0.0
    else:
        inter = idea_tokens & ex_tokens
        union = idea_tokens | ex_tokens
        overlap = len(inter) / len(union) if union else 0.0

    score = overlap
    if idea.get("section") and idea.get("section") == example.get("section"):
        score += 0.25
        if idea.get("window") is not None and idea.get("window") == example.get("window"):
            score += 0.15

    try:
        score += 0.05 * float(example.get("rank_score") or 0.0)
    except (TypeError, ValueError):
        pass

    return score


def link_ideas_to_examples(
    ideas: Iterable[Dict[str, Any]],
    examples: Iterable[Dict[str, Any]],
    *,
    top_k: int = 3,
    min_score: float = 0.05,
) -> Dict[str, List[Dict[str, Any]]]:
    """Return {idea_key: [example, ...]} sorted by match score desc.

    Only examples that share at least one content token with the idea (or share
    the section) and exceed `min_score` are returned. Duplicate example texts
    are suppressed. The input lists are not mutated.
    """
    idea_list: Sequence[Dict[str, Any]] = [i for i in ideas if isinstance(i, dict)]
    example_list: Sequence[Dict[str, Any]] = [
        e for e in examples if isinstance(e, dict) and str(e.get("text") or "").strip()
    ]

    result: Dict[str, List[Dict[str, Any]]] = {}
    for idea in idea_list:
        key = idea_key(idea)
        scored: List[tuple] = []
        seen_texts: set = set()
        for ex in example_list:
            if ex is idea:
                continue
            text_norm = str(ex.get("text") or "").strip().lower()
            if not text_norm or text_norm in seen_texts:
                continue
            score = _score_pair(idea, ex)
            if score < min_score:
                continue
            seen_texts.add(text_norm)
            scored.append((score, ex))

        scored.sort(key=lambda p: p[0], reverse=True)
        matched = []
        for score, ex in scored[:top_k]:
            matched.append({**ex, "match_score": round(score, 4)})
        result[key] = matched
    return result


def _float_signal(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _specificity_score(example: Dict[str, Any]) -> float:
    """Small inspectable prior for vivid examples independent of idea links."""
    text = str(example.get("text") or "")
    if not text:
        return 0.0

    score = 0.0
    length = len(text)
    if 50 <= length <= 260:
        score += 0.12
    elif 25 <= length < 50 or 260 < length <= 420:
        score += 0.06

    if re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", text):
        score += 0.05
    if re.search(r"\d", text):
        score += 0.04
    if re.search(r"\[[Pp]\.\s*\d+\]", text):
        score += 0.04
    if '"' in text or "\u2018" in text or "\u2019" in text or "\u201c" in text or "\u201d" in text:
        score += 0.03

    return min(score, 0.25)


def rank_examples_by_idea_links(
    ideas: Iterable[Dict[str, Any]],
    examples: Iterable[Dict[str, Any]],
    *,
    top_k: int | None = None,
    links_per_idea: int = 5,
    min_score: float = 0.05,
) -> List[Dict[str, Any]]:
    """Return examples ranked by linked idea strength.

    The score is deterministic and intentionally simple:
      - each linked idea contributes its match score,
      - high-ranked ideas increase that contribution,
      - pinned ideas add a visible boost,
      - example specificity adds a small prior,
      - supporting multiple ideas adds a small capped bonus.

    Returned examples are copies enriched with ``example_score``,
    ``example_score_parts``, and ``associated_ideas``. Inputs are not mutated.
    """
    idea_list: Sequence[Dict[str, Any]] = [
        i for i in ideas if isinstance(i, dict) and i.get("type") != "example"
    ]
    example_list: Sequence[Dict[str, Any]] = [
        e for e in examples if isinstance(e, dict) and str(e.get("text") or "").strip()
    ]

    deduped_examples: List[Dict[str, Any]] = []
    seen_texts: set = set()
    for ex in example_list:
        text_norm = str(ex.get("text") or "").strip().lower()
        if text_norm in seen_texts:
            continue
        seen_texts.add(text_norm)
        deduped_examples.append(ex)

    links = link_ideas_to_examples(
        idea_list,
        deduped_examples,
        top_k=links_per_idea,
        min_score=min_score,
    )

    by_key: Dict[str, Dict[str, Any]] = {
        example_key(ex): {
            "example": ex,
            "link_score": 0.0,
            "pinned_boost": 0.0,
            "rank_boost": 0.0,
            "associated_ideas": [],
        }
        for ex in deduped_examples
    }

    idea_by_key = {idea_key(idea): idea for idea in idea_list}
    for key, matched_examples in links.items():
        idea = idea_by_key.get(key)
        if not idea:
            continue

        idea_rank = max(_float_signal(idea.get("rank_score"), 0.0), 0.0)
        is_pinned = bool(idea.get("pinned"))
        for matched in matched_examples:
            ex_key = example_key(matched)
            entry = by_key.get(ex_key)
            if not entry:
                continue

            match = _float_signal(matched.get("match_score"), 0.0)
            rank_boost = match * min(idea_rank, 1.5) * 0.5
            pinned_boost = match * 0.5 if is_pinned else 0.0
            entry["link_score"] += match
            entry["rank_boost"] += rank_boost
            entry["pinned_boost"] += pinned_boost
            entry["associated_ideas"].append({
                "key": key,
                "text": idea.get("text"),
                "type": idea.get("type"),
                "section": idea.get("section"),
                "window": idea.get("window"),
                "rank_score": idea.get("rank_score"),
                "pinned": is_pinned,
                "match_score": round(match, 4),
            })

    ranked: List[Dict[str, Any]] = []
    for ex in deduped_examples:
        key = example_key(ex)
        entry = by_key[key]
        specificity = _specificity_score(ex)
        association_bonus = min(len(entry["associated_ideas"]) * 0.05, 0.20)
        score = (
            entry["link_score"]
            + entry["rank_boost"]
            + entry["pinned_boost"]
            + association_bonus
            + specificity
        )
        ranked.append({
            **ex,
            "example_key": key,
            "example_score": round(score, 4),
            "example_score_parts": {
                "link_score": round(entry["link_score"], 4),
                "rank_boost": round(entry["rank_boost"], 4),
                "pinned_boost": round(entry["pinned_boost"], 4),
                "association_bonus": round(association_bonus, 4),
                "specificity": round(specificity, 4),
            },
            "associated_ideas": sorted(
                entry["associated_ideas"],
                key=lambda i: (
                    bool(i.get("pinned")),
                    _float_signal(i.get("rank_score"), 0.0),
                    _float_signal(i.get("match_score"), 0.0),
                    str(i.get("text") or ""),
                ),
                reverse=True,
            ),
        })

    ranked.sort(
        key=lambda ex: (
            _float_signal(ex.get("example_score"), 0.0),
            len(ex.get("associated_ideas") or []),
            _float_signal(ex.get("rank_score"), 0.0),
            str(ex.get("section") or ""),
            _float_signal(ex.get("window"), -1.0),
            str(ex.get("text") or ""),
        ),
        reverse=True,
    )
    if top_k is not None:
        return ranked[:top_k]
    return ranked
