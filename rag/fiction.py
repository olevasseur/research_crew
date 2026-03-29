"""Fiction extraction pipeline.

Processes an already-summarised book through a fiction lens:
extracts characters, relationships, places, events and time-markers per
chapter, then accumulates a spoiler-aware cumulative state by chapter.

Prerequisite: run `rag_cli.py summarize <book_id>` first.

Artifacts written to data/results/<book_id>/:
  fiction_chapter_facts.json  — per-chapter extracted facts
  fiction_state.json          — cumulative state snapshots keyed by chapter index
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .config import RAGConfig
from .llm import LLMClient


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

FICTION_SYSTEM = """\
You are a careful fiction analyst. Your task is to extract structured facts
from chapter summaries of a novel.

Rules:
- Extract only facts clearly stated or strongly implied in the provided text.
- Do not speculate or invent details not present in the source.
- If a field is empty or uncertain, use an empty list or empty string.
- Respond with valid JSON only. No explanation. No markdown fences."""

FICTION_EXTRACT_PROMPT = """\
Chapter: {chapter}

The following are summaries of key passages from this chapter:

{context}

Extract structured facts from this chapter. Be conservative — only include what the text clearly supports.

Respond with JSON in exactly this format (no other text, no markdown):
{{
  "characters": [
    {{"name": "Character Name", "aliases": [], "role": "short description of role in story", "new_this_chapter": true}}
  ],
  "relationships": [
    {{"character_a": "Name A", "character_b": "Name B", "description": "nature of relationship", "evidence": "brief supporting quote or paraphrase"}}
  ],
  "places": [
    {{"name": "Place Name", "description": "what it is or its significance", "new_this_chapter": true}}
  ],
  "events": [
    {{"description": "what happened", "characters": ["Name A", "Name B"], "place": "Place Name or empty string"}}
  ],
  "time_markers": ["explicit time cue e.g. 'in 1847' or 'three days later' or 'during winter'"],
  "open_questions": ["unresolved mystery or question clearly raised by the chapter"]
}}
"""

_EMPTY_FACTS: dict = {
    "characters": [],
    "relationships": [],
    "places": [],
    "events": [],
    "time_markers": [],
    "open_questions": [],
}


# ---------------------------------------------------------------------------
# JSON parsing (tolerant of LLM formatting noise)
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> dict:
    """Extract a JSON object from LLM output, tolerating markdown fences."""
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text.strip(), flags=re.MULTILINE).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Section parsing (mirrors logic in navigation.py)
# ---------------------------------------------------------------------------

def _parse_sections(insights_text: str) -> list[tuple[str, str]]:
    """Parse chapter_insights.md into (section_name, section_text) pairs."""
    parts = re.split(r"\n---\n", insights_text)
    results: list[tuple[str, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^##\s+(.+?)(?:\s+\(in .+?\))?\s*$", part, re.MULTILINE)
        name = m.group(1).strip() if m else "(unnamed)"
        results.append((name, part))
    return results


def _section_body(sec_text: str) -> str:
    """Strip ## heading and **Type:**/**Selected:** metadata lines."""
    lines = sec_text.split("\n")
    i = 0
    while i < len(lines) and (
        lines[i].startswith("##")
        or lines[i].startswith("**Type:")
        or lines[i].startswith("**Selected:")
    ):
        i += 1
    return "\n".join(lines[i:]).strip()


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context(sec_text: str, windows: list[dict], max_chars: int = 6000) -> str:
    """Combine section summary + window summaries for the extraction prompt."""
    parts: list[str] = []

    body = _section_body(sec_text)
    if body:
        parts.append(f"[Section Summary]\n{body}")

    for w in windows:
        summary = w.get("summary", "").strip()
        if not summary:
            continue
        wi = w.get("window", "?")
        label = wi + 1 if isinstance(wi, int) else wi
        parts.append(f"[Window {label}]\n{summary}")

    context = "\n\n---\n\n".join(parts)
    if len(context) > max_chars:
        context = context[:max_chars] + "\n[... truncated]"
    return context


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


# ---------------------------------------------------------------------------
# Cumulative state merger
# ---------------------------------------------------------------------------

def _empty_state() -> dict:
    return {
        "characters": [],
        "relationships": [],
        "places": [],
        "events": [],
        "time_markers": [],
    }


def _merge(state: dict, facts: dict, chapter_index: int, chapter_name: str) -> None:
    """Merge one chapter's facts into the running cumulative state (in-place)."""

    # Characters — key by normalised name, accumulate from first appearance
    char_map: dict[str, dict] = {_norm(c["name"]): c for c in state["characters"]}
    for c in facts.get("characters", []):
        key = _norm(c.get("name", ""))
        if not key:
            continue
        if key not in char_map:
            char_map[key] = {
                "name": c["name"],
                "aliases": list(c.get("aliases", [])),
                "role": c.get("role", ""),
                "first_chapter": chapter_name,
                "first_chapter_index": chapter_index,
            }
        else:
            existing = char_map[key]
            # Keep longer/richer role description
            if len(c.get("role", "")) > len(existing.get("role", "")):
                existing["role"] = c["role"]
            # Merge aliases
            for alias in c.get("aliases", []):
                if alias and alias not in existing["aliases"]:
                    existing["aliases"].append(alias)
    state["characters"] = list(char_map.values())

    # Relationships — deduplicate by unordered (a, b) pair
    seen_rels: set[tuple[str, str]] = set()
    for r in state["relationships"]:
        a, b = _norm(r["character_a"]), _norm(r["character_b"])
        seen_rels.add((min(a, b), max(a, b)))

    for r in facts.get("relationships", []):
        a = _norm(r.get("character_a", ""))
        b = _norm(r.get("character_b", ""))
        if not a or not b:
            continue
        pair = (min(a, b), max(a, b))
        if pair not in seen_rels:
            seen_rels.add(pair)
            state["relationships"].append({
                "character_a": r["character_a"],
                "character_b": r["character_b"],
                "description": r.get("description", ""),
                "evidence": r.get("evidence", ""),
                "first_chapter": chapter_name,
                "first_chapter_index": chapter_index,
            })

    # Places — key by normalised name
    place_map: dict[str, dict] = {_norm(p["name"]): p for p in state["places"]}
    for p in facts.get("places", []):
        key = _norm(p.get("name", ""))
        if not key:
            continue
        if key not in place_map:
            place_map[key] = {
                "name": p["name"],
                "description": p.get("description", ""),
                "first_chapter": chapter_name,
                "first_chapter_index": chapter_index,
            }
    state["places"] = list(place_map.values())

    # Events — append all (each is a unique occurrence)
    for e in facts.get("events", []):
        desc = e.get("description", "").strip()
        if not desc:
            continue
        state["events"].append({
            "description": desc,
            "characters": list(e.get("characters", [])),
            "place": e.get("place", ""),
            "chapter": chapter_name,
            "chapter_index": chapter_index,
        })

    # Time markers — deduplicate by normalised text
    seen_tm: set[str] = {_norm(t) for t in state["time_markers"]}
    for tm in facts.get("time_markers", []):
        if tm and _norm(tm) not in seen_tm:
            seen_tm.add(_norm(tm))
            state["time_markers"].append(tm)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def extract_fiction_book(
    book_id: str,
    config: RAGConfig,
    force: bool = False,
) -> None:
    """Run fiction extraction for every summarised chapter of a book."""
    results_dir = Path(config.storage.results_directory) / book_id
    insights_path = results_dir / "chapter_insights.md"
    windows_path  = results_dir / "window_summaries.json"
    meta_path     = results_dir / "summary_meta.json"
    out_facts     = results_dir / "fiction_chapter_facts.json"
    out_state     = results_dir / "fiction_state.json"

    if not insights_path.exists():
        print(f"No summaries found for '{book_id}'. Run 'summarize' first.")
        return

    if out_facts.exists() and not force:
        print(f"Fiction facts already exist for '{book_id}'. Use --force to re-extract.")
        print(f"  {out_facts}")
        return

    meta        = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    insights    = insights_path.read_text()
    window_data = json.loads(windows_path.read_text()) if windows_path.exists() else {}
    sections    = _parse_sections(insights)

    llm = LLMClient(config.generation)

    print(f"Fiction extraction: '{book_id}'  ({len(sections)} sections)")
    print(f"Model: {config.generation.model}")
    print()

    chapter_facts_list: list[dict] = []
    cumulative = _empty_state()
    fiction_state: dict[str, dict] = {}

    for i, (sec_name, sec_text) in enumerate(sections):
        print(f"  [{i + 1}/{len(sections)}] {sec_name} … ", end="", flush=True)

        windows = window_data.get(sec_name, [])
        context = _build_context(sec_text, windows)
        prompt  = FICTION_EXTRACT_PROMPT.format(chapter=sec_name, context=context)

        try:
            raw   = llm.generate(prompt, system=FICTION_SYSTEM)
            facts = _parse_json(raw)
        except Exception as exc:
            print(f"ERROR: {exc}")
            facts = {}

        for key in _EMPTY_FACTS:
            facts.setdefault(key, [])

        entry = {"chapter": sec_name, "chapter_index": i, **facts}
        chapter_facts_list.append(entry)

        _merge(cumulative, facts, i, sec_name)

        import copy
        fiction_state[str(i)] = {
            "chapter_index": i,
            "chapter": sec_name,
            "state": copy.deepcopy(cumulative),
        }

        n_c = len(facts.get("characters", []))
        n_r = len(facts.get("relationships", []))
        n_p = len(facts.get("places", []))
        n_e = len(facts.get("events", []))
        print(f"chars={n_c} rels={n_r} places={n_p} events={n_e}")

    out_facts.write_text(json.dumps(chapter_facts_list, indent=2, ensure_ascii=False))
    out_state.write_text(json.dumps(fiction_state, indent=2, ensure_ascii=False))

    total_chars  = len(cumulative["characters"])
    total_places = len(cumulative["places"])
    total_events = len(cumulative["events"])

    print()
    print(f"Done. Cumulative totals: {total_chars} characters, "
          f"{total_places} places, {total_events} events")
    print(f"Saved:")
    print(f"  {out_facts}")
    print(f"  {out_state}")


# ---------------------------------------------------------------------------
# Chapter number -> snapshot key resolver
# ---------------------------------------------------------------------------

def _resolve_snapshot_key(fiction_state: dict, chapter_number: int) -> int | None:
    """
    Return the highest snapshot key whose chapter label number is <= chapter_number.

    Searches by the numeric value embedded in the chapter name (e.g. "Chapter 10"),
    so front-matter entries (e.g. "Chapter 24: Author's Note" at index 0) do not
    cause an off-by-one when the user specifies --chapter N.

    Falls back to raw index arithmetic if no chapter-numbered entries are found.
    """
    best_key: int | None = None
    best_num = -1

    for k, v in fiction_state.items():
        chapter_name = v.get("chapter", "")
        m = re.match(r"chapter\s+(\d+)(?:\s*:|$)", chapter_name, re.IGNORECASE)
        if m:
            n = int(m.group(1))
            key_int = int(k)
            if n <= chapter_number and (n > best_num or (n == best_num and key_int > (best_key or -1))):
                best_num = n
                best_key = key_int

    if best_key is None:
        # Fallback: treat keys as 0-based indices, chapter_number as 1-based
        target_index = chapter_number - 1
        available    = sorted(int(k) for k in fiction_state)
        best_key     = max((k for k in available if k <= target_index), default=None)

    return best_key


# ---------------------------------------------------------------------------
# State inspection
# ---------------------------------------------------------------------------

def show_fiction_state(
    book_id: str,
    config: RAGConfig,
    chapter_number: int,           # 1-based (user-facing)
) -> None:
    """Print cumulative fiction state up to chapter N (1-based)."""
    results_dir = Path(config.storage.results_directory) / book_id
    state_path  = results_dir / "fiction_state.json"

    if not state_path.exists():
        print(f"No fiction state found for '{book_id}'. Run 'fiction-extract' first.")
        return

    fiction_state = json.loads(state_path.read_text())

    resolved = _resolve_snapshot_key(fiction_state, chapter_number)

    if resolved is None:
        print(f"No data for chapter {chapter_number}. Available indices: {[k + 1 for k in available]}")
        return

    snap         = fiction_state[str(resolved)]
    chapter_name = snap["chapter"]
    state        = snap["state"]

    print(f"\nFiction state — up to: {chapter_name}")
    print("=" * 70)

    # Characters
    chars = state.get("characters", [])
    print(f"\nCharacters ({len(chars)})")
    print("─" * 40)
    if chars:
        for c in chars:
            aliases = f"  [aka: {', '.join(c['aliases'])}]" if c.get("aliases") else ""
            print(f"  {c['name']}{aliases}")
            if c.get("role"):
                print(f"    role: {c['role']}")
            print(f"    first seen: {c.get('first_chapter', '?')}")
    else:
        print("  (none)")

    # Relationships
    rels = state.get("relationships", [])
    if rels:
        print(f"\nRelationships ({len(rels)})")
        print("─" * 40)
        for r in rels:
            print(f"  {r['character_a']}  ↔  {r['character_b']}")
            if r.get("description"):
                print(f"    {r['description']}")
            if r.get("evidence"):
                print(f"    evidence: {r['evidence']}")

    # Places
    places = state.get("places", [])
    if places:
        print(f"\nPlaces ({len(places)})")
        print("─" * 40)
        for p in places:
            desc = f" — {p['description']}" if p.get("description") else ""
            print(f"  {p['name']}{desc}")
            print(f"    first seen: {p.get('first_chapter', '?')}")

    # Events
    events = state.get("events", [])
    if events:
        print(f"\nEvents ({len(events)})")
        print("─" * 40)
        for e in events:
            chars_str = f" ({', '.join(e['characters'])})" if e.get("characters") else ""
            place_str = f" @ {e['place']}" if e.get("place") else ""
            print(f"  [{e.get('chapter', '?')}] {e['description']}{chars_str}{place_str}")

    # Time markers
    tms = state.get("time_markers", [])
    if tms:
        print(f"\nTime markers ({len(tms)})")
        print("─" * 40)
        for tm in tms:
            print(f"  · {tm}")

    print()


# ---------------------------------------------------------------------------
# Character profile (whois)
# ---------------------------------------------------------------------------

def fiction_whois(
    book_id: str,
    config: "RAGConfig",
    chapter_number: int,    # 1-based (user-facing)
    character_name: str,
) -> None:
    """Print a character profile up to the given chapter (1-based)."""
    results_dir = Path(config.storage.results_directory) / book_id
    state_path  = results_dir / "fiction_state.json"

    if not state_path.exists():
        print(f"Error: No fiction state found for '{book_id}'. Run 'fiction-extract' first.")
        return

    fiction_state = json.loads(state_path.read_text())

    if not fiction_state:
        print(f"Error: Fiction state for '{book_id}' is empty.")
        return

    if chapter_number < 1:
        print("Error: --chapter must be >= 1.")
        return

    resolved = _resolve_snapshot_key(fiction_state, chapter_number)
    available = sorted(int(k) for k in fiction_state)

    if resolved is None:
        print(
            f"Error: Chapter {chapter_number} is before any available data. "
            f"First available: chapter {min(available) + 1}."
        )
        return

    if resolved < max(available) and chapter_number > max(available):
        print(
            f"Warning: Chapter {chapter_number} exceeds available data "
            f"(max: {max(available) + 1}). Showing state up to chapter {max(available) + 1}."
        )

    snap  = fiction_state[str(resolved)]
    state = snap["state"]

    # --- Resolve character (case-insensitive on name and aliases) ---
    query   = _norm(character_name)
    matched = None
    for c in state.get("characters", []):
        if _norm(c["name"]) == query:
            matched = c
            break
        for alias in c.get("aliases", []):
            if _norm(alias) == query:
                matched = c
                break
        if matched:
            break

    if matched is None:
        known = [c["name"] for c in state.get("characters", [])]
        if known:
            print(
                f"Error: Character '{character_name}' not found up to chapter {chapter_number}.\n"
                f"Known characters: {', '.join(known)}"
            )
        else:
            print(
                f"Error: Character '{character_name}' not found. "
                f"No characters known up to chapter {chapter_number}."
            )
        return

    # Build the set of normalised names/aliases for this character
    char_keys: set[str] = {_norm(matched["name"])}
    for alias in matched.get("aliases", []):
        if alias:
            char_keys.add(_norm(alias))

    def _involves_rel(r: dict) -> bool:
        return _norm(r.get("character_a", "")) in char_keys \
            or _norm(r.get("character_b", "")) in char_keys

    def _involves_event(e: dict) -> bool:
        return any(_norm(n) in char_keys for n in e.get("characters", []))

    relevant_rels   = [r for r in state.get("relationships", []) if _involves_rel(r)]
    relevant_events = sorted(
        [e for e in state.get("events", []) if _involves_event(e)],
        key=lambda e: e.get("chapter_index", 0),
    )
    recent_events   = list(reversed(relevant_events))   # most-recent first

    # Places: deduplicated, drawn from events involving this character
    seen_places: dict[str, str] = {}   # norm_key -> display name
    for e in relevant_events:
        place = e.get("place", "").strip()
        if place:
            key = _norm(place)
            if key not in seen_places:
                seen_places[key] = place

    # --- Print report ---
    ch_name = snap["chapter"]
    print(f"\nCharacter: {matched['name']}")
    print(f"Book: {book_id}  |  State up to: {ch_name}")
    print("=" * 60)

    # Role
    role = matched.get("role") or "Unknown"
    print(f"\nRole\n{'─' * 40}")
    print(f"  {role}")

    # Aliases
    aliases = matched.get("aliases", [])
    print(f"\nAliases\n{'─' * 40}")
    print(f"  {', '.join(aliases) if aliases else 'None'}")

    # First seen
    fc_idx  = matched.get("first_chapter_index", 0)
    fc_name = matched.get("first_chapter", "Unknown")
    print(f"\nFirst Seen\n{'─' * 40}")
    print(f"  {fc_name}")

    # Relationships
    print(f"\nRelationships Known So Far ({len(relevant_rels)})\n{'─' * 40}")
    if relevant_rels:
        for r in relevant_rels:
            other = (
                r["character_b"]
                if _norm(r.get("character_a", "")) in char_keys
                else r["character_a"]
            )
            desc = f": {r['description']}" if r.get("description") else ""
            print(f"  ↔ {other}{desc}")
            if r.get("evidence"):
                print(f"      evidence: {r['evidence']}")
    else:
        print("  None so far")

    # Recent events (most-recent first, capped at 10 for conciseness)
    _MAX_EVENTS = 10
    print(f"\nRecent Events Involving {matched['name']}\n{'─' * 40}")
    if recent_events:
        for e in recent_events[:_MAX_EVENTS]:
            place_str = f" @ {e['place']}" if e.get("place") else ""
            print(f"  [{e.get('chapter', '?')}] {e['description']}{place_str}")
        if len(recent_events) > _MAX_EVENTS:
            print(f"  … and {len(recent_events) - _MAX_EVENTS} earlier event(s)")
    else:
        print("  None so far")

    # Places
    print(f"\nPlaces Associated So Far\n{'─' * 40}")
    if seen_places:
        for place in seen_places.values():
            print(f"  · {place}")
    else:
        print("  None so far")

    print()


# ---------------------------------------------------------------------------
# Diff between two reading points
# ---------------------------------------------------------------------------

def fiction_diff(
    book_id: str,
    config: RAGConfig,
    from_chapter: int,   # 1-based
    to_chapter: int,     # 1-based
) -> None:
    """Print what changed between two chapter reading points (spoiler-safe)."""
    if from_chapter >= to_chapter:
        print(f"Error: --from-chapter ({from_chapter}) must be less than --to-chapter ({to_chapter}).")
        return

    results_dir = Path(config.storage.results_directory) / book_id
    state_path  = results_dir / "fiction_state.json"

    if not state_path.exists():
        print(f"No fiction state found for '{book_id}'. Run 'fiction-extract' first.")
        return

    fiction_state = json.loads(state_path.read_text())

    from_key = _resolve_snapshot_key(fiction_state, from_chapter)
    to_key   = _resolve_snapshot_key(fiction_state, to_chapter)

    if from_key is None:
        print(f"Error: No data available at or before chapter {from_chapter}.")
        return
    if to_key is None:
        print(f"Error: No data available at or before chapter {to_chapter}.")
        return
    if from_key == to_key:
        print(f"No new items: both chapters resolve to the same snapshot "
              f"({fiction_state[str(from_key)]['chapter']}).")
        return

    from_state = fiction_state[str(from_key)]["state"]
    to_state   = fiction_state[str(to_key)]["state"]
    from_name  = fiction_state[str(from_key)]["chapter"]
    to_name    = fiction_state[str(to_key)]["chapter"]

    # --- Compute diffs ---
    from_char_keys = {_norm(c["name"]) for c in from_state.get("characters", [])}
    new_chars = [
        c for c in to_state.get("characters", [])
        if _norm(c["name"]) not in from_char_keys
    ]

    from_rel_keys: set[tuple[str, str]] = set()
    for r in from_state.get("relationships", []):
        a, b = _norm(r["character_a"]), _norm(r["character_b"])
        from_rel_keys.add((min(a, b), max(a, b)))
    new_rels = []
    for r in to_state.get("relationships", []):
        a, b = _norm(r["character_a"]), _norm(r["character_b"])
        if (min(a, b), max(a, b)) not in from_rel_keys:
            new_rels.append(r)

    from_place_keys = {_norm(p["name"]) for p in from_state.get("places", [])}
    new_places = [
        p for p in to_state.get("places", [])
        if _norm(p["name"]) not in from_place_keys
    ]

    from_event_keys = {_norm(e["description"]) for e in from_state.get("events", [])}
    new_events = [
        e for e in to_state.get("events", [])
        if _norm(e["description"]) not in from_event_keys
    ]

    from_tm_keys = {_norm(t) for t in from_state.get("time_markers", [])}
    new_tms = [
        t for t in to_state.get("time_markers", [])
        if _norm(t) not in from_tm_keys
    ]

    _MAX = 10  # max items per section

    # --- Print ---
    print(f"\nFiction Progress Diff")
    print(f"Book: {book_id}")
    print(f"From: {from_name}  →  To: {to_name}")
    print("=" * 60)

    any_changes = False

    if new_chars:
        any_changes = True
        print(f"\nNew Characters ({len(new_chars)})")
        print("─" * 40)
        for c in new_chars[:_MAX]:
            role_str = f" — {c['role']}" if c.get("role") else ""
            print(f"  · {c['name']}{role_str}")
        if len(new_chars) > _MAX:
            print(f"  … and {len(new_chars) - _MAX} more")

    if new_rels:
        any_changes = True
        print(f"\nNew Relationships ({len(new_rels)})")
        print("─" * 40)
        for r in new_rels[:_MAX]:
            desc = f": {r['description']}" if r.get("description") else ""
            print(f"  · {r['character_a']}  ↔  {r['character_b']}{desc}")
        if len(new_rels) > _MAX:
            print(f"  … and {len(new_rels) - _MAX} more")

    if new_places:
        any_changes = True
        print(f"\nNew Places ({len(new_places)})")
        print("─" * 40)
        for p in new_places[:_MAX]:
            desc = f" — {p['description']}" if p.get("description") else ""
            print(f"  · {p['name']}{desc}")
        if len(new_places) > _MAX:
            print(f"  … and {len(new_places) - _MAX} more")

    if new_events:
        any_changes = True
        print(f"\nNew Events ({len(new_events)})")
        print("─" * 40)
        for e in new_events[:_MAX]:
            chars_str = f" ({', '.join(e['characters'])})" if e.get("characters") else ""
            place_str = f" @ {e['place']}" if e.get("place") else ""
            print(f"  · [{e.get('chapter', '?')}] {e['description']}{chars_str}{place_str}")
        if len(new_events) > _MAX:
            print(f"  … and {len(new_events) - _MAX} more")

    if new_tms:
        any_changes = True
        print(f"\nNew Time Markers ({len(new_tms)})")
        print("─" * 40)
        for tm in new_tms[:_MAX]:
            print(f"  · {tm}")
        if len(new_tms) > _MAX:
            print(f"  … and {len(new_tms) - _MAX} more")

    if not any_changes:
        print("\nNo new items between these two reading points.")

    print()
