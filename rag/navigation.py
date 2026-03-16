"""Navigation: idea tracing and section exploration over summary artifacts.

Deterministic text operations over section summaries and window summaries.
No LLM calls required — works entirely from saved artifacts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from textwrap import indent

from .config import RAGConfig


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _term_variants(idea: str) -> list[str]:
    """Generate search variants: original, hyphen-collapsed, split on hyphens."""
    base = _normalize(idea)
    variants = [base]
    if "-" in base:
        variants.append(base.replace("-", " "))
        variants.append(base.replace("-", ""))
    if " " in base:
        variants.append(base.replace(" ", "-"))
    return list(dict.fromkeys(variants))


def _count_matches(text: str, variants: list[str]) -> int:
    lowered = _normalize(text)
    return sum(lowered.count(v) for v in variants)


def _snippet_around(text: str, variant: str, context_chars: int = 120) -> str | None:
    """Extract a short snippet around the first match of variant in text."""
    lowered = text.lower()
    pos = lowered.find(variant)
    if pos == -1:
        return None
    start = max(0, pos - context_chars)
    end = min(len(text), pos + len(variant) + context_chars)
    snippet = text[start:end].replace("\n", " ").strip()
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


def trace_idea(
    book_id: str,
    idea: str,
    config: RAGConfig,
    limit: int = 20,
    show: str = "both",
) -> None:
    """Trace an idea through a book's summary artifacts.

    Args:
        show: "both" (section + window), "sections" (section summaries only),
              "windows" (window summaries only)
    """
    results_dir = Path(config.storage.results_directory) / book_id
    meta_path = results_dir / "summary_meta.json"
    insights_path = results_dir / "chapter_insights.md"
    windows_path = results_dir / "window_summaries.json"

    if not insights_path.exists():
        print(f"No summaries found for '{book_id}'. Run summarize first.")
        return

    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    insights_text = insights_path.read_text()
    window_data = json.loads(windows_path.read_text()) if windows_path.exists() else {}
    section_meta_map = {s["name"]: s for s in meta.get("sections", [])}

    variants = _term_variants(idea)

    # Parse section summaries from chapter_insights.md
    sections = _parse_section_summaries(insights_text)

    matches: list[dict] = []

    for sec_name, sec_text in sections:
        sec_hits = _count_matches(sec_text, variants)
        sec_snippet = None
        if sec_hits > 0:
            for v in variants:
                sec_snippet = _snippet_around(sec_text, v)
                if sec_snippet:
                    break

        # Check window summaries for this section
        win_matches: list[dict] = []
        for ws in window_data.get(sec_name, []):
            ws_text = ws.get("summary", "")
            ws_hits = _count_matches(ws_text, variants)
            if ws_hits > 0:
                ws_snippet = None
                for v in variants:
                    ws_snippet = _snippet_around(ws_text, v)
                    if ws_snippet:
                        break
                wi = ws.get("window", "?")
                win_matches.append({
                    "window": wi + 1 if isinstance(wi, int) else wi,
                    "hits": ws_hits,
                    "labels": ws.get("labels", []),
                    "snippet": ws_snippet or "",
                })

        total_hits = sec_hits + sum(w["hits"] for w in win_matches)
        if total_hits == 0:
            continue

        sm = section_meta_map.get(sec_name, {})
        match_source = []
        if sec_hits > 0:
            match_source.append("section_summary")
        if win_matches:
            match_source.append("window_summaries")

        matches.append({
            "section": sec_name,
            "section_type": sm.get("type", "?"),
            "pages": sm.get("pages", "?"),
            "total_hits": total_hits,
            "sec_hits": sec_hits,
            "sec_snippet": sec_snippet,
            "window_matches": win_matches,
            "match_source": match_source,
        })

    matches.sort(key=lambda m: m["total_hits"], reverse=True)

    # Output
    print(f"\nTrace: \"{idea}\" in {book_id}")
    print(f"Searched: {len(sections)} sections, {sum(len(window_data.get(s[0], [])) for s in sections)} window summaries")
    print(f"Matching sections: {len(matches)}")
    if meta:
        print(f"Summary quality: {meta.get('quality', '?')}, model: {meta.get('model', '?')}")
    print(f"{'=' * 70}")

    if not matches:
        print(f"\nNo matches for \"{idea}\" in the summary artifacts.")
        print("Try a different phrasing, or check with: inspect search")
        return

    displayed = min(len(matches), limit)
    for rank, m in enumerate(matches[:limit], 1):
        source_str = " + ".join(m["match_source"])
        print(f"\n  #{rank}  {m['section']}")
        print(f"       type={m['section_type']}  pages={m['pages']}  "
              f"hits={m['total_hits']}  via: {source_str}")

        if show in ("both", "sections") and m["sec_hits"] > 0:
            print(f"       Section summary ({m['sec_hits']} hits):")
            print(f"         {m['sec_snippet']}")

        if show in ("both", "windows") and m["window_matches"]:
            for wm in m["window_matches"]:
                labels_str = "+".join(wm["labels"]) if wm["labels"] else "general"
                print(f"       Window {wm['window']} ({wm['hits']} hits, [{labels_str}]):")
                print(f"         {wm['snippet']}")

    if len(matches) > limit:
        print(f"\n  ... {len(matches) - limit} more matching sections (use --limit to show more)")

    print()


def _parse_section_summaries(insights_text: str) -> list[tuple[str, str]]:
    """Parse chapter_insights.md into (section_name, section_text) pairs."""
    parts = re.split(r"\n---\n", insights_text)
    results: list[tuple[str, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        match = re.match(r"^##\s+(.+?)(?:\s+\(in .+?\))?\s*$", part, re.MULTILINE)
        if match:
            name = match.group(1).strip()
            results.append((name, part))
        elif part:
            results.append(("(unnamed)", part))
    return results


# =====================================================================
# Section exploration
# =====================================================================

_SUBSECTION_ORDER = [
    "Core Argument",
    "Key Supporting Ideas",
    "Strongest Examples",
    "Frameworks & Mental Models",
    "Actionable Takeaways",
    "Notable Quotes",
    "Source",
]


def _extract_subsections(summary_text: str) -> dict[str, str]:
    """Extract ### subsections from a section summary into a dict."""
    pattern = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)
    positions = [(m.start(), m.group(1)) for m in pattern.finditer(summary_text)]
    result: dict[str, str] = {}
    for i, (start, name) in enumerate(positions):
        header_end = summary_text.index("\n", start) + 1
        end = positions[i + 1][0] if i + 1 < len(positions) else len(summary_text)
        body = summary_text[header_end:end].strip()
        result[name] = body
    return result


def _compact_preview(text: str, max_lines: int = 6, max_chars: int = 400) -> str:
    """Trim text to a compact preview."""
    lines = text.split("\n")
    out_lines: list[str] = []
    total = 0
    for line in lines:
        if total + len(line) > max_chars or len(out_lines) >= max_lines:
            break
        out_lines.append(line)
        total += len(line)
    result = "\n".join(out_lines)
    if len(result) < len(text):
        result += "\n  ..."
    return result


def _find_section(
    sections: list[tuple[str, str]], target: str,
) -> tuple[str, str] | None:
    """Case-insensitive fuzzy match on section name."""
    target_lower = target.lower().strip()
    for name, text in sections:
        if name.lower().strip() == target_lower:
            return name, text
    for name, text in sections:
        if target_lower in name.lower():
            return name, text
    return None


def explore_section(
    book_id: str,
    section_name: str,
    config: RAGConfig,
    show: str = "all",
) -> None:
    """Explore one section in structured, readable format.

    Args:
        show: "all", "summary", or "windows"
    """
    results_dir = Path(config.storage.results_directory) / book_id
    meta_path = results_dir / "summary_meta.json"
    insights_path = results_dir / "chapter_insights.md"
    windows_path = results_dir / "window_summaries.json"
    selection_path = results_dir / "selection_detail.json"

    if not insights_path.exists():
        print(f"No summaries found for '{book_id}'. Run summarize first.")
        return

    insights_text = insights_path.read_text()
    sections = _parse_section_summaries(insights_text)

    found = _find_section(sections, section_name)
    if not found:
        available = [name for name, _ in sections]
        print(f"Section '{section_name}' not found in summaries for '{book_id}'.")
        print(f"\nAvailable sections:")
        for name in available:
            print(f"  - {name}")
        return

    sec_name, sec_text = found

    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    window_data = json.loads(windows_path.read_text()) if windows_path.exists() else {}
    selection_data = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    section_meta_map = {s["name"]: s for s in meta.get("sections", [])}
    sm = section_meta_map.get(sec_name, {})

    # --- Header ---
    print(f"\n{'=' * 70}")
    print(f"  {sec_name}")
    print(f"{'=' * 70}")
    print(f"  Type:    {sm.get('type', '?')}")
    print(f"  Pages:   {sm.get('pages', '?')}")
    print(f"  Chunks:  {sm.get('chunks', '?')}")
    sel_count = sm.get("windows_selected", "?")
    tot_count = sm.get("windows_total", "?")
    print(f"  Windows: {sel_count}/{tot_count} selected")
    quality = meta.get("quality", "?")
    model = meta.get("model", "?")
    print(f"  Quality: {quality}  Model: {model}")

    # --- Structured summary subsections ---
    if show in ("all", "summary"):
        subsections = _extract_subsections(sec_text)
        print(f"\n{'─' * 70}")

        for label in _SUBSECTION_ORDER:
            body = subsections.get(label)
            if not body:
                continue

            if label == "Source":
                print(f"\n  [{label}]")
                print(indent(body, "    "))
                continue

            print(f"\n  [{label}]")
            compact = _compact_preview(body)
            print(indent(compact, "    "))

        leftover = set(subsections.keys()) - set(_SUBSECTION_ORDER)
        for label in sorted(leftover):
            print(f"\n  [{label}]")
            compact = _compact_preview(subsections[label])
            print(indent(compact, "    "))

    # --- Selected windows ---
    if show in ("all", "windows"):
        windows = window_data.get(sec_name, [])
        sel_detail = selection_data.get(sec_name, [])

        if windows:
            print(f"\n{'─' * 70}")
            print(f"  Selected Window Summaries ({len(windows)})")
            print(f"{'─' * 70}")

            for ws in windows:
                wi = ws.get("window", "?")
                wi_display = wi + 1 if isinstance(wi, int) else wi
                labels = ws.get("labels", [])
                labels_str = "+".join(labels) if labels else "general"
                score = ws.get("score", 0)
                summary = ws.get("summary", "")

                print(f"\n  Window {wi_display}  [{labels_str}]  score={score:.3f}")
                compact = _compact_preview(summary, max_lines=5, max_chars=350)
                print(indent(compact, "    "))

        if sel_detail:
            skipped = [d for d in sel_detail if not d.get("selected")]
            if skipped:
                print(f"\n  Skipped windows: ", end="")
                parts = []
                for d in skipped:
                    idx = d.get("index", "?")
                    idx_display = idx + 1 if isinstance(idx, int) else idx
                    sc = d.get("scores", {}).get("composite", 0)
                    lb = "+".join(d.get("content_labels", [])) or "general"
                    parts.append(f"w{idx_display}({sc:.2f},{lb})")
                print(", ".join(parts))

    print()

