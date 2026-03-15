"""Book analysis: quality-focused windowed summarization with caching.

Pipeline:
  1. Group each section's subchunks into summary windows.
  2. Score windows on content quality (examples, frameworks, claims, actions,
     concept density, position, title overlap) and select via MMR diversity.
  3. Summarize only selected windows (cached).
  4. Synthesize section summary from selected window summaries (cached).
  5. Synthesize book summary from section summaries (cached).
  6. Save detailed selection metadata for every window (selected or not).

Budget tiers: fast=3, default=6, thorough=all windows/section.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .cache import SummaryCache
from .chunker import SECTION_TYPES_SUMMARIZABLE, SECTION_TYPES_SKIPPABLE
from .config import RAGConfig, SummarizationConfig
from .llm import LLMClient
from .retrieval import Retrieval

# ---------------------------------------------------------------------------
# Section type filters
# ---------------------------------------------------------------------------

SUMMARIZE_MODES = {
    "default": SECTION_TYPES_SUMMARIZABLE,
    "body-only": {"chapter"},
    "chapter-only": {"chapter"},
    "full": SECTION_TYPES_SUMMARIZABLE | SECTION_TYPES_SKIPPABLE,
    "back-matter": {"acknowledgments", "notes", "index", "about_author"},
}


def resolve_section_filter(
    mode: str = "default",
    include_types: list[str] | None = None,
    exclude_types: list[str] | None = None,
) -> set[str]:
    allowed = set(SUMMARIZE_MODES.get(mode, SUMMARIZE_MODES["default"]))
    if include_types:
        allowed |= set(include_types)
    if exclude_types:
        allowed -= set(exclude_types)
    return allowed


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def build_windows(
    chunks: list[dict],
    window_size: int,
    window_overlap: int,
) -> list[list[dict]]:
    """Group sorted chunks into windows of approximately window_size characters."""
    if not chunks:
        return []

    windows: list[list[dict]] = []
    current_window: list[dict] = []
    current_size = 0

    for c in chunks:
        clen = len(c.get("text", ""))
        if current_size + clen > window_size and current_window:
            windows.append(current_window)
            overlap_chunks: list[dict] = []
            overlap_size = 0
            for oc in reversed(current_window):
                oc_len = len(oc.get("text", ""))
                if overlap_size + oc_len > window_overlap:
                    break
                overlap_chunks.insert(0, oc)
                overlap_size += oc_len
            current_window = list(overlap_chunks)
            current_size = overlap_size

        current_window.append(c)
        current_size += clen

    if current_window:
        windows.append(current_window)

    return windows


# ---------------------------------------------------------------------------
# Content-type detection patterns
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "the a an and or but in on at to for of is it this that with as by from be "
    "are was were been have has had do does did will would could should may might "
    "can shall not no so if than they he she we you i my our his her its their "
    "about also been being between both come could each few get got how into just "
    "like make many more most much new only other own part same see some still such "
    "take tell than that them then there these thing those through too under very "
    "want way well what when where which while who will with work year your".split()
)

_EXAMPLE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bfor example\b", r"\bfor instance\b", r"\bsuch as\b",
        r"\bcase study\b", r"\bconsider the\b", r"\btake the case\b",
        r"\bstory of\b", r"\bin \d{4}\b",
        r"\bexperiment\b", r"\bstudy\b(?:\s+(?:found|showed|revealed))",
        r"\baccording to\b", r"\bfound that\b", r"\bresearch(?:ers?)?\s+(?:at|from|by)\b",
        r"\bsurvey(?:ed)?\b", r"\binterview(?:ed|s)?\b",
        r"\bpercent\b|\b\d+%", r"\banecdot", r"\bparticipant",
    ]
]

_FRAMEWORK_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bframework\b", r"\bmodel\b", r"\bprinciple\b", r"\brule\b",
        r"\bmethod\b", r"\bstrategy\b", r"\bapproach\b",
        r"\bprocess\b", r"\bstep\s+\d\b", r"\bprotocol\b",
        r"\bconcept of\b", r"\bdefined as\b", r"\brefers to\b",
        r"\bphilosophy\b", r"\bhypothesis\b", r"\btheory\b",
        r"\blaw of\b", r"\bfallacy\b",
    ]
]

_CLAIM_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bargues?\b", r"\bclaims?\b", r"\bthesis\b", r"\bcontends?\b",
        r"\bevidence suggests\b", r"\bthe key\s+(?:point|idea|insight|argument)\b",
        r"\bfundamental(?:ly)?\b", r"\bcentral\s+(?:claim|argument|point|thesis)\b",
        r"\bcore\s+(?:idea|argument|point)\b",
        r"\bthe point is\b", r"\bthe problem is\b",
        r"\bin other words\b", r"\bthe real\s+(?:issue|question|problem)\b",
        r"\bcritical(?:ly)?\b", r"\bessential(?:ly)?\b",
    ]
]

_ACTION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\byou should\b", r"\byou must\b", r"\btry to\b",
        r"\bpractice\b", r"\bimplement\b", r"\bstart by\b",
        r"\brecommend\b", r"\badvise\b", r"\bsuggest(?:ion|s|ed)?\b",
        r"\bexercise\b", r"\bchallenge\b", r"\bcommit to\b",
        r"\bschedul(?:e|ing)\b", r"\broutine\b",
    ]
]


def _count_pattern_hits(text: str, patterns: list[re.Pattern]) -> int:
    return sum(1 for p in patterns if p.search(text))


def _detect_content_types(text: str) -> dict[str, float]:
    """Score presence of different content categories in window text (0.0–1.0)."""
    return {
        "examples": min(_count_pattern_hits(text, _EXAMPLE_PATTERNS) / 3.0, 1.0),
        "frameworks": min(_count_pattern_hits(text, _FRAMEWORK_PATTERNS) / 2.0, 1.0),
        "claims": min(_count_pattern_hits(text, _CLAIM_PATTERNS) / 2.0, 1.0),
        "actions": min(_count_pattern_hits(text, _ACTION_PATTERNS) / 2.0, 1.0),
    }


def _concept_density(text: str) -> float:
    """Information density: blend of type-token ratio and absolute unique term count."""
    words = re.findall(r"[a-z]{3,}", text.lower())
    if len(words) < 10:
        return 0.0
    content = [w for w in words if w not in _STOPWORDS]
    if not content:
        return 0.0
    unique = set(content)
    ttr = len(unique) / len(content)
    absolute = min(len(unique) / 120.0, 1.0)
    return ttr * 0.5 + absolute * 0.5


def _specificity_score(text: str) -> float:
    """Detect presence of concrete specifics: proper nouns, numbers, quotes."""
    signals = 0
    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", text):
        signals += 1
    if re.search(r"\b\d{4}\b", text):
        signals += 1
    if re.search(r"\d+%|\b\d+\s+percent\b", text, re.IGNORECASE):
        signals += 1
    if re.search(r'["\u201c\u201d].{10,}["\u201c\u201d]', text):
        signals += 1
    if re.search(r"\b(?:Dr\.|Professor|University|Institute|Harvard|Stanford|MIT)\b", text):
        signals += 1
    return min(signals / 3.0, 1.0)


def _title_overlap(text: str, title: str) -> float:
    title_words = set(re.findall(r"[a-z]{3,}", title.lower())) - _STOPWORDS
    if not title_words:
        return 0.0
    text_words = set(re.findall(r"[a-z]{3,}", text.lower()))
    return len(title_words & text_words) / len(title_words)


def _position_score(idx: int, total: int) -> float:
    if total <= 1:
        return 1.0
    if idx == 0:
        return 1.0
    if idx == total - 1:
        return 0.85
    if idx == 1:
        return 0.6
    return max(0.2, 0.5 - 0.3 * abs(idx - total / 2) / (total / 2))


# ---------------------------------------------------------------------------
# Composite scoring + MMR selection
# ---------------------------------------------------------------------------

def _score_window(text: str, idx: int, total: int, section_title: str) -> dict:
    """Compute all scoring signals for a single window."""
    ct = _detect_content_types(text)
    content_type_score = (
        ct["examples"] * 0.35
        + ct["frameworks"] * 0.25
        + ct["claims"] * 0.25
        + ct["actions"] * 0.15
    )
    cd = _concept_density(text)
    sp = _specificity_score(text)
    to = _title_overlap(text, section_title)
    ps = _position_score(idx, total)

    composite = (
        content_type_score * 0.30
        + cd * 0.15
        + sp * 0.20
        + to * 0.10
        + ps * 0.15
        + min((ct["examples"] + ct["frameworks"]) / 2.0, 1.0) * 0.10
    )
    return {
        "composite": round(composite, 4),
        "content_type": round(content_type_score, 3),
        "concept_density": round(cd, 3),
        "specificity": round(sp, 3),
        "title_overlap": round(to, 3),
        "position": round(ps, 3),
        "detected_types": {k: round(v, 2) for k, v in ct.items()},
    }


def _jaccard_bigrams(text1: str, text2: str) -> float:
    """Jaccard similarity on word bigrams — lightweight diversity measure."""
    w1 = re.findall(r"[a-z]{3,}", text1.lower())
    w2 = re.findall(r"[a-z]{3,}", text2.lower())
    if len(w1) < 2 or len(w2) < 2:
        return 0.0
    b1 = set(zip(w1, w1[1:]))
    b2 = set(zip(w2, w2[1:]))
    inter = b1 & b2
    union = b1 | b2
    return len(inter) / len(union) if union else 0.0


def select_windows(
    windows: list[list[dict]],
    budget: int,
    section_title: str,
    always_first: bool = True,
    always_last: bool = True,
    strategy: str = "content_diversity",
    mmr_lambda: float = 0.7,
) -> tuple[list[dict], list[dict]]:
    """Score and select windows via content analysis + MMR diversity.

    Returns:
        (all_window_details, selected_window_details) where each detail is a dict:
        {index, chunks, scores, selected, reason, content_labels}
    """
    if not windows:
        return [], []

    # Build text and score every window
    window_texts: list[str] = []
    all_details: list[dict] = []
    for i, w in enumerate(windows):
        w_text = " ".join(c.get("text", "") for c in w)
        window_texts.append(w_text)
        scores = _score_window(w_text, i, len(windows), section_title)

        labels = []
        dt = scores["detected_types"]
        if dt.get("examples", 0) >= 0.33:
            labels.append("example")
        if dt.get("frameworks", 0) >= 0.50:
            labels.append("framework")
        if dt.get("claims", 0) >= 0.50:
            labels.append("claim")
        if dt.get("actions", 0) >= 0.50:
            labels.append("action")
        if scores["specificity"] >= 0.33:
            labels.append("specific")

        all_details.append({
            "index": i,
            "chunks": w,
            "chunk_ids": [c.get("id", "?") for c in w],
            "scores": scores,
            "selected": False,
            "reason": "",
            "content_labels": labels,
        })

    # Select all if budget >= total or strategy is "all"
    if budget >= len(windows) or strategy == "all":
        for d in all_details:
            d["selected"] = True
            d["reason"] = "all_windows_selected"
        return all_details, all_details

    # Forced first/last
    forced: set[int] = set()
    if always_first:
        forced.add(0)
        all_details[0]["reason"] = "forced_first"
    if always_last and len(windows) > 1:
        forced.add(len(windows) - 1)
        all_details[len(windows) - 1]["reason"] = "forced_last"

    for i in forced:
        all_details[i]["selected"] = True

    # MMR selection for remaining budget
    remaining = budget - len(forced)
    selected_indices = list(forced)
    selected_texts = [window_texts[i] for i in forced]

    candidates = [i for i in range(len(windows)) if i not in forced]

    for _ in range(remaining):
        if not candidates:
            break

        best_idx = -1
        best_mmr = -1.0

        for ci in candidates:
            relevance = all_details[ci]["scores"]["composite"]
            max_sim = 0.0
            for st in selected_texts:
                sim = _jaccard_bigrams(window_texts[ci], st)
                if sim > max_sim:
                    max_sim = sim
            mmr = mmr_lambda * relevance - (1.0 - mmr_lambda) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = ci

        if best_idx >= 0:
            all_details[best_idx]["selected"] = True
            rel = all_details[best_idx]["scores"]["composite"]
            labels_str = "+".join(all_details[best_idx]["content_labels"]) or "general"
            all_details[best_idx]["reason"] = f"mmr(rel={rel:.2f}, labels={labels_str})"
            selected_indices.append(best_idx)
            selected_texts.append(window_texts[best_idx])
            candidates.remove(best_idx)

    # Mark unselected with reason
    for d in all_details:
        if not d["selected"] and not d["reason"]:
            d["reason"] = "below_budget"

    selected_details = [d for d in all_details if d["selected"]]
    selected_details.sort(key=lambda d: d["index"])
    return all_details, selected_details


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

WINDOW_SYSTEM = """\
You are a close-reading analyst. Extract key content from a book excerpt.
Rules:
- Only state what is explicitly in the text. No interpretation beyond what is written.
- Use specific names, numbers, examples from the text.
- Cite page numbers as [p.N] when present in the source.
- Be concise but thorough: cover all major points in the excerpt."""

WINDOW_PROMPT = """\
Excerpt from "{title}" by {author}
Section: {section_label} ({section_type})
Pages: {page_range}
Window {window_idx} of {total_windows} (selected for summarization) | Chunks: {chunk_ids}

--- TEXT ---
{text}
--- END TEXT ---

Extract from this excerpt ONLY:

**Key claims**: The specific arguments or assertions made. State the claim, not the topic.

**Evidence & examples**: Named people, studies, statistics, anecdotes with concrete details.

**Frameworks/concepts**: Any named or described framework, model, or defined term.

**Actionable points**: Specific advice or prescriptions the author gives.

Do NOT add generic commentary. Only report what this specific excerpt contains."""

SECTION_SYSTEM = """\
You are a book analyst specializing in faithful, grounded summarization.
Absolute rules:
- Every claim in your output must be traceable to a window summary below.
- If a point appears in only one window, present it as that window's contribution.
- If a point appears in multiple windows, state it once and note the convergence.
- Do NOT infer themes the windows do not contain.
- Do NOT use phrases like "explores the concept of" or "discusses the importance of."
  State what the author argues and why, using the evidence from the windows.
- If the windows contain examples, name them. If they contain numbers, cite them.
- If the windows do NOT contain enough evidence for a subsection, say so rather than
  filling in with vague language."""

SECTION_PROMPT = """\
Synthesize these {n_selected} selected window summaries into a faithful summary \
of section "{section_label}" from "{title}" by {author}.
Section type: {section_type} | Pages: {page_range}
Selected from {n_total} total windows covering {n_chunks} retrieval chunks.

--- WINDOW SUMMARIES ---
{window_summaries}
--- END ---

Produce exactly these subsections:

### Core Argument
One paragraph: what is the author's central claim in this section? State it as a \
specific, falsifiable assertion — not a topic label. What key evidence or logic \
supports it? Cite window numbers.

### Key Supporting Ideas
3–5 bullet points. Each must:
- State a distinct idea (not a restatement of the core argument)
- Include at least one specific detail from a window
- End with [Window N] citation

### Strongest Examples
The most compelling named examples, case studies, or anecdotes. For each:
- Who or what is involved
- What happened or was found
- What it demonstrates
- [Window N]
If none present in windows, write "No concrete examples found in selected windows."

### Frameworks & Mental Models
Named frameworks with one-sentence definitions. Only include if explicitly \
present in the window summaries. If none, write "None explicitly named."

### Actionable Takeaways
Specific actions grounded in the windows. Each must reference a window. \
If none, write "No specific actions prescribed in selected windows."

### Notable Quotes
1–2 verbatim quotes with page numbers, only if present in the windows. \
If none, write "None extracted."

### Source
Selected windows: {selected_window_ids} of {n_total} | Chunk IDs: {chunk_ids}"""

BOOK_SYSTEM = """\
You are a book summary synthesiser producing a polished, high-fidelity summary.
Rules:
- Every sentence must be traceable to at least one section analysis below.
- Do NOT introduce themes, concepts, or claims not present in the section analyses.
- Cross-cutting themes must cite specific sections and specific examples.
- Do NOT use vague thematic labels ("balance", "intentionality", "mindfulness")
  unless you can point to a specific definition or argument in the sections.
- If two sections make overlapping points, merge them into one crisp statement.
- Prefer specificity and grounding over breadth and vagueness."""

BOOK_PROMPT = """\
Synthesise the section analyses below into a structured summary of
"{title}" by {author}.

Section Analyses:
{section_summaries}

Output these sections:

## Book Metadata
- **Title**: {title}
- **Author**: {author}
- **Domain/Category**: (infer from content)
- **Central thesis**: One paragraph stating the author's core argument as a \
specific, falsifiable claim. Support it with the strongest piece of evidence \
from the section analyses. Do not use vague phrasing.

## Section-by-Section Summary
For each analysed section, one focused paragraph containing:
- The section's core claim (the actual assertion, not "this section discusses X")
- The strongest specific example from that section (name it)
- Any framework introduced (by name)
- One actionable takeaway if present
Each section MUST contribute at least one idea not found in other sections. \
If two sections overlap, merge the overlapping point into one and note both sections.

## Cross-Cutting Themes
3–5 themes that appear across multiple sections. For each:
- **Theme**: a specific descriptive name (not a generic word)
- **Evidence**: cite at least 2 specific examples or arguments from different sections
Do not list a theme unless you can cite concrete evidence from at least 2 sections.

## Frameworks & Mental Models
For each: **Name** — one-sentence definition with the section it comes from. \
Only include frameworks explicitly named in the section analyses.

## Top Actionable Items
5–8 distinct, specific actions ranked by impact. \
Each must be concrete enough to act on without re-reading the book. \
Cite the section each comes from.

## Connections & Building Blocks
Ideas from this book that could combine with concepts from other domains:
- **Idea**: brief description (from section X)
- **Pairs with**: specific domain/concept
- **Why**: what the combination produces
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse_book(
    book_id: str,
    config: RAGConfig,
    mode: str = "default",
    quality: str | None = None,
    section_filter: str | None = None,
    include_types: list[str] | None = None,
    exclude_types: list[str] | None = None,
    force: bool = False,
    verify: bool | None = None,
) -> dict:
    """Quality-focused windowed book analysis with caching and resume."""
    retrieval = Retrieval(config)
    llm = LLMClient(config.generation)
    summ_cfg = config.summarization

    quality = quality or "default"
    budget = summ_cfg.budget_for(quality)
    window_size = summ_cfg.window_size
    window_overlap = summ_cfg.window_overlap
    strategy = summ_cfg.selection_strategy
    always_first = summ_cfg.always_include_first
    always_last = summ_cfg.always_include_last
    mmr_lambda = summ_cfg.mmr_lambda

    book_info = retrieval.store.get_book_info(book_id)
    if not book_info:
        raise ValueError(f"Book '{book_id}' not found. Ingest it first.")

    title = book_info["title"]
    author = book_info["author"]
    sections_meta = book_info.get("sections", [])
    chapters = retrieval.get_book_chapters(book_id)

    allowed_types = resolve_section_filter(mode, include_types, exclude_types)
    meta_by_name: dict[str, dict] = {s["name"]: s for s in sections_meta}

    cache = SummaryCache(summ_cfg.cache_dir, book_id, config.generation.model, quality)
    if force:
        cache.clear()
        print("  Cache cleared (--force)")

    print(f"  Quality={quality}, budget={budget} windows/section, "
          f"strategy={strategy}, mmr_lambda={mmr_lambda}")

    section_results: list[dict] = []
    selection_detail: dict[str, list[dict]] = {}
    total_llm_calls = 0
    section_keys: list[str] = []

    for ch in chapters:
        meta = meta_by_name.get(ch, {})
        stype = meta.get("section_type", "unknown")
        start_page = meta.get("start_page", "?")
        end_page = meta.get("end_page", "?")
        page_range = f"{start_page}-{end_page}"
        parent = meta.get("parent", "")

        if stype not in allowed_types:
            print(f"  ⊘ SKIP  {ch}  (type={stype})")
            continue

        if section_filter and ch != section_filter:
            print(f"  ⊘ SKIP  {ch}  (not target section)")
            continue

        chunks = retrieval.get_chapter_chunks(book_id, ch)
        if not chunks:
            print(f"  ⊘ SKIP  {ch}  (no chunks)")
            continue

        chunks.sort(key=lambda c: c.get("chunk_index", 0))
        chunk_ids = [c["id"] for c in chunks]

        # --- Build all windows, score and select ---
        all_windows = build_windows(chunks, window_size, window_overlap)
        all_detail, selected_detail = select_windows(
            all_windows, budget, ch,
            always_first=always_first, always_last=always_last,
            strategy=strategy, mmr_lambda=mmr_lambda,
        )

        n_selected = len(selected_detail)
        n_skipped = len(all_detail) - n_selected
        print(f"  ▶ {ch}  ({len(chunks)} chunks → {len(all_windows)} windows "
              f"→ {n_selected} selected, {n_skipped} skipped)")

        # Save per-window detail for selection_detail.json (strip chunk objects)
        selection_detail[ch] = [
            {
                "index": d["index"],
                "selected": d["selected"],
                "reason": d["reason"],
                "content_labels": d["content_labels"],
                "scores": d["scores"],
                "chunk_ids": d["chunk_ids"],
            }
            for d in all_detail
        ]

        # --- Phase 1: Summarize selected windows (cached) ---
        window_summaries: list[dict] = []
        window_keys: list[str] = []
        for sd in selected_detail:
            wi = sd["index"]
            w_chunks = sd["chunks"]
            w_chunk_ids = sd["chunk_ids"]
            w_key = cache.window_key(w_chunk_ids)
            window_keys.append(w_key)
            score = sd["scores"]["composite"]
            labels = sd["content_labels"]

            cached = cache.get_window(w_key)
            if cached is not None:
                window_summaries.append({
                    "window_idx": wi, "chunk_ids": w_chunk_ids,
                    "summary": cached, "score": score,
                    "labels": labels, "cached": True,
                })
                w_pages = f"{w_chunks[0].get('page_range', '?').split('-')[0]}-{w_chunks[-1].get('page_range', '?').split('-')[-1]}"
                print(f"    ✓ w{wi+1}/{len(all_windows)}  CACHED  "
                      f"score={score:.3f}  [{'+'.join(labels) or 'general'}]  p.{w_pages}")
                continue

            w_text = "\n\n".join(c.get("text", "") for c in w_chunks)
            w_pages = f"{w_chunks[0].get('page_range', '?').split('-')[0]}-{w_chunks[-1].get('page_range', '?').split('-')[-1]}"

            prompt = WINDOW_PROMPT.format(
                title=title, author=author,
                section_label=ch, section_type=stype,
                page_range=w_pages,
                window_idx=wi + 1, total_windows=len(all_windows),
                chunk_ids=", ".join(w_chunk_ids),
                text=w_text,
            )
            summary = llm.generate(prompt, system=WINDOW_SYSTEM)
            total_llm_calls += 1

            cache.put_window(w_key, summary, w_chunk_ids)
            window_summaries.append({
                "window_idx": wi, "chunk_ids": w_chunk_ids,
                "summary": summary, "score": score,
                "labels": labels, "cached": False,
            })
            print(f"    ✓ w{wi+1}/{len(all_windows)}  "
                  f"score={score:.3f}  [{'+'.join(labels) or 'general'}]  p.{w_pages}")

        # --- Phase 2: Synthesize section summary (cached) ---
        s_key = cache.section_key(ch, window_keys)
        section_keys.append(s_key)

        cached_section = cache.get_section(s_key)
        if cached_section is not None:
            print(f"    ✓ section synthesis  (CACHED)")
            section_summary = cached_section
        else:
            combined_windows = "\n\n---\n\n".join(
                f"[Window {ws['window_idx']+1}, "
                f"labels={'+'.join(ws['labels']) or 'general'}, "
                f"chunks: {', '.join(ws['chunk_ids'])}]\n{ws['summary']}"
                for ws in window_summaries
            )
            section_prompt = SECTION_PROMPT.format(
                title=title, author=author,
                section_label=ch, section_type=stype,
                page_range=page_range,
                n_selected=n_selected, n_total=len(all_windows),
                n_chunks=len(chunks),
                window_summaries=combined_windows,
                selected_window_ids=", ".join(str(ws["window_idx"]+1) for ws in window_summaries),
                chunk_ids=", ".join(chunk_ids),
            )
            section_summary = llm.generate(section_prompt, system=SECTION_SYSTEM)
            total_llm_calls += 1
            cache.put_section(s_key, section_summary, ch, window_keys, meta={
                "quality": quality, "budget": budget,
                "n_windows_total": len(all_windows), "n_windows_selected": n_selected,
                "n_chunks": len(chunks), "strategy": strategy,
            })
            print(f"    ✓ section synthesis done")

        section_results.append({
            "section": ch,
            "section_type": stype,
            "page_range": page_range,
            "parent": parent,
            "n_chunks": len(chunks),
            "n_windows_total": len(all_windows),
            "n_windows_selected": n_selected,
            "chunk_ids": chunk_ids,
            "selected_windows": [
                {"window_idx": ws["window_idx"], "score": ws["score"],
                 "chunk_ids": ws["chunk_ids"], "labels": ws["labels"],
                 "cached": ws["cached"]}
                for ws in window_summaries
            ],
            "window_summaries": window_summaries,
            "section_summary": section_summary,
        })

    if not section_results:
        print("\nNo sections matched the filter. Try --mode full or check --section name.")
        return {"section_results": [], "book_summary": ""}

    # --- Phase 3: Book-level synthesis ---
    book_summary = ""
    if section_filter:
        print(f"\n  ⊘ Skipping book synthesis (single-section mode)")
    else:
        b_key = cache.book_key(section_keys)
        cached_book = cache.get_book(b_key)

        if cached_book is not None:
            print(f"\n  ▶ Book summary  (CACHED)")
            book_summary = cached_book
        else:
            print(f"\n  ▶ Synthesizing book summary from {len(section_results)} sections …")
            all_section_text = "\n\n---\n\n".join(
                f"### {sr['section']} ({sr['section_type']}, p.{sr['page_range']})\n{sr['section_summary']}"
                for sr in section_results
            )
            book_summary = llm.generate(
                BOOK_PROMPT.format(
                    title=title, author=author, section_summaries=all_section_text,
                ),
                system=BOOK_SYSTEM,
            )
            total_llm_calls += 1
            cache.put_book(b_key, book_summary, meta={
                "quality": quality, "n_sections": len(section_results),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    # --- Save results ---
    results_dir = Path(config.storage.results_directory) / book_id
    results_dir.mkdir(parents=True, exist_ok=True)

    insights_parts: list[str] = []
    for sr in section_results:
        parent_str = f" (in {sr['parent']})" if sr.get("parent") else ""
        sel_info = ", ".join(
            f"w{w['window_idx']+1}[{'+'.join(w['labels']) or 'gen'}]"
            for w in sr["selected_windows"]
        )
        header = (
            f"## {sr['section']}{parent_str}\n"
            f"**Type:** {sr['section_type']} | "
            f"**Pages:** {sr['page_range']} | "
            f"**Chunks:** {sr['n_chunks']} | "
            f"**Windows:** {sr['n_windows_selected']}/{sr['n_windows_total']} selected\n"
            f"**Selected:** {sel_info}\n"
        )
        insights_parts.append(header + "\n" + sr["section_summary"])
    (results_dir / "chapter_insights.md").write_text("\n\n---\n\n".join(insights_parts))

    window_data: dict[str, list[dict]] = {}
    for sr in section_results:
        window_data[sr["section"]] = [
            {"window": ws["window_idx"], "chunk_ids": ws["chunk_ids"],
             "score": ws["score"], "labels": ws["labels"],
             "cached": ws["cached"], "summary": ws["summary"]}
            for ws in sr["window_summaries"]
        ]
    (results_dir / "window_summaries.json").write_text(json.dumps(window_data, indent=2))

    if book_summary:
        (results_dir / "book_summary.md").write_text(book_summary)

    (results_dir / "chunk_map.json").write_text(json.dumps(
        {sr["section"]: sr["chunk_ids"] for sr in section_results}, indent=2
    ))

    (results_dir / "selection_detail.json").write_text(json.dumps(selection_detail, indent=2))

    summary_meta = {
        "book_id": book_id,
        "quality": quality,
        "mode": mode,
        "budget_per_section": budget,
        "strategy": strategy,
        "mmr_lambda": mmr_lambda,
        "model": config.generation.model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_llm_calls": total_llm_calls,
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "sections": [
            {
                "name": sr["section"],
                "type": sr["section_type"],
                "pages": sr["page_range"],
                "chunks": sr["n_chunks"],
                "windows_total": sr["n_windows_total"],
                "windows_selected": sr["n_windows_selected"],
                "selected_window_indices": [w["window_idx"] for w in sr["selected_windows"]],
                "selected_labels": [w["labels"] for w in sr["selected_windows"]],
            }
            for sr in section_results
        ],
    }
    (results_dir / "summary_meta.json").write_text(json.dumps(summary_meta, indent=2))

    stats = cache.stats()
    print(f"\n  ✓ Done. {total_llm_calls} LLM calls, "
          f"{stats['hits']} cache hits, {stats['misses']} misses. "
          f"Results → {results_dir}/")

    # --- Optional verification ---
    do_verify = verify if verify is not None else summ_cfg.verify_by_default
    if do_verify:
        from .critic import verify_book_summary
        print("\n  Running verification …")
        verify_book_summary(book_id, config)

    return {"section_results": section_results, "book_summary": book_summary}
