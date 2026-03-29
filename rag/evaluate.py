"""Summary quality evaluation: LLM-based faithfulness assessment.

Evaluates section and book summaries against their source window summaries
to identify well-supported points, gaps, and over-generalizations.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .config import RAGConfig
from .llm import LLMClient


# ---------------------------------------------------------------------------
# Evaluation prompts
# ---------------------------------------------------------------------------

EVAL_SECTION_SYSTEM = """\
You are a summary quality auditor. Your job is to assess whether a section \
summary faithfully represents the window summaries it was built from. \
Be specific, concise, and constructive. Cite window numbers."""

EVAL_SECTION_PROMPT = """\
Evaluate whether this section summary faithfully represents its source \
window summaries.

## Section Summary
{section_summary}

## Source Window Summaries (the summary was synthesized from these)
{window_summaries}

Produce exactly these subsections:

### Well-Supported Points
List specific claims from the section summary that are clearly present in \
the window summaries. For each, cite the window number(s).

### Under-Supported Points
List claims in the section summary that are weakly grounded or not clearly \
present in any window. Explain what is missing.

### Missed Content
List important ideas present in the window summaries that the section summary \
does NOT cover. These are potential gaps.

### Over-Generalizations
List places where the section summary overstates, adds interpretation not in \
the windows, or uses vague language where the windows are specific.

### Scores
- **Coverage**: What percentage of important window content is reflected? (0-100)
- **Faithfulness**: What percentage of summary claims are well-grounded? (0-100)
- **Quality**: Overall rating (1-5, where 5 = excellent faithful summary)

### Recommendations
1-3 specific suggestions for improving this section summary."""

EVAL_BOOK_SYSTEM = """\
You are a book summary quality auditor. Assess whether a book summary \
faithfully represents its section summaries without inventing themes, \
repeating points, or using vague filler. Be specific."""

EVAL_BOOK_PROMPT = """\
Evaluate this book summary against the section summaries it was built from.

## Book Summary
{book_summary}

## Section Summaries
{section_summaries}

Produce exactly these subsections:

### Well-Supported Themes
Cross-cutting themes or claims in the book summary that are clearly grounded \
in multiple section summaries. Cite the sections.

### Under-Supported Themes
Themes or claims in the book summary that lack clear grounding in the section \
summaries.

### Missed Content
Important ideas from the section summaries that the book summary does not \
adequately cover.

### Repetition Issues
Places where the book summary essentially says the same thing more than once.

### Over-Generalizations
Places where the book summary uses vague language ("balance", "intentionality") \
where the section summaries are more specific.

### Scores
- **Coverage**: (0-100) how much section content is represented
- **Faithfulness**: (0-100) how well-grounded are the book summary's claims
- **Quality**: (1-5)

### Recommendations
1-3 specific suggestions for improvement."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_section(
    book_id: str,
    section_name: str,
    config: RAGConfig,
) -> str:
    """Evaluate a single section's summary against its window summaries."""
    results_dir = Path(config.storage.results_directory) / book_id
    ws_path = results_dir / "window_summaries.json"
    insights_path = results_dir / "chapter_insights.md"

    if not ws_path.exists() or not insights_path.exists():
        return f"No summaries found for '{book_id}'. Run summarize first."

    window_data = json.loads(ws_path.read_text())
    if section_name not in window_data:
        available = ", ".join(window_data.keys())
        return f"Section '{section_name}' not found. Available: {available}"

    # Extract section summary from chapter_insights.md
    insights_text = insights_path.read_text()
    section_summary = _extract_section_from_insights(insights_text, section_name)
    if not section_summary:
        return f"Could not find section summary for '{section_name}' in chapter_insights.md"

    # Build window summaries text
    windows = window_data[section_name]
    window_text = "\n\n---\n\n".join(
        f"[Window {w.get('window', '?') + 1 if isinstance(w.get('window'), int) else w.get('window', '?')}, "
        f"labels={'+'.join(w.get('labels', []))}, "
        f"chunks: {', '.join(w.get('chunk_ids', []))}]\n{w.get('summary', '')}"
        for w in windows
    )

    llm = LLMClient(config.generation)
    prompt = EVAL_SECTION_PROMPT.format(
        section_summary=section_summary,
        window_summaries=window_text,
    )
    evaluation = llm.generate(prompt, system=EVAL_SECTION_SYSTEM)

    # Save evaluation
    eval_dir = results_dir / "evaluations"
    eval_dir.mkdir(exist_ok=True)
    safe_name = section_name.replace("/", "_").replace(" ", "_")[:60]
    (eval_dir / f"eval_section_{safe_name}.md").write_text(
        f"# Evaluation: {section_name}\n\n{evaluation}"
    )

    return evaluation


def evaluate_book(
    book_id: str,
    config: RAGConfig,
) -> str:
    """Evaluate the book summary against all section summaries."""
    results_dir = Path(config.storage.results_directory) / book_id
    book_path = results_dir / "book_summary.md"
    insights_path = results_dir / "chapter_insights.md"

    if not book_path.exists():
        return f"No book summary found for '{book_id}'. Run summarize first."
    if not insights_path.exists():
        return f"No chapter insights found for '{book_id}'. Run summarize first."

    book_summary = book_path.read_text()
    section_summaries = insights_path.read_text()

    llm = LLMClient(config.generation)
    prompt = EVAL_BOOK_PROMPT.format(
        book_summary=book_summary,
        section_summaries=section_summaries,
    )
    evaluation = llm.generate(prompt, system=EVAL_BOOK_SYSTEM)

    eval_dir = results_dir / "evaluations"
    eval_dir.mkdir(exist_ok=True)
    (eval_dir / "eval_book.md").write_text(
        f"# Book Summary Evaluation: {book_id}\n\n{evaluation}"
    )

    return evaluation


def review_eval(book_id: str, config: RAGConfig) -> str:
    """Return a concise human-readable summary of existing evaluation artifacts.

    Reads markdown files from ``evaluations/`` without calling the LLM.
    If no artifacts exist, returns an actionable message telling the user how
    to generate them.
    """
    results_dir = Path(config.storage.results_directory) / book_id
    eval_dir = results_dir / "evaluations"

    if not eval_dir.exists() or not any(eval_dir.glob("*.md")):
        return (
            f"No evaluation artifacts found for '{book_id}'.\n"
            f"Run:  python rag_cli.py evaluate {book_id}\n"
            f"      python rag_cli.py evaluate {book_id} --section \"<section name>\""
        )

    eval_files = sorted(eval_dir.glob("*.md"))
    lines: list[str] = [
        f"Evaluation summary for '{book_id}'  ({len(eval_files)} file(s))",
        "=" * 60,
    ]

    for path in eval_files:
        text = path.read_text()

        # Title from H1 header, fallback to filename stem
        title_m = re.match(r"^#\s+(.+)", text)
        title = title_m.group(1).strip() if title_m else path.stem

        scores = _parse_eval_scores(text)
        score_str = (
            f"Coverage={scores['coverage']}  "
            f"Faithfulness={scores['faithfulness']}  "
            f"Quality={scores['quality']}"
            if scores
            else "Scores: not found"
        )

        item_parts: list[str] = []
        for label, headings in _EVAL_SECTION_LABELS:
            for heading in headings:
                n = _count_eval_items(text, heading)
                if n is not None:
                    if n > 0:
                        item_parts.append(f"{label}={n}")
                    break  # use first matching heading variant

        lines.append("")
        lines.append(title)
        lines.append(f"  Scores  {score_str}")
        if item_parts:
            lines.append(f"  Items   {', '.join(item_parts)}")
        lines.append(f"  File    {path}")

    lines += ["", "=" * 60, f"Detail files: {eval_dir}"]
    return "\n".join(lines)


# Heading variants: (display label, [book-level heading, section-level heading])
_EVAL_SECTION_LABELS: list[tuple[str, list[str]]] = [
    ("Well-supported", ["Well-Supported Themes", "Well-Supported Points"]),
    ("Under-supported", ["Under-Supported Themes", "Under-Supported Points"]),
    ("Missed", ["Missed Content"]),
    ("Repetition", ["Repetition Issues"]),
    ("Over-gen", ["Over-Generalizations"]),
    ("Recs", ["Recommendations"]),
]


def _parse_eval_scores(text: str) -> dict[str, str] | None:
    """Extract Coverage/Faithfulness/Quality values from an evaluation markdown."""
    coverage = re.search(r"\*\*Coverage\*\*[:\s]+(\d+)", text)
    faithfulness = re.search(r"\*\*Faithfulness\*\*[:\s]+(\d+)", text)
    quality = re.search(r"\*\*Quality\*\*[:\s]+(\d+(?:/\d+)?)", text)
    if not (coverage and faithfulness and quality):
        return None
    return {
        "coverage": coverage.group(1),
        "faithfulness": faithfulness.group(1),
        "quality": quality.group(1),
    }


def _count_eval_items(text: str, section_heading: str) -> int | None:
    """Count bullet/numbered items under a ``### Heading`` in evaluation markdown."""
    pattern = rf"###\s+{re.escape(section_heading)}\s*\n(.*?)(?=\n###|\Z)"
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        return None
    body = m.group(1)
    items = re.findall(r"^\s*(?:[-*]|\d+\.)\s+\S", body, re.MULTILINE)
    return len(items)


def _extract_section_from_insights(insights_text: str, section_name: str) -> str | None:
    """Extract a single section's content from chapter_insights.md."""
    marker = f"## {section_name}"
    start = insights_text.find(marker)
    if start == -1:
        return None
    # Find the next section marker or end
    next_marker = insights_text.find("\n---\n", start + len(marker))
    if next_marker == -1:
        return insights_text[start:]
    return insights_text[start:next_marker].strip()
