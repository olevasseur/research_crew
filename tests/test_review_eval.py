"""Tests for review_eval and its private helpers.

All tests are offline/local: no Ollama or LLM calls are made.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag.evaluate import (
    _count_eval_items,
    _parse_eval_scores,
    review_eval,
)

# ---------------------------------------------------------------------------
# Minimal fake config (avoids loading rag_config.yaml from disk)
# ---------------------------------------------------------------------------

def _make_config(results_dir: Path):
    return SimpleNamespace(
        storage=SimpleNamespace(results_directory=str(results_dir))
    )


# ---------------------------------------------------------------------------
# Sample evaluation markdown fixtures
# ---------------------------------------------------------------------------

EVAL_BOOK_MD = textwrap.dedent("""\
    # Book Summary Evaluation: test-book

    ### Well-Supported Themes
    - Theme A (Chapter 1)
    - Theme B (Chapter 2, Chapter 3)

    ### Under-Supported Themes
    - Theme C only mentioned in Chapter 2

    ### Missed Content
    - Detail X not covered

    ### Repetition Issues
    - Intentionality repeated twice

    ### Over-Generalizations
    - Uses "intentionality" vaguely

    ### Scores

    - **Coverage**: 80
    - **Faithfulness**: 90
    - **Quality**: 4

    ### Recommendations
    1. Add more specific examples
    2. Condense repetition
    3. Use precise language
""")

EVAL_SECTION_MD = textwrap.dedent("""\
    # Evaluation: Chapter 1: A Lopsided Arms Race

    ### Well-Supported Points
    1. Social media changes were unexpected. (Window 1)
    2. Technology is designed to be addictive. (Window 2)
    3. Behavioral addictions recognized. (Window 4)

    ### Under-Supported Points
    * "Lopsided arms race" lacks detail

    ### Missed Content
    * Techno-apologists concept missing
    * Metastasizing not discussed

    ### Over-Generalizations
    * Understates vs. "not predicted"
    * Intentional addiction design overstated

    ### Scores

    * **Coverage**: 60% of important window content is reflected
    * **Faithfulness**: 70% of summary claims are well-grounded
    * **Quality**: 3/5

    ### Recommendations
    1. More examples from Window 4
    2. Explore "lopsided arms race" deeper
""")


# ---------------------------------------------------------------------------
# _parse_eval_scores
# ---------------------------------------------------------------------------

def test_parse_scores_book():
    scores = _parse_eval_scores(EVAL_BOOK_MD)
    assert scores == {"coverage": "80", "faithfulness": "90", "quality": "4"}


def test_parse_scores_section_with_percent():
    # Quality line reads "3/5"; coverage/faithfulness lines have a trailing "%" word
    scores = _parse_eval_scores(EVAL_SECTION_MD)
    assert scores == {"coverage": "60", "faithfulness": "70", "quality": "3/5"}


def test_parse_scores_missing():
    assert _parse_eval_scores("No scores here.") is None


# ---------------------------------------------------------------------------
# _count_eval_items
# ---------------------------------------------------------------------------

def test_count_items_well_supported_themes():
    assert _count_eval_items(EVAL_BOOK_MD, "Well-Supported Themes") == 2


def test_count_items_recommendations_book():
    assert _count_eval_items(EVAL_BOOK_MD, "Recommendations") == 3


def test_count_items_well_supported_points():
    assert _count_eval_items(EVAL_SECTION_MD, "Well-Supported Points") == 3


def test_count_items_missed_content_section():
    assert _count_eval_items(EVAL_SECTION_MD, "Missed Content") == 2


def test_count_items_missing_heading():
    assert _count_eval_items(EVAL_BOOK_MD, "Nonexistent Section") is None


# ---------------------------------------------------------------------------
# review_eval — no artifacts case
# ---------------------------------------------------------------------------

def test_review_eval_no_artifacts(tmp_path):
    # evaluations/ dir does not exist at all
    config = _make_config(tmp_path)
    out = review_eval("unknown-book", config)
    assert "No evaluation artifacts found" in out
    assert "unknown-book" in out
    assert "python rag_cli.py evaluate unknown-book" in out


def test_review_eval_empty_evaluations_dir(tmp_path):
    # evaluations/ dir exists but is empty
    (tmp_path / "my-book" / "evaluations").mkdir(parents=True)
    config = _make_config(tmp_path)
    out = review_eval("my-book", config)
    assert "No evaluation artifacts found" in out
    assert "my-book" in out


# ---------------------------------------------------------------------------
# review_eval — success case
# ---------------------------------------------------------------------------

@pytest.fixture()
def eval_dir(tmp_path):
    book_dir = tmp_path / "test-book" / "evaluations"
    book_dir.mkdir(parents=True)
    (book_dir / "eval_book.md").write_text(EVAL_BOOK_MD)
    (book_dir / "eval_section_Chapter_1.md").write_text(EVAL_SECTION_MD)
    return tmp_path


def test_review_eval_header(eval_dir):
    config = _make_config(eval_dir)
    out = review_eval("test-book", config)
    assert "Evaluation summary for 'test-book'" in out
    assert "2 file(s)" in out


def test_review_eval_book_scores(eval_dir):
    config = _make_config(eval_dir)
    out = review_eval("test-book", config)
    assert "Coverage=80" in out
    assert "Faithfulness=90" in out
    assert "Quality=4" in out


def test_review_eval_section_scores(eval_dir):
    config = _make_config(eval_dir)
    out = review_eval("test-book", config)
    assert "Coverage=60" in out
    assert "Faithfulness=70" in out
    assert "Quality=3/5" in out


def test_review_eval_item_counts(eval_dir):
    config = _make_config(eval_dir)
    out = review_eval("test-book", config)
    # Book eval: Well-supported=2, Recs=3
    assert "Well-supported=2" in out
    assert "Recs=3" in out
    # Section eval: Well-supported=3, Missed=2
    assert "Well-supported=3" in out
    assert "Missed=2" in out


def test_review_eval_titles(eval_dir):
    config = _make_config(eval_dir)
    out = review_eval("test-book", config)
    assert "Book Summary Evaluation: test-book" in out
    assert "Evaluation: Chapter 1: A Lopsided Arms Race" in out


def test_review_eval_file_paths_present(eval_dir):
    config = _make_config(eval_dir)
    out = review_eval("test-book", config)
    assert "eval_book.md" in out
    assert "eval_section_Chapter_1.md" in out
    assert "Detail files:" in out


# ---------------------------------------------------------------------------
# review_eval — tolerance of partial / inconsistent artifacts
# ---------------------------------------------------------------------------

EVAL_PARTIAL_MD = textwrap.dedent("""\
    # Partial Evaluation

    Some free-form text without structured headings or score lines.
    This might happen if an LLM produced non-standard output.
""")


def test_review_eval_partial_artifact(tmp_path):
    """review_eval must not crash when an eval file has no scores or item sections."""
    book_dir = tmp_path / "partial-book" / "evaluations"
    book_dir.mkdir(parents=True)
    (book_dir / "eval_book.md").write_text(EVAL_PARTIAL_MD)
    config = _make_config(tmp_path)
    out = review_eval("partial-book", config)
    assert "Evaluation summary for 'partial-book'" in out
    assert "Scores: not found" in out
    # Should not raise; item counts section simply omitted when nothing matches
    assert "Partial Evaluation" in out
