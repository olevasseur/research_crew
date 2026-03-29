"""Sanity checks that TODO.md exists and is non-empty."""

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
TODO_PATH = REPO_ROOT / "TODO.md"


def test_todo_md_exists():
    assert TODO_PATH.exists(), "TODO.md not found at repo root"


def test_todo_md_non_empty():
    assert TODO_PATH.read_text().strip(), "TODO.md is empty (or contains only whitespace)"
