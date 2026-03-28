"""Tests for fiction_whois chapter-cutoff and character-lookup behaviour.

Covers:
- No future-chapter information leaks when a lower chapter is requested.
- Character lookup is case-insensitive.
- Character lookup matches on aliases as well as canonical name.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------
# Two-chapter fiction_state.json:
#   Chapter 1: Alice (alias "Ali") and Bob; one relationship; one event at City Hall.
#   Chapter 2: Carol added; Alice-Carol relationship added; new event at Police Station.
# This arrangement lets us assert that chapter-1 queries exclude all chapter-2 content.

FIXTURE_STATE: dict = {
    "0": {
        "chapter_index": 0,
        "chapter": "Chapter 1",
        "state": {
            "characters": [
                {
                    "name": "Alice",
                    "aliases": ["Ali"],
                    "role": "detective",
                    "first_chapter": "Chapter 1",
                    "first_chapter_index": 0,
                },
                {
                    "name": "Bob",
                    "aliases": [],
                    "role": "suspect",
                    "first_chapter": "Chapter 1",
                    "first_chapter_index": 0,
                },
            ],
            "relationships": [
                {
                    "character_a": "Alice",
                    "character_b": "Bob",
                    "description": "investigator and suspect",
                    "evidence": "Alice interrogates Bob",
                    "first_chapter": "Chapter 1",
                    "first_chapter_index": 0,
                }
            ],
            "places": [
                {
                    "name": "City Hall",
                    "description": "scene of the crime",
                    "first_chapter": "Chapter 1",
                    "first_chapter_index": 0,
                }
            ],
            "events": [
                {
                    "description": "Alice interrogates Bob at City Hall",
                    "characters": ["Alice", "Bob"],
                    "place": "City Hall",
                    "chapter": "Chapter 1",
                    "chapter_index": 0,
                }
            ],
            "time_markers": [],
        },
    },
    "1": {
        "chapter_index": 1,
        "chapter": "Chapter 2",
        "state": {
            "characters": [
                {
                    "name": "Alice",
                    "aliases": ["Ali"],
                    "role": "detective",
                    "first_chapter": "Chapter 1",
                    "first_chapter_index": 0,
                },
                {
                    "name": "Bob",
                    "aliases": [],
                    "role": "suspect",
                    "first_chapter": "Chapter 1",
                    "first_chapter_index": 0,
                },
                # Carol only appears from Chapter 2 onwards.
                {
                    "name": "Carol",
                    "aliases": [],
                    "role": "witness",
                    "first_chapter": "Chapter 2",
                    "first_chapter_index": 1,
                },
            ],
            "relationships": [
                {
                    "character_a": "Alice",
                    "character_b": "Bob",
                    "description": "investigator and suspect",
                    "evidence": "Alice interrogates Bob",
                    "first_chapter": "Chapter 1",
                    "first_chapter_index": 0,
                },
                # Chapter-2-only relationship — must not appear in chapter-1 queries.
                {
                    "character_a": "Alice",
                    "character_b": "Carol",
                    "description": "detective and witness",
                    "evidence": "Carol gives testimony to Alice",
                    "first_chapter": "Chapter 2",
                    "first_chapter_index": 1,
                },
            ],
            "places": [
                {
                    "name": "City Hall",
                    "description": "scene of the crime",
                    "first_chapter": "Chapter 1",
                    "first_chapter_index": 0,
                },
                # Chapter-2-only place — must not appear in chapter-1 queries for Alice.
                {
                    "name": "Police Station",
                    "description": "where interrogations happen",
                    "first_chapter": "Chapter 2",
                    "first_chapter_index": 1,
                },
            ],
            "events": [
                {
                    "description": "Alice interrogates Bob at City Hall",
                    "characters": ["Alice", "Bob"],
                    "place": "City Hall",
                    "chapter": "Chapter 1",
                    "chapter_index": 0,
                },
                # Chapter-2-only event — must not appear in chapter-1 queries.
                {
                    "description": "Carol gives testimony",
                    "characters": ["Alice", "Carol"],
                    "place": "Police Station",
                    "chapter": "Chapter 2",
                    "chapter_index": 1,
                },
            ],
            "time_markers": [],
        },
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state_file(tmp_path: Path, book_id: str = "test-book") -> Path:
    """Write FIXTURE_STATE to a fiction_state.json in a temp results dir."""
    book_dir = tmp_path / book_id
    book_dir.mkdir(parents=True)
    state_path = book_dir / "fiction_state.json"
    state_path.write_text(json.dumps(FIXTURE_STATE))
    return tmp_path


def _make_config(results_dir: Path):
    """Build a RAGConfig pointing at *results_dir* without loading any YAML."""
    from rag.config import RAGConfig, StorageConfig
    cfg = RAGConfig()
    cfg.storage = StorageConfig(results_directory=str(results_dir))
    return cfg


# ---------------------------------------------------------------------------
# Unit tests for _resolve_snapshot_key
# ---------------------------------------------------------------------------

class TestResolveSnapshotKey:
    """_resolve_snapshot_key should return the highest key whose chapter
    number is <= the requested chapter number."""

    def _call(self, state: dict, chapter: int):
        from rag.fiction import _resolve_snapshot_key
        return _resolve_snapshot_key(state, chapter)

    def test_chapter_1_resolves_to_key_0(self):
        assert self._call(FIXTURE_STATE, 1) == 0

    def test_chapter_2_resolves_to_key_1(self):
        assert self._call(FIXTURE_STATE, 2) == 1

    def test_chapter_100_resolves_to_last_available(self):
        # A chapter number beyond the data should clamp to the last snapshot.
        assert self._call(FIXTURE_STATE, 100) == 1

    def test_chapter_1_does_not_resolve_to_key_1(self):
        # Key 1 is Chapter 2 — must not be returned for chapter=1.
        assert self._call(FIXTURE_STATE, 1) != 1


# ---------------------------------------------------------------------------
# Integration-style tests via fiction_whois (captured stdout)
# ---------------------------------------------------------------------------

class TestFictionWhoisChapterCutoff:
    """Requesting chapter 1 must not reveal chapter-2-only content."""

    def _run(self, tmp_path, chapter: int, character: str, book_id: str = "test-book"):
        results_dir = _make_state_file(tmp_path, book_id)
        cfg = _make_config(results_dir)
        from rag.fiction import fiction_whois
        return fiction_whois(book_id, cfg, chapter_number=chapter, character_name=character)

    def test_chapter1_excludes_carol(self, tmp_path, capsys):
        self._run(tmp_path, chapter=1, character="Alice")
        out = capsys.readouterr().out
        assert "Carol" not in out, "Carol must not appear in chapter-1 output"

    def test_chapter2_includes_carol(self, tmp_path, capsys):
        self._run(tmp_path, chapter=2, character="Alice")
        out = capsys.readouterr().out
        assert "Carol" in out, "Carol should appear in chapter-2 output"

    def test_chapter1_excludes_police_station(self, tmp_path, capsys):
        self._run(tmp_path, chapter=1, character="Alice")
        out = capsys.readouterr().out
        assert "Police Station" not in out, (
            "Police Station (introduced in ch.2) must not appear in chapter-1 output"
        )

    def test_chapter1_excludes_carol_testimony_event(self, tmp_path, capsys):
        self._run(tmp_path, chapter=1, character="Alice")
        out = capsys.readouterr().out
        assert "Carol gives testimony" not in out, (
            "Chapter-2 event must not leak into chapter-1 query"
        )

    def test_chapter1_includes_bob_relationship(self, tmp_path, capsys):
        self._run(tmp_path, chapter=1, character="Alice")
        out = capsys.readouterr().out
        assert "Bob" in out, "Bob (introduced in ch.1) should be visible at chapter 1"

    def test_chapter1_includes_city_hall_place(self, tmp_path, capsys):
        self._run(tmp_path, chapter=1, character="Alice")
        out = capsys.readouterr().out
        assert "City Hall" in out, (
            "City Hall (introduced in ch.1) should appear in chapter-1 output"
        )


class TestFictionWhoisCharacterLookup:
    """Character lookup must be case-insensitive and alias-aware."""

    def _run(self, tmp_path, chapter: int, character: str, book_id: str = "test-book"):
        results_dir = _make_state_file(tmp_path, book_id)
        cfg = _make_config(results_dir)
        from rag.fiction import fiction_whois
        return fiction_whois(book_id, cfg, chapter_number=chapter, character_name=character)

    def test_canonical_name_exact(self, tmp_path, capsys):
        self._run(tmp_path, chapter=1, character="Alice")
        out = capsys.readouterr().out
        assert "Character: Alice" in out

    def test_canonical_name_lowercase(self, tmp_path, capsys):
        """Lookup "alice" (all-lowercase) must resolve to Alice."""
        self._run(tmp_path, chapter=1, character="alice")
        out = capsys.readouterr().out
        assert "Character: Alice" in out, (
            "Case-insensitive lookup on canonical name failed"
        )

    def test_canonical_name_uppercase(self, tmp_path, capsys):
        self._run(tmp_path, chapter=1, character="ALICE")
        out = capsys.readouterr().out
        assert "Character: Alice" in out

    def test_alias_lookup(self, tmp_path, capsys):
        """Lookup by alias "Ali" must resolve to Alice."""
        self._run(tmp_path, chapter=1, character="Ali")
        out = capsys.readouterr().out
        assert "Character: Alice" in out, (
            "Alias lookup failed — 'Ali' should resolve to Alice"
        )

    def test_alias_lookup_case_insensitive(self, tmp_path, capsys):
        """Lookup by alias "ali" (lowercase) must also resolve to Alice."""
        self._run(tmp_path, chapter=1, character="ali")
        out = capsys.readouterr().out
        assert "Character: Alice" in out, (
            "Case-insensitive alias lookup failed — 'ali' should resolve to Alice"
        )

    def test_unknown_character_prints_error(self, tmp_path, capsys):
        self._run(tmp_path, chapter=1, character="Zephyr")
        out = capsys.readouterr().out
        assert "not found" in out.lower(), (
            "Expected an error message for an unknown character"
        )

    def test_unknown_character_lists_known_names(self, tmp_path, capsys):
        self._run(tmp_path, chapter=1, character="Zephyr")
        out = capsys.readouterr().out
        # The error message should hint at who IS known at this chapter
        assert "Alice" in out or "Bob" in out, (
            "Error for unknown character should list known characters"
        )
