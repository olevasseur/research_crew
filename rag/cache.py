"""Content-aware caching for generated summaries.

Cache layout:
    <cache_dir>/<book_id>/
        window_<hash>.json      — per-window summary
        section_<hash>.json     — per-section summary
        book_<hash>.json        — book-level summary
        manifest.json           — maps section names to their cache hashes

Cache keys are SHA-256 hashes of the inputs that determine the output:
    window:  sorted chunk IDs + model name + prompt version
    section: sorted window hashes + model name + prompt version
    book:    sorted section hashes + model name + prompt version
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROMPT_VERSION = "v2"  # bump when prompts change materially


class SummaryCache:
    """File-backed cache for generated summaries."""

    def __init__(self, cache_dir: str, book_id: str, model: str):
        self._dir = Path(cache_dir) / book_id
        self._dir.mkdir(parents=True, exist_ok=True)
        self._model = model
        self._manifest_path = self._dir / "manifest.json"
        self._manifest: dict = self._load_manifest()
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------
    # Key computation
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_key(*parts: str) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(p.encode())
        return h.hexdigest()[:16]

    def window_key(self, chunk_ids: list[str]) -> str:
        return self._hash_key(
            "window", ",".join(sorted(chunk_ids)), self._model, PROMPT_VERSION,
        )

    def section_key(self, section_name: str, window_keys: list[str]) -> str:
        return self._hash_key(
            "section", section_name, ",".join(sorted(window_keys)),
            self._model, PROMPT_VERSION,
        )

    def book_key(self, section_keys: list[str]) -> str:
        return self._hash_key(
            "book", ",".join(sorted(section_keys)),
            self._model, PROMPT_VERSION,
        )

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------

    def get_window(self, key: str) -> str | None:
        return self._get("window", key)

    def put_window(self, key: str, summary: str, chunk_ids: list[str]) -> None:
        self._put("window", key, {"summary": summary, "chunk_ids": chunk_ids})

    def get_section(self, key: str) -> str | None:
        return self._get("section", key)

    def put_section(self, key: str, summary: str, section_name: str, window_keys: list[str]) -> None:
        self._put("section", key, {
            "summary": summary, "section": section_name, "window_keys": window_keys,
        })
        self._manifest[section_name] = key
        self._save_manifest()

    def get_book(self, key: str) -> str | None:
        return self._get("book", key)

    def put_book(self, key: str, summary: str) -> None:
        self._put("book", key, {"summary": summary})
        self._manifest["__book__"] = key
        self._save_manifest()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "total": self.hits + self.misses}

    def clear(self) -> None:
        """Remove all cached artifacts for this book."""
        import shutil
        if self._dir.exists():
            shutil.rmtree(self._dir)
            self._dir.mkdir(parents=True, exist_ok=True)
        self._manifest = {}
        self._save_manifest()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _path(self, prefix: str, key: str) -> Path:
        return self._dir / f"{prefix}_{key}.json"

    def _get(self, prefix: str, key: str) -> str | None:
        p = self._path(prefix, key)
        if p.exists():
            data = json.loads(p.read_text())
            self.hits += 1
            return data.get("summary")
        self.misses += 1
        return None

    def _put(self, prefix: str, key: str, data: dict) -> None:
        self._path(prefix, key).write_text(json.dumps(data, indent=2))

    def _load_manifest(self) -> dict:
        if self._manifest_path.exists():
            return json.loads(self._manifest_path.read_text())
        return {}

    def _save_manifest(self) -> None:
        self._manifest_path.write_text(json.dumps(self._manifest, indent=2))
