"""Content-aware caching for generated summaries.

Cache layout:
    <cache_dir>/<book_id>/
        window_<hash>.json      — per-window summary
        section_<hash>.json     — per-section summary
        book_<hash>.json        — book-level summary
        manifest.json           — maps section names to their cache hashes

Cache keys include quality tier so fast/default/thorough don't collide
at the section and book level (window keys are content-addressed and shared).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROMPT_VERSION = "v4"  # bumped: content_diversity selection + improved prompts


class SummaryCache:
    """File-backed cache for generated summaries."""

    def __init__(self, cache_dir: str, book_id: str, model: str, quality: str = "default"):
        self._dir = Path(cache_dir) / book_id
        self._dir.mkdir(parents=True, exist_ok=True)
        self._model = model
        self._quality = quality
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
            self._model, self._quality, PROMPT_VERSION,
        )

    def book_key(self, section_keys: list[str]) -> str:
        return self._hash_key(
            "book", ",".join(sorted(section_keys)),
            self._model, self._quality, PROMPT_VERSION,
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

    def put_section(self, key: str, summary: str, section_name: str,
                    window_keys: list[str], meta: dict | None = None) -> None:
        data = {
            "summary": summary, "section": section_name, "window_keys": window_keys,
        }
        if meta:
            data["meta"] = meta
        self._put("section", key, data)
        self._manifest[section_name] = {"key": key, "quality": self._quality}
        self._save_manifest()

    def get_book(self, key: str) -> str | None:
        return self._get("book", key)

    def put_book(self, key: str, summary: str, meta: dict | None = None) -> None:
        data = {"summary": summary}
        if meta:
            data["meta"] = meta
        self._put("book", key, data)
        self._manifest["__book__"] = {"key": key, "quality": self._quality}
        self._save_manifest()

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def list_cached_files(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {"window": [], "section": [], "book": []}
        for f in self._dir.glob("*.json"):
            if f.name == "manifest.json":
                continue
            for prefix in result:
                if f.name.startswith(prefix + "_"):
                    result[prefix].append(f.name)
        return result

    def get_manifest(self) -> dict:
        return dict(self._manifest)

    def get_raw(self, prefix: str, key: str) -> dict | None:
        p = self._path(prefix, key)
        if p.exists():
            return json.loads(p.read_text())
        return None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "total": self.hits + self.misses}

    def clear(self) -> None:
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
