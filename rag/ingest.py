"""Book ingestion pipeline: load → clean → detect chapters → chunk → embed → store."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pdfplumber

from .chunker import Chapter, Chunk, PageText, Section, chunk_pages, detect_structure
from .config import RAGConfig
from .embeddings import OllamaEmbedder
from .store import VectorStore


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "unknown"


def load_file(path: str) -> list[PageText]:
    """Load a file and return page-level text (page numbers are 1-indexed)."""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(path)
    elif suffix in (".txt", ".md"):
        return _load_text(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def clean_text(text: str) -> str:
    """Normalise whitespace and fix common PDF artefacts."""
    text = unicodedata.normalize("NFKC", text)
    # Re-join hyphenated line breaks (e.g. "knowl-\nedge" → "knowledge")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ingest_book(
    file_path: str,
    config: RAGConfig,
    title: str | None = None,
    author: str | None = None,
    book_id: str | None = None,
) -> dict:
    """Full ingestion pipeline.  Returns a summary dict."""
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # 1. Load
    pages = load_file(str(path))
    if not pages:
        raise ValueError("No text extracted from file")

    # 2. Clean
    for page in pages:
        page.text = clean_text(page.text)

    # 3. Auto-detect metadata if not provided
    if not title:
        title = path.stem.replace("_", " ").replace("-", " ").title()
    if not author:
        author = "Unknown"
    if not book_id:
        book_id = slugify(title)

    # 4. Detect structure
    chapters, _ = detect_structure(pages)
    print(f"Detected {len(chapters)} sections:")
    for ch in chapters:
        span = ch.end_page - ch.start_page + 1
        print(f"  p.{ch.start_page:>3d}-{ch.end_page:<3d}  ({span:>3d} pp)  "
              f"[{ch.kind.value:12s}]  {ch.name}")

    # 5. Chunk
    chunks: list[Chunk] = chunk_pages(
        pages, chapters, book_id, title, author, str(path), config.chunking,
    )

    # 6. Store
    embedder = OllamaEmbedder(config.embedding)
    store = VectorStore(config.vectorstore, embedder)

    # Re-ingest from scratch: remove any existing chunks for this book_id first
    existing = store.get_book_info(book_id)
    if existing:
        store.delete_book(book_id)
        print(f"Replacing previous ingestion for '{book_id}' ({existing.get('total_chunks', 0)} chunks removed).")

    print(f"Embedding {len(chunks)} chunks (this may take a while) …")
    store.add_chunks(chunks)

    # 7. Register book metadata
    chapter_names = [c.name for c in chapters]
    store.register_book(book_id, {
        "title": title,
        "author": author,
        "source_path": str(path),
        "total_pages": len(pages),
        "total_chunks": len(chunks),
        "chapters": chapter_names,
    })

    summary = {
        "book_id": book_id,
        "title": title,
        "author": author,
        "total_pages": len(pages),
        "total_chunks": len(chunks),
        "chapters": chapter_names,
    }
    print(f"Ingested '{title}' → {len(chunks)} chunks, {len(chapters)} chapters")
    return summary


def ingest_folder(folder_path: str, config: RAGConfig) -> list[dict]:
    """Ingest every supported file in a folder."""
    folder = Path(folder_path)
    results = []
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() in (".pdf", ".txt", ".md") and not f.name.startswith("."):
            print(f"\n--- Ingesting {f.name} ---")
            try:
                results.append(ingest_book(str(f), config))
            except Exception as e:
                print(f"  ERROR: {e}")
    return results


# ---------------------------------------------------------------------------
# Private loaders
# ---------------------------------------------------------------------------

def _load_pdf(path: str) -> list[PageText]:
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append(PageText(page_number=i + 1, text=text))
    return pages


def _load_text(path: str) -> list[PageText]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    # Treat the whole file as one "page"
    return [PageText(page_number=1, text=text)]
