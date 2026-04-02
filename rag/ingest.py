"""Book ingestion pipeline: load → clean → detect chapters → chunk → embed → store."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pdfplumber

from .chunker import Chapter, Chunk, PageText, Section, SectionKind, chunk_pages, detect_structure
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
    elif suffix == ".epub":
        pages, *_ = _load_epub(path)
        return pages
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
    epub_sections: list[Section] | None = None
    epub_meta_title: str | None = None
    epub_meta_author: str | None = None
    if path.suffix.lower() == ".epub":
        pages, epub_sections, epub_meta_title, epub_meta_author = _load_epub(str(path))
    else:
        pages = load_file(str(path))
    if not pages:
        raise ValueError("No text extracted from file")

    # 2. Clean
    for page in pages:
        page.text = clean_text(page.text)

    # 3. Auto-detect metadata if not provided
    # EPUB package metadata takes precedence over filename derivation
    if not title:
        title = epub_meta_title or path.stem.replace("_", " ").replace("-", " ").title()
    if not author:
        author = epub_meta_author or "Unknown"
    if not book_id:
        book_id = slugify(title)

    # 4. Detect structure (use EPUB TOC directly when available)
    if epub_sections is not None:
        chapters = epub_sections
        print(f"EPUB TOC: {len(chapters)} sections")
    else:
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

    # 7. Register book metadata (including full section structure)
    section_meta = [
        {
            "name": c.name,
            "section_type": c.section_type,
            "start_page": c.start_page,
            "end_page": c.end_page,
            "parent": c.parent,
            "confidence": c.confidence,
            "detection_reason": c.detection_reason,
        }
        for c in chapters
    ]
    chapter_names = [c.name for c in chapters]
    store.register_book(book_id, {
        "title": title,
        "author": author,
        "source_path": str(path),
        "total_pages": len(pages),
        "total_chunks": len(chunks),
        "chapters": chapter_names,
        "sections": section_meta,
    })

    summary = {
        "book_id": book_id,
        "title": title,
        "author": author,
        "total_pages": len(pages),
        "total_chunks": len(chunks),
        "chapters": chapter_names,
    }
    print(f"Ingested '{title}' → {len(chunks)} chunks, {len(chapters)} sections")
    return summary


def ingest_folder(folder_path: str, config: RAGConfig) -> list[dict]:
    """Ingest every supported file in a folder."""
    folder = Path(folder_path)
    results = []
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() in (".pdf", ".txt", ".md", ".epub") and not f.name.startswith("."):
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


# ---------------------------------------------------------------------------
# EPUB loader (no new dependencies: stdlib zipfile + ET, already-installed bs4)
# ---------------------------------------------------------------------------

def _load_epub(path: str) -> tuple[list[PageText], list[Section] | None]:
    """Load an EPUB file.

    Returns:
        pages:    one PageText per non-empty spine document (page_number is
                  a 1-based sequential index over non-empty spine items).
        sections: Section list built from the EPUB TOC (nav or NCX), or None
                  if no usable TOC is found (caller falls back to detect_structure).
    """
    import warnings
    import zipfile
    from xml.etree import ElementTree as ET
    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

    # Suppress warning about using HTML parser on XHTML (intentional for robustness)
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

    epub_path = Path(path)

    with zipfile.ZipFile(epub_path, "r") as epub:
        # ── Step 1: container.xml → OPF path ───────────────────────────────
        try:
            container_xml = epub.read("META-INF/container.xml")
        except KeyError:
            raise ValueError("Invalid EPUB: missing META-INF/container.xml")

        container = ET.fromstring(container_xml)
        ns_c = "urn:oasis:names:tc:opendocument:xmlns:container"
        rootfile_el = container.find(f".//{{{ns_c}}}rootfile")
        if rootfile_el is None:
            raise ValueError("Invalid EPUB: no rootfile in container.xml")
        opf_path = rootfile_el.get("full-path", "")
        opf_dir = opf_path.rsplit("/", 1)[0] if "/" in opf_path else ""

        # ── Step 2: OPF → manifest + spine ─────────────────────────────────
        opf_xml = epub.read(opf_path)
        opf = ET.fromstring(opf_xml)
        opf_ns = "http://www.idpf.org/2007/opf"
        dc_ns = "http://purl.org/dc/elements/1.1/"

        # Extract package metadata (title / creator) for use as ingest defaults
        _title_el = opf.find(f".//{{{dc_ns}}}title")
        _creator_el = opf.find(f".//{{{dc_ns}}}creator")
        epub_title: str | None = _title_el.text.strip() if _title_el is not None and _title_el.text else None
        epub_author: str | None = _creator_el.text.strip() if _creator_el is not None and _creator_el.text else None

        manifest: dict[str, dict] = {}
        for item in opf.findall(f".//{{{opf_ns}}}item"):
            item_id = item.get("id", "")
            manifest[item_id] = {
                "href": item.get("href", ""),
                "media_type": item.get("media-type", ""),
                "properties": item.get("properties", ""),
            }

        spine_hrefs: list[str] = []
        for itemref in opf.findall(f".//{{{opf_ns}}}itemref"):
            idref = itemref.get("idref", "")
            if idref in manifest:
                spine_hrefs.append(manifest[idref]["href"])

        # ── Step 3: Extract text from each spine document ───────────────────
        def _resolve(href: str) -> str:
            if not opf_dir:
                return href
            parts: list[str] = []
            for seg in f"{opf_dir}/{href}".split("/"):
                if seg == "..":
                    if parts:
                        parts.pop()
                elif seg and seg != ".":
                    parts.append(seg)
            return "/".join(parts)

        def _read_safe(z: zipfile.ZipFile, p: str) -> bytes | None:
            for candidate in (p, p.replace("%20", " ")):
                try:
                    return z.read(candidate)
                except KeyError:
                    pass
            return None

        pages: list[PageText] = []
        spine_idx_to_page: dict[int, int] = {}  # spine position → page_number

        for spine_i, raw_href in enumerate(spine_hrefs):
            data = _read_safe(epub, _resolve(raw_href)) or _read_safe(epub, raw_href)
            if data is None:
                continue

            try:
                soup = BeautifulSoup(data, "lxml")
            except Exception:
                soup = BeautifulSoup(data, "html.parser")

            for tag in soup.find_all(["nav", "script", "style", "head"]):
                tag.decompose()

            raw_text = soup.get_text(separator="\n")
            lines = [ln for ln in raw_text.splitlines() if ln.strip()]
            text = "\n".join(lines)
            if not text.strip():
                continue

            page_num = len(pages) + 1
            pages.append(PageText(page_number=page_num, text=text))
            spine_idx_to_page[spine_i] = page_num

        if not pages:
            raise ValueError("EPUB contained no extractable text in the spine")

        # ── Step 4: Build sections from TOC ────────────────────────────────
        sections = _parse_epub_toc(
            epub, opf, opf_ns, manifest, opf_dir,
            spine_hrefs, spine_idx_to_page, pages,
        )

    return pages, sections, epub_title, epub_author


def _parse_epub_toc(
    epub: "zipfile.ZipFile",  # type: ignore[name-defined]
    opf: "ET.Element",         # type: ignore[name-defined]
    opf_ns: str,
    manifest: dict[str, dict],
    opf_dir: str,
    spine_hrefs: list[str],
    spine_idx_to_page: dict[int, int],
    pages: list[PageText],
) -> list[Section] | None:
    """Extract Section list from EPUB 3 nav or EPUB 2 NCX.

    Returns a Section list, or None if no usable TOC is available.
    """
    import zipfile
    from xml.etree import ElementTree as ET
    from bs4 import BeautifulSoup

    def _resolve(href: str) -> str:
        if not opf_dir:
            return href
        parts: list[str] = []
        for seg in f"{opf_dir}/{href}".split("/"):
            if seg == "..":
                if parts:
                    parts.pop()
            elif seg and seg != ".":
                parts.append(seg)
        return "/".join(parts)

    def _read_safe(z: zipfile.ZipFile, p: str) -> bytes | None:
        for candidate in (p, p.replace("%20", " ")):
            try:
                return z.read(candidate)
            except KeyError:
                pass
        return None

    # Lookup: href basename → spine index (fast matching)
    href_to_spine: dict[str, int] = {}
    for si, sh in enumerate(spine_hrefs):
        href_to_spine[sh] = si
        href_to_spine[sh.rsplit("/", 1)[-1]] = si

    toc_entries: list[tuple[str, str]] = []  # (title, href-without-fragment)

    # ── Try EPUB 3 nav ──────────────────────────────────────────────────────
    nav_item = next(
        (it for it in manifest.values() if "nav" in it.get("properties", "").split()),
        None,
    )
    if nav_item:
        data = _read_safe(epub, _resolve(nav_item["href"]))
        if data:
            # Use html.parser so epub:type attribute name is preserved as-is
            nav_soup = BeautifulSoup(data, "html.parser")
            toc_nav = (
                nav_soup.find("nav", attrs={"epub:type": "toc"})
                or nav_soup.find("nav", attrs={"type": "toc"})
                or nav_soup.find("nav")
            )
            if toc_nav:
                for a_tag in toc_nav.find_all("a", href=True):
                    bare = a_tag["href"].split("#")[0].strip()
                    title = a_tag.get_text(strip=True)
                    if bare and title:
                        toc_entries.append((title, bare))

    # ── Fall back to EPUB 2 NCX ─────────────────────────────────────────────
    if not toc_entries:
        ncx_item = next(
            (it for it in manifest.values()
             if it.get("media_type") == "application/x-dtbncx+xml"),
            None,
        )
        if ncx_item:
            data = _read_safe(epub, _resolve(ncx_item["href"]))
            if data:
                try:
                    ncx_ns = "http://www.daisy.org/z3986/2005/ncx/"
                    ncx_root = ET.fromstring(data)
                    for nav_point in ncx_root.findall(f".//{{{ncx_ns}}}navPoint"):
                        label_el = nav_point.find(f".//{{{ncx_ns}}}text")
                        content_el = nav_point.find(f"{{{ncx_ns}}}content")
                        if label_el is not None and content_el is not None:
                            bare = content_el.get("src", "").split("#")[0].strip()
                            title = (label_el.text or "").strip()
                            if bare and title:
                                toc_entries.append((title, bare))
                except ET.ParseError:
                    pass

    if not toc_entries:
        return None

    # ── Map TOC entries → page numbers ──────────────────────────────────────
    matched: list[tuple[str, int]] = []  # (title, page_number)
    seen_pages: set[int] = set()

    for title, bare_href in toc_entries:
        basename = bare_href.rsplit("/", 1)[-1]
        spine_i = href_to_spine.get(bare_href) if bare_href in href_to_spine \
            else href_to_spine.get(basename)
        if spine_i is None:
            continue
        page_num = spine_idx_to_page.get(spine_i)
        if page_num is None:
            # Spine item was empty (e.g. cover image); use next non-empty page
            for delta in range(1, 6):
                page_num = spine_idx_to_page.get(spine_i + delta)
                if page_num is not None:
                    break
        if page_num is None or page_num in seen_pages:
            continue
        seen_pages.add(page_num)
        matched.append((title, page_num))

    if not matched:
        return None

    matched.sort(key=lambda x: x[1])

    # ── Build Section objects ────────────────────────────────────────────────
    last_page = pages[-1].page_number
    sections: list[Section] = []

    # Front matter: spine docs before the first TOC entry
    if matched[0][1] > 1:
        sections.append(Section(
            name="Front Matter",
            start_page=1,
            end_page=matched[0][1] - 1,
            kind=SectionKind.FRONT_MATTER,
            confidence=1.0,
            detection_reason="epub-spine-before-toc",
            section_type="front_matter",
            parent="",
        ))

    for i, (title, page_num) in enumerate(matched):
        end_page = (matched[i + 1][1] - 1) if i + 1 < len(matched) else last_page
        end_page = max(end_page, page_num)
        stype = _infer_epub_section_type(title)
        if stype in {"notes", "index", "about_author", "acknowledgments"}:
            kind = SectionKind.BACK_MATTER
        elif stype in {"front_matter", "toc"}:
            kind = SectionKind.FRONT_MATTER
        else:
            kind = SectionKind.BODY
        sections.append(Section(
            name=title,
            start_page=page_num,
            end_page=end_page,
            kind=kind,
            confidence=1.0,
            detection_reason="epub-toc",
            section_type=stype,
            parent="",
        ))

    return sections or None


def _infer_epub_section_type(title: str) -> str:
    """Infer normalised section_type from an EPUB TOC title."""
    low = title.lower().strip()
    # Common front matter titles that should be skipped by the summariser
    if low in {"cover", "cover page", "title page", "half title", "half-title",
               "copyright", "copyright page", "dedication", "epigraph"}:
        return "front_matter"
    if low in {"contents", "table of contents", "toc"}:
        return "toc"
    if "chapter" in low or re.match(r"^\d+[.:\s]", low):
        return "chapter"
    # Part: starts with "Part" + any word/number — no length limit (was too strict)
    if re.match(r"^part\s+\S", low):
        return "part"
    if any(w in low for w in ("introduction", "preface", "foreword", "prologue")):
        return "introduction"
    if any(w in low for w in ("conclusion", "epilogue", "afterword")):
        return "epilogue"
    if "appendix" in low:
        return "appendix"
    if "acknowledg" in low:
        return "acknowledgments"
    if any(w in low for w in ("note", "endnote", "bibliograph", "reference", "glossary")):
        return "notes"
    if "index" in low:
        return "index"
    if "about" in low and "author" in low:
        return "about_author"
    # Default: treat as a named chapter (reasonable for most fiction/nonfiction TOC entries)
    return "chapter"
