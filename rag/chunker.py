"""Text chunking with chapter detection and page tracking."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .config import ChunkingConfig

# ---------------------------------------------------------------------------
# Chapter detection patterns (order matters — first match wins per line)
# ---------------------------------------------------------------------------
_CHAPTER_RE = [
    re.compile(
        r"^\s*(?:CHAPTER|Chapter)\s+(\d+|[IVXLCDM]+|"
        r"One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|"
        r"Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|"
        r"Seventeen|Eighteen|Nineteen|Twenty)",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:PART|Part)\s+(\d+|[IVXLCDM]+|"
        r"One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:Introduction|Conclusion|Preface|Foreword|Epilogue|"
        r"Prologue|Afterword|Appendix)\b",
        re.IGNORECASE,
    ),
]


@dataclass
class PageText:
    page_number: int
    text: str


@dataclass
class Chapter:
    name: str
    start_page: int
    end_page: int


@dataclass
class Chunk:
    chunk_id: str
    text: str
    book_id: str
    title: str
    author: str
    chapter: str
    section: str
    source_path: str
    chunk_index: int
    parent_section_id: str
    page_range: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_chapters(pages: list[PageText]) -> list[Chapter]:
    """Scan page text for chapter headings and return chapter boundaries."""
    boundaries: list[tuple[int, str]] = []

    for page in pages:
        for line in page.text.split("\n"):
            stripped = line.strip()
            if not stripped or len(stripped) > 120:
                continue
            for pattern in _CHAPTER_RE:
                if pattern.match(stripped):
                    heading = _normalise_heading(stripped)
                    if not boundaries or boundaries[-1][1] != heading:
                        boundaries.append((page.page_number, heading))
                    break

    if not boundaries:
        return [Chapter(name="Full Book", start_page=pages[0].page_number,
                        end_page=pages[-1].page_number)]

    chapters: list[Chapter] = []

    if boundaries[0][0] > pages[0].page_number:
        chapters.append(Chapter(
            name="Front Matter",
            start_page=pages[0].page_number,
            end_page=boundaries[0][0] - 1,
        ))

    for idx, (page_num, heading) in enumerate(boundaries):
        end = (boundaries[idx + 1][0] - 1
               if idx + 1 < len(boundaries)
               else pages[-1].page_number)
        chapters.append(Chapter(name=heading, start_page=page_num, end_page=end))

    return chapters


def chunk_pages(
    pages: list[PageText],
    chapters: list[Chapter],
    book_id: str,
    title: str,
    author: str,
    source_path: str,
    config: ChunkingConfig,
) -> list[Chunk]:
    """Split page text into overlapping chunks, respecting chapter boundaries."""
    chunks: list[Chunk] = []
    global_index = 0

    for chapter in chapters:
        chapter_pages = [p for p in pages
                         if chapter.start_page <= p.page_number <= chapter.end_page]
        if not chapter_pages:
            continue

        page_map: list[tuple[int, int]] = []  # (char_offset, page_number)
        full_text = ""
        for p in chapter_pages:
            page_map.append((len(full_text), p.page_number))
            full_text += p.text + "\n"

        raw_chunks = _split_text(full_text, config)

        for raw in raw_chunks:
            start_p = _page_at(raw["start"], page_map)
            end_p = _page_at(raw["end"], page_map)
            text = raw["text"].strip()
            if len(text) < config.min_chunk_size:
                continue

            chunks.append(Chunk(
                chunk_id=f"{book_id}::ch{global_index}",
                text=text,
                book_id=book_id,
                title=title,
                author=author,
                chapter=chapter.name,
                section="",        # TODO v2: sub-section detection
                source_path=source_path,
                chunk_index=global_index,
                parent_section_id=chapter.name,
                page_range=f"{start_p}-{end_p}",
            ))
            global_index += 1

    return chunks


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _normalise_heading(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text[:120]


def _split_text(text: str, config: ChunkingConfig) -> list[dict]:
    """Split text into chunks at paragraph boundaries with overlap."""
    paragraphs = re.split(r"\n\s*\n", text)
    result: list[dict] = []
    current = ""
    current_start = 0
    offset = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            offset += 1
            continue

        if len(current) + len(para) + 1 > config.target_size and current:
            result.append({"text": current, "start": current_start, "end": offset})
            # overlap: keep tail of current chunk
            overlap_text = current[-config.overlap:] if config.overlap else ""
            current_start = offset - len(overlap_text)
            current = overlap_text

        if current:
            current += "\n\n" + para
        else:
            current_start = offset
            current = para
        offset += len(para) + 2  # account for \n\n

    if current.strip():
        result.append({"text": current, "start": current_start, "end": offset})

    return result


def _page_at(char_offset: int, page_map: list[tuple[int, int]]) -> int:
    """Find which page a character offset falls on."""
    page = page_map[0][1]
    for off, pnum in page_map:
        if off > char_offset:
            break
        page = pnum
    return page
