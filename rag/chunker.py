"""Text chunking with strict chapter detection and page tracking."""

from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass
from enum import Enum

from .config import ChunkingConfig


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PageText:
    page_number: int
    text: str


class SectionKind(Enum):
    FRONT_MATTER = "Front Matter"
    TOC = "Table of Contents"
    BODY = "Body"
    BACK_MATTER = "Back Matter"


@dataclass
class Chapter:
    name: str
    start_page: int
    end_page: int
    kind: SectionKind = SectionKind.BODY
    confidence: float = 1.0  # 0.0 – 1.0
    detection_reason: str = ""


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
# Heading patterns with confidence tiers
#
# HIGH  (0.95) — Unambiguous chapter/part markers with numbers
# GOOD  (0.85) — Known structural labels (Introduction, Conclusion, etc.)
# BACK  (0.80) — Back-matter labels (Notes, Index, etc.)
# ---------------------------------------------------------------------------

_HEADING_RULES: list[tuple[re.Pattern, float, str]] = [
    # --- HIGH confidence: numbered chapters ---
    (re.compile(
        r"^\s*(?:CHAPTER|Chapter)\s+"
        r"(\d{1,3}|[IVXLCDM]{1,10}|"
        r"One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|"
        r"Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|"
        r"Seventeen|Eighteen|Nineteen|Twenty)"
        r"(?:\s*[:.—–\-]\s*.+)?$",
        re.IGNORECASE,
    ), 0.95, "numbered chapter"),

    # --- HIGH confidence: numbered parts ---
    (re.compile(
        r"^\s*(?:PART|Part)\s+"
        r"(\d{1,3}|[IVXLCDM]{1,10}|"
        r"One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)"
        r"(?:\s*[:.—–\-]\s*.+)?$",
        re.IGNORECASE,
    ), 0.95, "numbered part"),

    # --- GOOD confidence: known structural sections ---
    (re.compile(
        r"^\s*(?:Introduction|Conclusion|Preface|Foreword|Epilogue|"
        r"Prologue|Afterword|Appendix)"
        r"(?:\s*[:.—–\-]\s*.+)?$",
        re.IGNORECASE,
    ), 0.85, "structural section"),

    # --- BACK confidence: back-matter sections ---
    (re.compile(
        r"^\s*(?:Notes|Endnotes|Bibliography|References|"
        r"Acknowledgments|Acknowledgements|"
        r"About\s+the\s+Author|Index|Glossary|"
        r"Further\s+Reading|Recommended\s+Reading|"
        r"Selected\s+Bibliography)"
        r"(?:\s*[:.—–\-]\s*.+)?$",
        re.IGNORECASE,
    ), 0.80, "back matter"),
]

# Lines that look like headings but are really TOC entries or body text
_TOC_INDICATORS = re.compile(
    r"(?:"
    r"\.{3,}"               # dot leaders (Introduction.........12)
    r"|\d{1,4}\s*$"         # bare trailing page number
    r"|(?:chapter|part)\s+\d.*chapter\s+\d"  # multiple chapters on one line
    r")",
    re.IGNORECASE,
)

# Reject lines that are clearly sentences (body text), not headings
_SENTENCE_RE = re.compile(
    r"[,;]"                  # commas/semicolons (headings almost never have these)
    r"|"
    r"\b(?:is|are|was|were|have|has|had|will|would|could|should|"
    r"that|which|because|although|however|therefore|"
    r"the\s+\w+\s+\w+\s+\w+)"  # 4+ word phrases starting with "the"
    r"",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Heading candidate
# ---------------------------------------------------------------------------

@dataclass
class _Candidate:
    page: int
    text: str
    confidence: float
    reason: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_chapters(
    pages: list[PageText],
    debug: bool = False,
    min_confidence: float = 0.80,
) -> list[Chapter]:
    """Scan pages for chapter headings, returning chapter boundaries.

    Uses strict pattern matching with confidence scores. Prefers fewer,
    correct headings over more, potentially wrong ones.
    """
    if not pages:
        return []

    candidates = _find_heading_candidates(pages, debug)
    accepted = _filter_candidates(candidates, min_confidence, debug)

    if not accepted:
        if debug:
            print("[chapter-detect] No headings found — treating as one section", file=sys.stderr)
        return [Chapter(
            name="Full Book",
            start_page=pages[0].page_number,
            end_page=pages[-1].page_number,
            kind=SectionKind.BODY,
            confidence=1.0,
            detection_reason="no headings detected",
        )]

    chapters = _build_chapters(accepted, pages)

    if debug:
        print(f"\n[chapter-detect] Final chapter list ({len(chapters)}):", file=sys.stderr)
        for ch in chapters:
            print(f"  p.{ch.start_page}-{ch.end_page}  [{ch.kind.value}]  "
                  f"conf={ch.confidence:.2f}  {ch.name!r}  ({ch.detection_reason})",
                  file=sys.stderr)

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

        page_map: list[tuple[int, int]] = []
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
                section="",
                source_path=source_path,
                chunk_index=global_index,
                parent_section_id=chapter.name,
                page_range=f"{start_p}-{end_p}",
            ))
            global_index += 1

    return chunks


# ---------------------------------------------------------------------------
# Heading detection internals
# ---------------------------------------------------------------------------

def _find_heading_candidates(pages: list[PageText], debug: bool) -> list[_Candidate]:
    """First pass: collect every line that matches a heading pattern."""
    candidates: list[_Candidate] = []

    for page in pages:
        for line in page.text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue

            # Hard length limits: headings are short
            if len(stripped) > 80 or len(stripped) < 3:
                continue

            # Reject TOC-style lines (dot leaders, trailing page numbers)
            if _TOC_INDICATORS.search(stripped):
                if debug:
                    print(f"  [reject-toc]  p.{page.page_number}  {stripped!r}", file=sys.stderr)
                continue

            # Reject lines that look like sentences
            if _SENTENCE_RE.search(stripped):
                if debug:
                    print(f"  [reject-sent] p.{page.page_number}  {stripped!r}", file=sys.stderr)
                continue

            # Try each heading rule
            for pattern, confidence, reason in _HEADING_RULES:
                if pattern.match(stripped):
                    heading = _normalise_heading(stripped)
                    candidates.append(_Candidate(
                        page=page.page_number,
                        text=heading,
                        confidence=confidence,
                        reason=reason,
                    ))
                    if debug:
                        print(f"  [candidate]   p.{page.page_number}  conf={confidence:.2f}  "
                              f"{reason:20s}  {heading!r}", file=sys.stderr)
                    break

    return candidates


def _filter_candidates(
    candidates: list[_Candidate],
    min_confidence: float,
    debug: bool,
) -> list[_Candidate]:
    """Second pass: deduplicate, reject low-confidence, reject TOC clusters."""
    if not candidates:
        return []

    # 1. Filter by confidence
    filtered = [c for c in candidates if c.confidence >= min_confidence]

    # 2. Deduplicate: if the same heading text appears on multiple pages,
    #    keep only the LATER occurrence (the real chapter, not the TOC entry).
    seen: dict[str, _Candidate] = {}
    for c in filtered:
        key = c.text.lower()
        if key in seen:
            if debug:
                print(f"  [dedup]       p.{seen[key].page} → p.{c.page}  {c.text!r}  "
                      f"(keeping later)", file=sys.stderr)
        seen[key] = c
    filtered = sorted(seen.values(), key=lambda c: c.page)

    # 3. Reject TOC clusters: if many headings appear within a small page range
    #    (e.g. 5+ headings within 3 pages), that's a table of contents, not real
    #    chapter starts. Remove them.
    filtered = _remove_toc_clusters(filtered, debug)

    # 4. Reject if only 1 heading found and it's on page 1-2 (likely TOC/title page)
    if len(filtered) == 1 and filtered[0].page <= 2:
        if debug:
            print(f"  [reject-solo] p.{filtered[0].page}  {filtered[0].text!r}  "
                  f"(only heading, too early)", file=sys.stderr)
        return []

    return filtered


def _remove_toc_clusters(candidates: list[_Candidate], debug: bool) -> list[_Candidate]:
    """Remove clusters of headings that are likely a table of contents."""
    if len(candidates) < 4:
        return candidates

    # Sliding window: if 4+ headings fall within 3 pages, it's a TOC cluster
    toc_pages: set[int] = set()
    window_size = 3  # pages

    for i in range(len(candidates)):
        cluster = [candidates[i]]
        for j in range(i + 1, len(candidates)):
            if candidates[j].page - candidates[i].page <= window_size:
                cluster.append(candidates[j])
            else:
                break
        if len(cluster) >= 4:
            for c in cluster:
                toc_pages.add(c.page)

    if not toc_pages:
        return candidates

    result = []
    for c in candidates:
        if c.page in toc_pages:
            if debug:
                print(f"  [reject-cluster] p.{c.page}  {c.text!r}  "
                      f"(TOC cluster)", file=sys.stderr)
        else:
            result.append(c)

    return result


def _build_chapters(
    accepted: list[_Candidate],
    pages: list[PageText],
) -> list[Chapter]:
    """Turn accepted heading candidates into Chapter objects with boundaries."""
    chapters: list[Chapter] = []
    first_page = pages[0].page_number
    last_page = pages[-1].page_number

    # Front matter: everything before the first heading
    if accepted[0].page > first_page:
        chapters.append(Chapter(
            name="Front Matter",
            start_page=first_page,
            end_page=accepted[0].page - 1,
            kind=SectionKind.FRONT_MATTER,
            confidence=1.0,
            detection_reason="pages before first heading",
        ))

    for idx, cand in enumerate(accepted):
        end = (accepted[idx + 1].page - 1
               if idx + 1 < len(accepted)
               else last_page)

        kind = _classify_kind(cand)

        chapters.append(Chapter(
            name=cand.text,
            start_page=cand.page,
            end_page=end,
            kind=kind,
            confidence=cand.confidence,
            detection_reason=cand.reason,
        ))

    # If the last chapter is back matter and it's very large (>30% of book),
    # split it: keep only a reasonable portion
    if len(chapters) > 1 and chapters[-1].kind == SectionKind.BACK_MATTER:
        total_pages = last_page - first_page + 1
        back_pages = chapters[-1].end_page - chapters[-1].start_page + 1
        if back_pages > total_pages * 0.3:
            chapters[-1].detection_reason += " (unusually large — review manually)"

    return chapters


def _classify_kind(cand: _Candidate) -> SectionKind:
    """Determine whether a heading belongs to body or back matter."""
    low = cand.text.lower()
    if cand.reason == "back matter":
        return SectionKind.BACK_MATTER
    if any(w in low for w in ("introduction", "preface", "foreword", "prologue")):
        return SectionKind.FRONT_MATTER
    return SectionKind.BODY


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _normalise_heading(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text[:80]


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
            overlap_text = current[-config.overlap:] if config.overlap else ""
            current_start = offset - len(overlap_text)
            current = overlap_text

        if current:
            current += "\n\n" + para
        else:
            current_start = offset
            current = para
        offset += len(para) + 2

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
