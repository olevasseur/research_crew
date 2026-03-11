"""Book structure detection and text chunking.

Structure detection is separate from chunking:
  1. detect_structure()  — find section boundaries from page text
  2. chunk_pages()       — split text into overlapping chunks within sections

The detector uses a page-level state machine (front → toc → body → back)
and explicit heading patterns with confidence scoring.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass, field
from enum import Enum

from .config import ChunkingConfig


# ═══════════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PageText:
    page_number: int
    text: str


class SectionKind(Enum):
    FRONT_MATTER = "Front Matter"
    TOC = "Table of Contents"
    BODY = "Body"
    BACK_MATTER = "Back Matter"


SECTION_TYPES_SUMMARIZABLE = {"introduction", "chapter", "conclusion", "epilogue", "appendix"}
SECTION_TYPES_SKIPPABLE = {"front_matter", "toc", "acknowledgments", "notes", "index", "about_author", "unknown"}


@dataclass
class HeadingMatch:
    """A detected heading candidate with provenance."""
    page: int
    raw_text: str
    label: str          # normalised display name
    heading_type: str   # e.g. "bare_number", "part", "structural", "back_matter"
    confidence: float
    reason: str         # human-readable explanation of why it matched
    kind: SectionKind = SectionKind.BODY
    section_type: str = "unknown"
    parent: str = ""


@dataclass
class Section:
    """A detected section of the book."""
    name: str
    start_page: int
    end_page: int
    kind: SectionKind = SectionKind.BODY
    confidence: float = 1.0
    detection_reason: str = ""
    section_type: str = "unknown"  # normalised type: chapter, part, introduction, etc.
    parent: str = ""               # parent section name (e.g. "PART 1" for chapters)


# Keep backward compat — other modules import "Chapter"
Chapter = Section


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
    section_type: str = "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# Heading patterns
#
# Each rule: (compiled regex, confidence, heading_type, SectionKind)
# Patterns are tried in order; first match wins per line.
# ═══════════════════════════════════════════════════════════════════════════

_HEADING_PATTERNS: list[tuple[re.Pattern, float, str, SectionKind]] = [
    # --- Bare number on its own line (very common in modern books) ---
    # Matches "1", "2", … "99" when the line contains ONLY the number.
    (re.compile(r"^\s*(\d{1,2})\s*$"),
     0.90, "bare_number", SectionKind.BODY),

    # --- "CHAPTER N" or "Chapter N" with optional subtitle ---
    (re.compile(
        r"^\s*(?:CHAPTER|Chapter)\s+"
        r"(\d{1,3}|[IVXLCDM]{1,10}|"
        r"One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|"
        r"Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|"
        r"Seventeen|Eighteen|Nineteen|Twenty)"
        r"(?:\s*[:.—–\-]\s*.+)?$",
        re.IGNORECASE),
     0.95, "chapter_keyword", SectionKind.BODY),

    # --- "N. Title" format (e.g. "1. A Lopsided Arms Race") ---
    (re.compile(r"^\s*(\d{1,2})\.\s+[A-Z]"),
     0.90, "numbered_dot_title", SectionKind.BODY),

    # --- "PART N" or "Part N" with optional subtitle ---
    (re.compile(
        r"^\s*(?:PART|Part)\s+"
        r"(\d{1,3}|[IVXLCDM]{1,10}|"
        r"One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)"
        r"(?:\s*[:.—–\-]\s*.+)?$",
        re.IGNORECASE),
     0.95, "part", SectionKind.BODY),

    # --- Structural sections (Introduction, Conclusion, etc.) ---
    # Must be the ENTIRE line (with optional short subtitle).
    (re.compile(
        r"^\s*(?:Introduction|Conclusion|Preface|Foreword|Epilogue|"
        r"Prologue|Afterword|Appendix)"
        r"(?:\s*[:.—–\-]\s*.{0,40})?$",
        re.IGNORECASE),
     0.85, "structural", SectionKind.BODY),

    # --- Back matter ---
    (re.compile(
        r"^\s*(?:Notes|Endnotes|Bibliography|References|"
        r"Acknowledgments|Acknowledgements|"
        r"About\s+the\s+Author|About\s+the\s+Authors|"
        r"Index|Glossary|"
        r"Further\s+Reading|Recommended\s+Reading|"
        r"Selected\s+Bibliography)"
        r"(?:\s*[:.—–\-]\s*.{0,40})?$",
        re.IGNORECASE),
     0.85, "back_matter", SectionKind.BACK_MATTER),
]


# ═══════════════════════════════════════════════════════════════════════════
# Rejection filters — lines that LOOK like headings but aren't
# ═══════════════════════════════════════════════════════════════════════════

def _is_toc_line(line: str) -> bool:
    """Dot leaders, trailing page numbers, multiple chapter refs on one line."""
    if re.search(r"\.{3,}", line):
        return True
    if re.search(r"(?:chapter|part)\s+\d.*(?:chapter|part)\s+\d", line, re.IGNORECASE):
        return True
    return False


def _is_sentence(line: str) -> bool:
    """Lines with commas, semicolons, or clause-style words are body text."""
    if re.search(r"[,;]", line):
        return True
    word_count = len(line.split())
    if word_count > 8:
        return True
    return False


def _is_notes_reference(line: str) -> bool:
    """Lines like 'CHAPTER 2: DIGITAL MINIMALISM' inside the Notes section."""
    if re.match(r"^\s*CHAPTER\s+\d", line, re.IGNORECASE):
        # Only reject if the line also contains a colon followed by text
        # (notes section headers like "CHAPTER 2: DIGITAL MINIMALISM")
        if ":" in line and len(line.split(":")) >= 2:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Page-level state machine
# ═══════════════════════════════════════════════════════════════════════════

class _BookState(Enum):
    FRONT = "front"
    TOC = "toc"
    BODY = "body"
    BACK = "back"


def _classify_state_transition(
    heading: HeadingMatch,
    current_state: _BookState,
) -> _BookState:
    """Determine the new state after encountering a heading."""
    if heading.heading_type == "back_matter":
        return _BookState.BACK
    if heading.heading_type in ("bare_number", "chapter_keyword", "numbered_dot_title", "part"):
        return _BookState.BODY
    if heading.heading_type == "structural":
        low = heading.label.lower()
        if any(w in low for w in ("introduction", "preface", "foreword", "prologue")):
            if current_state in (_BookState.FRONT, _BookState.TOC):
                return _BookState.BODY
            return _BookState.BODY
        if any(w in low for w in ("conclusion", "epilogue", "afterword")):
            return _BookState.BODY
    return current_state


# ═══════════════════════════════════════════════════════════════════════════
# Public API: Structure Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_structure(
    pages: list[PageText],
    debug: bool = False,
) -> tuple[list[Section], list[HeadingMatch]]:
    """Detect book structure from page text.

    Returns:
        sections: list of Section objects with boundaries
        all_matches: list of all HeadingMatch objects (for debugging)
    """
    if not pages:
        return [], []

    # Pass 1: Find all heading candidates
    raw_matches = _scan_for_headings(pages, debug)

    # Pass 2: Resolve bare numbers — look at the next line(s) for a title
    resolved = _resolve_bare_numbers(raw_matches, pages, debug)

    # Pass 3: Deduplicate (keep later occurrence for same heading)
    deduped = _deduplicate(resolved, debug)

    # Pass 4: Remove TOC clusters
    filtered = _remove_toc_clusters(deduped, debug)

    # Pass 5: Assign section kinds via state machine (also filters out
    # structural headings that appear inside back matter)
    filtered = _assign_kinds(filtered)

    # Pass 6: Build section list
    if not filtered:
        if debug:
            print("[structure] No headings found — one big section", file=sys.stderr)
        sections = [Section(
            name="Full Book",
            start_page=pages[0].page_number,
            end_page=pages[-1].page_number,
            kind=SectionKind.BODY,
            confidence=1.0,
            detection_reason="no headings detected",
        )]
        return sections, raw_matches

    sections = _build_sections(filtered, pages)

    if debug:
        print(f"\n[structure] Final sections ({len(sections)}):", file=sys.stderr)
        for s in sections:
            print(f"  p.{s.start_page:>3d}-{s.end_page:<3d}  "
                  f"[{s.kind.value:12s}]  conf={s.confidence:.2f}  "
                  f"{s.name!r}  ({s.detection_reason})", file=sys.stderr)

    return sections, raw_matches


def detect_chapters(
    pages: list[PageText],
    debug: bool = False,
    min_confidence: float = 0.0,  # kept for API compat; filtering is internal now
) -> list[Chapter]:
    """Backward-compatible wrapper around detect_structure."""
    sections, _ = detect_structure(pages, debug=debug)
    return sections


# ═══════════════════════════════════════════════════════════════════════════
# Public API: Chunking
# ═══════════════════════════════════════════════════════════════════════════

def chunk_pages(
    pages: list[PageText],
    chapters: list[Section],
    book_id: str,
    title: str,
    author: str,
    source_path: str,
    config: ChunkingConfig,
) -> list[Chunk]:
    """Split page text into overlapping chunks, respecting section boundaries."""
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
                parent_section_id=chapter.parent or chapter.name,
                page_range=f"{start_p}-{end_p}",
                section_type=chapter.section_type,
            ))
            global_index += 1

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# Internal: Heading scanning
# ═══════════════════════════════════════════════════════════════════════════

def _scan_for_headings(pages: list[PageText], debug: bool) -> list[HeadingMatch]:
    """First pass: collect every line matching a heading pattern."""
    matches: list[HeadingMatch] = []

    for page in pages:
        lines = page.text.split("\n")
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Length guard: real headings are short
            if len(stripped) > 60:
                continue

            # Reject TOC-style lines
            if _is_toc_line(stripped):
                continue

            # Reject sentences (unless it's a very short line)
            if len(stripped) > 5 and _is_sentence(stripped):
                continue

            # Reject notes-section chapter references
            if _is_notes_reference(stripped):
                if debug:
                    print(f"  [reject-noteref] p.{page.page_number}  {stripped!r}", file=sys.stderr)
                continue

            # Try each heading pattern
            for pattern, confidence, htype, kind in _HEADING_PATTERNS:
                if pattern.match(stripped):
                    label = _normalise_heading(stripped)
                    matches.append(HeadingMatch(
                        page=page.page_number,
                        raw_text=stripped,
                        label=label,
                        heading_type=htype,
                        confidence=confidence,
                        reason=f"matched {htype} pattern",
                        kind=kind,
                    ))
                    if debug:
                        print(f"  [match]  p.{page.page_number:>3d}  "
                              f"conf={confidence:.2f}  {htype:20s}  {label!r}",
                              file=sys.stderr)
                    break

    return matches


def _resolve_bare_numbers(
    matches: list[HeadingMatch],
    pages: list[PageText],
    debug: bool,
) -> list[HeadingMatch]:
    """For 'bare_number' matches, look at the next non-empty line on the same
    page for a chapter title. E.g. page has '1' then 'A Lopsided Arms Race'.
    """
    page_text: dict[int, str] = {p.page_number: p.text for p in pages}
    resolved: list[HeadingMatch] = []

    for m in matches:
        if m.heading_type != "bare_number":
            resolved.append(m)
            continue

        # Find the line after the number on the same page
        text = page_text.get(m.page, "")
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        title_line = None
        found_number = False
        for line in lines:
            if found_number:
                # Next non-empty line after the number
                if len(line) < 60 and not _is_sentence(line) and len(line) > 2:
                    title_line = line
                break
            if line == m.raw_text:
                found_number = True

        if title_line:
            new_label = f"Chapter {m.raw_text}: {_normalise_heading(title_line)}"
            m.label = new_label
            m.confidence = 0.92
            m.reason = f"bare number {m.raw_text!r} + title {title_line!r}"
        else:
            new_label = f"Chapter {m.raw_text}"
            m.label = new_label
            m.reason = f"bare number {m.raw_text!r} (no title found on same page)"

        if debug:
            print(f"  [resolve] p.{m.page:>3d}  {m.raw_text!r} → {m.label!r}", file=sys.stderr)

        resolved.append(m)

    return resolved


def _deduplicate(matches: list[HeadingMatch], debug: bool) -> list[HeadingMatch]:
    """If the same heading appears on multiple pages, keep the later one."""
    seen: dict[str, HeadingMatch] = {}
    for m in matches:
        key = m.label.lower()
        if key in seen:
            if debug:
                print(f"  [dedup]  p.{seen[key].page} → p.{m.page}  {m.label!r}", file=sys.stderr)
        seen[key] = m
    return sorted(seen.values(), key=lambda m: m.page)


def _remove_toc_clusters(matches: list[HeadingMatch], debug: bool) -> list[HeadingMatch]:
    """If 5+ headings appear within 2 pages, it's a TOC — remove them."""
    if len(matches) < 5:
        return matches

    toc_pages: set[int] = set()
    for i in range(len(matches)):
        cluster = [matches[i]]
        for j in range(i + 1, len(matches)):
            if matches[j].page - matches[i].page <= 2:
                cluster.append(matches[j])
            else:
                break
        if len(cluster) >= 5:
            for c in cluster:
                toc_pages.add(c.page)

    if not toc_pages:
        return matches

    result = []
    for m in matches:
        if m.page in toc_pages:
            if debug:
                print(f"  [reject-toc-cluster] p.{m.page}  {m.label!r}", file=sys.stderr)
        else:
            result.append(m)
    return result


def _assign_kinds(matches: list[HeadingMatch]) -> list[HeadingMatch]:
    """Walk through matches with the state machine and set .kind.

    Once in BACK state, suppress structural headings (Introduction, Conclusion,
    etc.) because they're likely note-section headers, not real book sections.
    Returns the filtered list.
    """
    state = _BookState.FRONT
    result: list[HeadingMatch] = []
    for m in matches:
        new_state = _classify_state_transition(m, state)

        # Once in back matter, reject structural headings that reappear
        # (e.g. "INTRODUCTION" as a notes sub-header)
        if state == _BookState.BACK and m.heading_type == "structural":
            continue
        # Also reject bare numbers inside back matter (page numbers, note numbers)
        if state == _BookState.BACK and m.heading_type == "bare_number":
            continue

        state = new_state
        if state == _BookState.BACK:
            m.kind = SectionKind.BACK_MATTER
        elif state == _BookState.FRONT:
            m.kind = SectionKind.FRONT_MATTER
        else:
            m.kind = SectionKind.BODY
        result.append(m)
    return result


def _infer_section_type(heading_type: str, label: str) -> str:
    """Map a heading match to a normalised section_type string."""
    low = label.lower()
    if heading_type == "part":
        return "part"
    if heading_type in ("bare_number", "chapter_keyword", "numbered_dot_title"):
        return "chapter"
    if heading_type == "structural":
        if "introduction" in low:
            return "introduction"
        if "conclusion" in low:
            return "conclusion"
        if "preface" in low or "foreword" in low or "prologue" in low:
            return "introduction"
        if "epilogue" in low or "afterword" in low:
            return "epilogue"
        if "appendix" in low:
            return "appendix"
        return "chapter"
    if heading_type == "back_matter":
        if "acknowledg" in low:
            return "acknowledgments"
        if "note" in low or "endnote" in low:
            return "notes"
        if "index" in low:
            return "index"
        if "about" in low and "author" in low:
            return "about_author"
        if "bibliograph" in low or "reference" in low:
            return "notes"
        if "glossary" in low:
            return "notes"
        return "notes"
    return "unknown"


def _try_split_introduction_from_front_matter(
    front_section: Section,
    pages: list[PageText],
) -> list[Section]:
    """If Front Matter contains a page where 'Introduction' appears as a
    heading-like line, split it into Front Matter + Introduction."""
    intro_re = re.compile(r"^\s*Introduction\s*$", re.IGNORECASE)
    for p in pages:
        if p.page_number < front_section.start_page or p.page_number > front_section.end_page:
            continue
        for line in p.text.split("\n"):
            if intro_re.match(line.strip()):
                # Only split if there's meaningful front matter before it
                if p.page_number > front_section.start_page:
                    return [
                        Section(
                            name="Front Matter",
                            start_page=front_section.start_page,
                            end_page=p.page_number - 1,
                            kind=SectionKind.FRONT_MATTER,
                            confidence=1.0,
                            detection_reason="pages before Introduction",
                            section_type="front_matter",
                        ),
                        Section(
                            name="Introduction",
                            start_page=p.page_number,
                            end_page=front_section.end_page,
                            kind=SectionKind.BODY,
                            confidence=0.85,
                            detection_reason="Introduction heading found inside front matter",
                            section_type="introduction",
                        ),
                    ]
                else:
                    front_section.name = "Introduction"
                    front_section.kind = SectionKind.BODY
                    front_section.section_type = "introduction"
                    front_section.confidence = 0.85
                    front_section.detection_reason = "Introduction heading on first page of front matter"
                    return [front_section]
    return [front_section]


def _build_sections(
    matches: list[HeadingMatch],
    pages: list[PageText],
) -> list[Section]:
    """Convert matched headings into Section objects with page boundaries."""
    sections: list[Section] = []
    first_page = pages[0].page_number
    last_page = pages[-1].page_number

    # Front matter: pages before the first heading
    if matches[0].page > first_page:
        front = Section(
            name="Front Matter",
            start_page=first_page,
            end_page=matches[0].page - 1,
            kind=SectionKind.FRONT_MATTER,
            confidence=1.0,
            detection_reason="pages before first heading",
            section_type="front_matter",
        )
        sections.extend(_try_split_introduction_from_front_matter(front, pages))

    # Track current part for parent assignment
    current_part = ""
    for idx, m in enumerate(matches):
        end = (matches[idx + 1].page - 1
               if idx + 1 < len(matches)
               else last_page)
        stype = _infer_section_type(m.heading_type, m.label)
        if stype == "part":
            current_part = m.label
        parent = current_part if stype == "chapter" else ""
        sections.append(Section(
            name=m.label,
            start_page=m.page,
            end_page=end,
            kind=m.kind,
            confidence=m.confidence,
            detection_reason=m.reason,
            section_type=stype,
            parent=parent,
        ))

    return sections


# ═══════════════════════════════════════════════════════════════════════════
# Text utilities
# ═══════════════════════════════════════════════════════════════════════════

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
