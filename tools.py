from pathlib import Path

import pdfplumber
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from crewai.tools import tool


@tool
def get_book_info(file_path: str) -> dict:
    """Get basic info about a book file (page count for PDFs). Call this first so you know how many pages to read. total_pages is the number of pages; you must read from page 0 to total_pages-1 by calling read_book in chunks."""
    path = Path(file_path)
    if not path.exists():
        return {"file": file_path, "error": f"File not found: {file_path}"}
    try:
        if path.suffix.lower() == ".pdf":
            with pdfplumber.open(path) as pdf:
                total = len(pdf.pages)
            return {"file": file_path, "total_pages": total, "format": "pdf"}
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            return {"file": file_path, "total_chars": len(text), "format": "text"}
    except Exception as e:
        return {"file": file_path, "error": str(e)}


@tool
def read_book(file_path: str, start_page: int = 0, max_pages: int = 50) -> dict:
    """Read a book from a local file and return its text. Supports .txt, .md, and .pdf files. For PDFs, extracts text page-by-page. Use start_page and max_pages to read in chunks if the book is large (call multiple times with increasing start_page to read the whole book)."""
    path = Path(file_path)
    if not path.exists():
        return {"file": file_path, "text": "", "error": f"File not found: {file_path}"}
    try:
        if path.suffix.lower() == ".pdf":
            pages = []
            with pdfplumber.open(path) as pdf:
                total = len(pdf.pages)
                end = min(start_page + max_pages, total)
                for i in range(start_page, end):
                    page_text = pdf.pages[i].extract_text() or ""
                    pages.append({"page": i + 1, "text": page_text})
            return {
                "file": file_path,
                "total_pages": total,
                "pages_returned": f"{start_page + 1}-{end}",
                "pages": pages,
            }
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            return {"file": file_path, "text": text[:200_000]}
    except Exception as e:
        return {"file": file_path, "text": "", "error": str(e)}


@tool
def extract_links(url: str, same_domain_only: bool = True, max_links: int = 100) -> dict:
    """Fetch a page and return a list of links on it as absolute URLs. Use this on index/category pages to discover real article URLs before fetching them. When same_domain_only is True, only returns links whose host matches the given URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        base_host = urlparse(url).netloc
        seen = set()
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith("#") or href.startswith("mailto:"):
                continue
            absolute = urljoin(url, href)
            parsed = urlparse(absolute)
            if parsed.scheme not in ("http", "https"):
                continue
            if same_domain_only and parsed.netloc != base_host:
                continue
            if absolute in seen:
                continue
            seen.add(absolute)
            text = (a.get_text() or "").strip()[:200]
            links.append({"url": absolute, "link_text": text or None})
            if len(links) >= max_links:
                break
        return {"page_url": url, "links": links, "count": len(links)}
    except Exception as e:
        return {"page_url": url, "links": [], "count": 0, "error": str(e)}


@tool
def fetch_webpage(url: str) -> dict:
    """Fetch a webpage and return structured content."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned = "\n".join(lines[:3000])

        return {
            "url": url,
            "title": title,
            "text": cleaned,
        }

    except Exception as e:
        return {
            "url": url,
            "title": "",
            "text": "",
            "error": str(e),
        }