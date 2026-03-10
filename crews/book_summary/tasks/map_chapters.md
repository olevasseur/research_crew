Read the book file at: {book_path}

**Step 1 — Get page count:** Call **get_book_info** with the book path. It returns total_pages (for PDFs). You must read the entire book; do not stop after one chunk.

**Step 2 — Read the full book:** Use **read_book** to read the book. read_book returns at most 50 pages per call. You MUST call read_book repeatedly with increasing start_page (0, then 50, then 100, …) until you have read every page up to total_pages. For each call, use start_page and max_pages=50. Confirm you have covered all pages before producing the chapter map.

**Step 3 — Produce the chapter map:** At the top of your output, include a line: **total_pages: N** (the value from get_book_info). The analyst needs this to ensure they read every chapter.

Produce a chapter map with the following for each chapter or major section:

- chapter_number (or "Intro", "Conclusion", "Appendix", etc.)
- chapter_title
- page_start
- page_end
- one_sentence_summary

Rules:
- Include every chapter, foreword, introduction, conclusion, and appendix as separate entries
- Be precise about where each chapter starts and ends — the analyst will use these page ranges to read each chapter
- If the book doesn't use explicit chapter numbers, use the section headings as chapter titles and assign sequential numbers
- Your only available tools are get_book_info and read_book; do not call any other tools
