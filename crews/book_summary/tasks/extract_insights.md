Using the chapter map from the previous task, read each chapter of the book at {book_path} and extract structured insights.

**Full coverage:** The chapter map includes total_pages at the top. You MUST call read_book for every chapter's page range. If a chapter spans pages you have not yet read in a single call, call read_book with the appropriate start_page (and possibly multiple times) to get all pages for that chapter. Do not skip chapters or assume content from the map alone — read the actual text for each chapter.

Use the **read_book** tool with the page ranges from the chapter map to read each chapter's content.

For each chapter, output:

### Chapter N: [Title]

**Core Ideas**
- 2–5 key ideas the author presents in this chapter. State the argument or claim, not just the topic.

**Key Examples**
- The most important examples, case studies, or stories. Include enough context (who, what, outcome) that the example is useful without re-reading the chapter. Include at least one **named** example (person, company, study, campaign) per chapter when the book provides it; otherwise write "No named example in this chapter."

**Frameworks & Mental Models**
- Any named or implied frameworks, models, or decision-making tools introduced. Include a short description of how each works and when to apply it.

**Actionable Takeaways**
- Concrete, specific actions a reader could take. "Schedule a weekly 1-hour block for X" not "do more X."

**Notable Quotes**
- 1–3 verbatim quotes that crystallize key points. Include page number if available. Include at least one verbatim quote per chapter when the text supports it; if there is no suitable quote, write "No notable quote extracted."

Rules:
- Do not skip chapters — produce output for every chapter in the chapter map. Read each chapter's pages with read_book before writing its section
- If a chapter repeats a point from an earlier chapter, note "Builds on Ch N" and focus on what is **new**: new examples, new nuance, new evidence. Do not rephrase the same idea for every chapter
- If a chapter has no meaningful substance (e.g. a one-page dedication), note that briefly and move on
- Core ideas should reflect what the author argues, not just what the chapter is about
- Examples must stand alone — a reader should understand the example without having read the book
- Be thorough but dense; no filler
- Your only available tool is read_book; do not call any other tools
