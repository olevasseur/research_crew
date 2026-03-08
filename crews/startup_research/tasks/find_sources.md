Read the provided seed URLs and extract only concrete, evidence-backed pain points.

**Available tools (use only these):** fetch_webpage(url), extract_links(url). Do not call any other tools.

**Discovering article URLs (important):** When a seed URL is an index, category, or listing page (e.g. https://github.blog/engineering/), do **not** guess article URLs. Use the **extract_links** tool on that seed URL to get a list of real links on the page. Then fetch only URLs that appear in that list (e.g. with fetch_webpage) to find pain points. **Only use as source_url a URL that was returned by extract_links and that you actually fetched**—never invent or guess URLs.

For every pain point found, output:

- source_url
- page_title
- extracted_pain_point
- target_user
- evidence_snippet
- why_this_looks_real
- confidence (high/medium/low)

Rules:
- **source_url must be the exact URL of the page where the evidence appears.** For index/listing seeds: call extract_links(seed_url), pick promising article URLs from the returned list, fetch those with fetch_webpage, and use those fetched URLs as source_url. Do not use the index URL as source_url and do not make up article URLs.
- Do not invent pain points not supported by the page text.
- Prefer explicit complaints, repeated friction, manual work, regulatory burden, costly inefficiency, or recurring operational pain.
- If a page is not useful, say so briefly.
- Keep evidence snippets short and specific.