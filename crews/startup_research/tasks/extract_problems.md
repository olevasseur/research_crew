Using the prior research output, group similar pain points into normalized problems.

For each normalized problem, output:

- pain_point_id
- normalized_problem
- mention_count
- distinct_source_count
- source_urls
- supporting_snippets
- target_user
- urgency
- frequency
- willingness_to_pay_signal
- notes

Rules:
- **source_urls must be the precise article/page URLs where evidence was found** (from the research output). Do not use index or category URLs (e.g. https://github.blog/engineering/); use only the full URL of each specific article or post.
- Merge near-duplicates.
- Do not include problems without source evidence.
- Count how many times the problem appears across the research.
- Prefer concrete and repeated pain over vague trends.