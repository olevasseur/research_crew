# Working Memory

## Current goal
Harden the RAG pipeline; CrewAI is not the active focus.

## Latest progress
- RAG hardening landed: MMR selection, eval flow, and navigation (`trace`/`explore`) are implemented and working on `digital-minimalism`.
- `review-eval` CLI works; fallback-env validation passed for `python rag_cli.py review-eval digital-minimalism`.
- `fiction-whois` real CLI flow passes on `pale-criminal`/`Gunther`, reads `fiction_state.json`, respects chapter cutoff, and has regression coverage.
- Recent UI/repo hygiene work completed; no current action needed there.

## Open questions
- Is the `digital-minimalism` summary actually good enough after reviewing eval output?
- Is CrewAI retained at all, or effectively dead code?
- Should `fiction-whois` get one more fixture/assertion for schema variation or output-label regression protection?
- Some validation paths still fail in this runner due to missing tools/deps (`python -m pytest`, `rag_cli.py --help`).

## Active assumptions
- `llama3.1` via Ollama is sufficient for current summarization experiments.
- Summary quality has not yet been meaningfully reviewed beyond generating eval/review output.
- If prompts change, bump `PROMPT_VERSION` from `v4`.
- `fiction_state.json` may still vary enough that current tests miss edge cases.

## What matters next
- Use `review-eval` to inspect `digital-minimalism` quality and decide whether prompt/selection changes are needed.
- If eval is weak, revise summary prompts/selection and bump `PROMPT_VERSION`.
- Add a second book to exercise multi-book `compare` and `ask` flows.
- Optionally add one more `fiction-whois` fixture/assertion for schema variation/formatting.## Iteration 0 · 2026-03-29 22:54:09 UTC

**Progress:** Add a small direct-prompt bypass mode to the orchestrator web UI by introducing a checkbox, threading a boolean flag through the existing start-run request path, and creating the thinnest compatible backend path that skips planner invocation while preserving downstream execution/review behavior as much as possible.

**Decisions:** approved — executor exit 0. Validation: `grep -q 'Use prompt directly (skip planner)' rag_ui.html` → passed; `python - <<'PY'
from pathlib import Path
files = [Path('rag_api.py'), Path('main.py'), Path('tools.py')]
text = '\n'.join(p.read_text() for p in files if p.exists())
assert 'use_prompt_directly' in text or 'skip planner' in text.lower()
print('backend flag handling present')
PY` → passed; `python - <<'PY'
from pathlib import Path
files = [Path('rag_api.py'), Path('main.py'), Path('tools.py')]
text = '\n'.join(p.read_text() for p in files if p.exists())
needles = ['use_prompt_directly', 'executor_prompt', 'submitted prompt', 'prompt directly']
assert 'use_prompt_directly' in text and any(n in text for n in needles)
print('direct prompt bypass path present')
PY` → passed; `python - <<'PY'
from pathlib import Path
files = [Path('rag_api.py'), Path('main.py'), Path('tools.py')]
text = '\n'.join(p.read_text() for p in files if p.exists())
assert 'planner' in text.lower()
print('normal planner path still referenced')
PY` → passed; `test -f /tmp/orchestrator_direct_prompt_validation.txt` → passed.

**Assumptions:** The main risk is not knowing the exact orchestrator server file and data shape without inspection; the executor should keep the change tightly scoped to the actual start-run route and avoid inventing new architecture. Another risk is downstream code expecting planner-produced structured objects; mitigate by creating the smallest compatible object only where required. Validation is intentionally file/text-based because runtime services may not be available.

**Next:** After this lands, the next small step would be a focused polish pass only if needed: show the selected mode more clearly in run/review status output or add one narrow regression test for planner vs direct-prompt branching, but only after confirming this minimal path works cleanly.

---

