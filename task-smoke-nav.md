# Task: Add navigation commands to validate_rag.sh

`scripts/validate_rag.sh` is the project's smoke test. It tests summarize, inspect,
and evaluate, but does NOT test the navigation commands added in iter 10/11.

Add `trace` and `explore` to the smoke test so they are covered.

## What to add

After step 11 (or at a logical place in the existing sequence), add:

```
echo "=== 12. Trace idea through summaries ==="
"$PYTHON" rag_cli.py trace "$BOOK_ID" --idea "attention"

echo ""
echo "=== 13. Explore one section ==="
"$PYTHON" rag_cli.py explore "$BOOK_ID" --section "Chapter 1: A Lopsided Arms Race"

echo ""
echo "=== 14. Trace with --show sections only ==="
"$PYTHON" rag_cli.py trace "$BOOK_ID" --idea "solitude" --show sections --limit 5

echo ""
echo "=== 15. Explore with --show windows only ==="
"$PYTHON" rag_cli.py explore "$BOOK_ID" --section "Introduction" --show windows
```

## Constraints

- Only touch `scripts/validate_rag.sh`
- Do not re-ingest or re-summarize
- Do not change any Python source
- Run the script to verify the new steps pass before finishing

## Validation

```
bash scripts/validate_rag.sh digital-minimalism 2>&1 | tail -30
```

Both `trace` and `explore` steps should complete without error.
