#!/usr/bin/env bash
# Run all RAG validation commands. From project root:
#   bash scripts/validate_rag.sh digital-minimalism
# Or from anywhere:
#   bash /path/to/research_crew/scripts/validate_rag.sh digital-minimalism

set -e
BOOK_ID="${1:-digital-minimalism}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
VENV="${VENV:-/Users/aiagent/venv}"
PYTHON="${VENV}/bin/python"

echo "=== 1. Default summarize ==="
"$PYTHON" rag_cli.py summarize "$BOOK_ID" --quality default --force

echo ""
echo "=== 2. Inspect selection ==="
"$PYTHON" rag_cli.py inspect selection "$BOOK_ID"

echo ""
echo "=== 3. Inspect windows ==="
"$PYTHON" rag_cli.py inspect windows "$BOOK_ID"

echo ""
echo "=== 4. Inspect summary-meta ==="
"$PYTHON" rag_cli.py inspect summary-meta "$BOOK_ID"

echo ""
echo "=== 5. Evaluate book summary ==="
"$PYTHON" rag_cli.py evaluate "$BOOK_ID"

echo ""
echo "=== 6. Fast summarize ==="
"$PYTHON" rag_cli.py summarize "$BOOK_ID" --quality fast --force

echo ""
echo "=== 7. Thorough summarize ==="
"$PYTHON" rag_cli.py summarize "$BOOK_ID" --quality thorough --force

echo ""
echo "=== 8. Summarize one section ==="
"$PYTHON" rag_cli.py summarize "$BOOK_ID" --section "Chapter 1: A Lopsided Arms Race"

echo ""
echo "=== 9. Evaluate one section ==="
"$PYTHON" rag_cli.py evaluate "$BOOK_ID" --section "Chapter 1: A Lopsided Arms Race"

echo ""
echo "=== 10. Cache reuse (no --force) ==="
"$PYTHON" rag_cli.py summarize "$BOOK_ID" --quality default

echo ""
echo "=== 11. Inspect summary-meta (final) ==="
"$PYTHON" rag_cli.py inspect summary-meta "$BOOK_ID"

echo ""
echo "Done."
