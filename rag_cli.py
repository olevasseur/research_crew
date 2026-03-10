#!/usr/bin/env python3
"""CLI for the RAG book analysis pipeline.

Usage:
    python rag_cli.py ingest <file>       [--title T] [--author A] [--book-id ID]
    python rag_cli.py ingest-folder <dir>
    python rag_cli.py summarize <book_id> [--verify]
    python rag_cli.py verify <book_id>
    python rag_cli.py compare <book_id> <book_id> [<book_id> ...]
    python rag_cli.py ask "<question>"    [--books <id,id,...>]
    python rag_cli.py inspect books
    python rag_cli.py inspect chunks <book_id> [--chapter CH]
    python rag_cli.py inspect summary <book_id>
    python rag_cli.py inspect search "<query>" [--book <book_id>]
"""

from __future__ import annotations

import argparse
import sys

from rag.config import load_config


def cmd_ingest(args):
    from rag.ingest import ingest_book
    config = load_config(args.config)
    ingest_book(
        args.file,
        config,
        title=args.title,
        author=args.author,
        book_id=args.book_id,
    )


def cmd_ingest_folder(args):
    from rag.ingest import ingest_folder
    config = load_config(args.config)
    results = ingest_folder(args.folder, config)
    print(f"\nIngested {len(results)} book(s).")


def cmd_summarize(args):
    from rag.analysis import analyse_book
    from rag.critic import verify_book_summary
    config = load_config(args.config)
    print(f"Analysing '{args.book_id}' …\n")
    analyse_book(args.book_id, config)
    if args.verify:
        print("\nRunning verification …")
        verify_book_summary(args.book_id, config)


def cmd_verify(args):
    from rag.critic import verify_book_summary
    config = load_config(args.config)
    verify_book_summary(args.book_id, config)


def cmd_compare(args):
    from rag.synthesis import compare_books
    config = load_config(args.config)
    print(f"Comparing: {', '.join(args.book_ids)} …\n")
    compare_books(args.book_ids, config)


def cmd_ask(args):
    from rag.synthesis import ask_question
    config = load_config(args.config)
    book_ids = args.books.split(",") if args.books else None
    answer = ask_question(args.question, config, book_ids=book_ids)
    print("\n" + answer)


def cmd_inspect(args):
    from rag import inspect_utils
    config = load_config(args.config)

    if args.what == "books":
        inspect_utils.inspect_books(config)
    elif args.what == "chunks":
        if not args.book_id:
            print("Usage: inspect chunks <book_id> [--chapter CH]")
            return
        inspect_utils.inspect_chunks(args.book_id, config, chapter=args.chapter)
    elif args.what == "summary":
        if not args.book_id:
            print("Usage: inspect summary <book_id>")
            return
        inspect_utils.inspect_summary(args.book_id, config)
    elif args.what == "search":
        # For "inspect search", the query can be the second positional or --query
        query = args.query or args.book_id
        if not query:
            print("Usage: inspect search \"<query>\" [--book <book_id>]")
            return
        inspect_utils.inspect_retrieval(query, config, book_id=args.book)


def main():
    parser = argparse.ArgumentParser(
        description="RAG book analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="rag_config.yaml", help="Config file path")
    sub = parser.add_subparsers(dest="command")

    # --- ingest ---
    p_ingest = sub.add_parser("ingest", help="Ingest a single book")
    p_ingest.add_argument("file", help="Path to book file (PDF, txt, md)")
    p_ingest.add_argument("--title", default=None)
    p_ingest.add_argument("--author", default=None)
    p_ingest.add_argument("--book-id", dest="book_id", default=None)
    p_ingest.set_defaults(func=cmd_ingest)

    # --- ingest-folder ---
    p_folder = sub.add_parser("ingest-folder", help="Ingest all books in a folder")
    p_folder.add_argument("folder")
    p_folder.set_defaults(func=cmd_ingest_folder)

    # --- summarize ---
    p_sum = sub.add_parser("summarize", help="Analyse and summarise a book")
    p_sum.add_argument("book_id")
    p_sum.add_argument("--verify", action="store_true", help="Run critic after summarising")
    p_sum.set_defaults(func=cmd_summarize)

    # --- verify ---
    p_ver = sub.add_parser("verify", help="Verify a book summary against sources")
    p_ver.add_argument("book_id")
    p_ver.set_defaults(func=cmd_verify)

    # --- compare ---
    p_cmp = sub.add_parser("compare", help="Cross-pollinate ideas across books")
    p_cmp.add_argument("book_ids", nargs="+")
    p_cmp.set_defaults(func=cmd_compare)

    # --- ask ---
    p_ask = sub.add_parser("ask", help="Ask a question across your book library")
    p_ask.add_argument("question")
    p_ask.add_argument("--books", default=None, help="Comma-separated book IDs to scope the search")
    p_ask.set_defaults(func=cmd_ask)

    # --- inspect ---
    p_ins = sub.add_parser("inspect", help="Inspect data and results")
    p_ins.add_argument("what", choices=["books", "chunks", "summary", "search"])
    p_ins.add_argument("book_id", nargs="?", default=None)
    p_ins.add_argument("--chapter", default=None)
    p_ins.add_argument("--query", default=None)
    p_ins.add_argument("--book", default=None, help="Scope search to a single book")
    p_ins.set_defaults(func=cmd_inspect)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
