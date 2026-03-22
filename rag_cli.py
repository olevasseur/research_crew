#!/usr/bin/env python3
"""CLI for the RAG book analysis pipeline.

Usage:
    python rag_cli.py ingest <file>       [--title T] [--author A] [--book-id ID]
    python rag_cli.py ingest-folder <dir>
    python rag_cli.py summarize <book_id> [--mode M] [--quality Q] [--section S] [--force] [--verify] [--no-verify]
    python rag_cli.py evaluate <book_id>  [--section S]
    python rag_cli.py trace <book_id> --idea "<query>" [--show both|sections|windows] [--limit N]
    python rag_cli.py explore <book_id> --section "<section>" [--show all|summary|windows]
    python rag_cli.py verify <book_id>
    python rag_cli.py compare <book_id> <book_id> [<book_id> ...]
    python rag_cli.py ask "<question>"    [--books <id,id,...>]
    python rag_cli.py inspect-window <book_id> --window N [--section "<section>"]
    python rag_cli.py inspect books
    python rag_cli.py inspect structure <book_id>
    python rag_cli.py inspect chunks <book_id> [--chapter CH]
    python rag_cli.py inspect subchunks <book_id> --chapter <section_name>
    python rag_cli.py inspect windows <book_id> [--chapter <section_name>]
    python rag_cli.py inspect selection <book_id> [--chapter <section_name>]
    python rag_cli.py inspect summary <book_id>
    python rag_cli.py inspect summary-meta <book_id>
    python rag_cli.py inspect search "<query>" [--book <book_id>]
    python rag_cli.py inspect-structure <file>  [--debug]
    python rag_cli.py --version
"""

from __future__ import annotations

import argparse
import sys

if {"-V", "--version"} & set(sys.argv[1:]):
    print("rag-pipeline 1.0.0")
    sys.exit(0)

from rag.config import load_config


def cmd_ingest(args):
    from rag.ingest import ingest_book
    config = load_config(args.config)
    ingest_book(
        args.file, config,
        title=args.title, author=args.author, book_id=args.book_id,
    )


def cmd_ingest_folder(args):
    from rag.ingest import ingest_folder
    config = load_config(args.config)
    results = ingest_folder(args.folder, config)
    print(f"\nIngested {len(results)} book(s).")


def cmd_summarize(args):
    from rag.analysis import analyse_book, SUMMARIZE_MODES
    config = load_config(args.config)

    mode = args.mode
    if mode not in SUMMARIZE_MODES:
        print(f"Unknown mode '{mode}'. Available: {', '.join(SUMMARIZE_MODES)}")
        return

    include_types = args.include.split(",") if args.include else None
    exclude_types = args.exclude.split(",") if args.exclude else None

    verify = None
    if args.verify:
        verify = True
    elif args.no_verify:
        verify = False

    quality = args.quality
    section = args.section
    print(f"Analysing '{args.book_id}' (mode={mode}, quality={quality}, "
          f"section={section or 'all'}, force={args.force}) …\n")
    analyse_book(
        args.book_id, config,
        mode=mode, quality=quality,
        section_filter=section,
        include_types=include_types, exclude_types=exclude_types,
        force=args.force, verify=verify,
    )


def cmd_evaluate(args):
    from rag.evaluate import evaluate_section, evaluate_book
    config = load_config(args.config)

    if args.section:
        print(f"Evaluating section '{args.section}' of '{args.book_id}' …\n")
        result = evaluate_section(args.book_id, args.section, config)
    else:
        print(f"Evaluating book summary for '{args.book_id}' …\n")
        result = evaluate_book(args.book_id, config)

    print(result)


def cmd_trace(args):
    from rag.navigation import trace_idea
    config = load_config(args.config)
    trace_idea(
        args.book_id, args.idea, config,
        limit=args.limit, show=args.show,
    )


def cmd_explore(args):
    from rag.navigation import explore_section
    config = load_config(args.config)
    explore_section(
        args.book_id, args.section, config,
        show=args.show,
        show_windows=args.windows,
    )


def cmd_inspect_window(args):
    from rag import inspect_utils
    config = load_config(args.config)
    inspect_utils.inspect_window(args.book_id, args.window, config, section=args.section)


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


def cmd_inspect_structure(args):
    """Dry-run structure detection on a file without ingesting."""
    from rag.ingest import load_file, clean_text
    from rag.chunker import detect_structure
    pages = load_file(args.file)
    for p in pages:
        p.text = clean_text(p.text)
    sections, all_matches = detect_structure(pages, debug=args.debug)
    print(f"\nDetected {len(sections)} section(s) from {len(pages)} pages:\n")
    for s in sections:
        page_span = s.end_page - s.start_page + 1
        print(f"  p.{s.start_page:>3d}-{s.end_page:<3d}  ({page_span:>3d} pp)  "
              f"[{s.kind.value:12s}]  conf={s.confidence:.2f}  {s.name}")
        print(f"    reason: {s.detection_reason}")
    if args.debug:
        print(f"\nAll heading matches ({len(all_matches)}):")
        for m in all_matches:
            print(f"  p.{m.page:>3d}  conf={m.confidence:.2f}  "
                  f"{m.heading_type:20s}  {m.label!r}")
            print(f"    reason: {m.reason}")


def cmd_inspect(args):
    from rag import inspect_utils
    config = load_config(args.config)

    if args.what == "books":
        inspect_utils.inspect_books(config)
    elif args.what == "chunks":
        if not args.book_id:
            print("Usage: inspect chunks <book_id> [--chapter <section>]")
            return
        inspect_utils.inspect_chunks(args.book_id, config, chapter=args.chapter)
    elif args.what == "subchunks":
        if not args.book_id or not args.chapter:
            print("Usage: inspect subchunks <book_id> --chapter <section_name>")
            return
        inspect_utils.inspect_subchunks(args.book_id, args.chapter, config)
    elif args.what == "windows":
        if not args.book_id:
            print("Usage: inspect windows <book_id> [--chapter <section_name>]")
            return
        inspect_utils.inspect_windows(args.book_id, config, section=args.chapter)
    elif args.what == "selection":
        if not args.book_id:
            print("Usage: inspect selection <book_id> [--chapter <section_name>]")
            return
        inspect_utils.inspect_selection(args.book_id, config, section=args.chapter)
    elif args.what == "summary":
        if not args.book_id:
            print("Usage: inspect summary <book_id>")
            return
        inspect_utils.inspect_summary(args.book_id, config)
    elif args.what == "summary-meta":
        if not args.book_id:
            print("Usage: inspect summary-meta <book_id>")
            return
        inspect_utils.inspect_summary_meta(args.book_id, config)
    elif args.what == "structure":
        if not args.book_id:
            print("Usage: inspect structure <book_id>")
            return
        inspect_utils.inspect_structure(args.book_id, config)
    elif args.what == "window":
        if not args.book_id or args.window is None:
            print("Usage: inspect window <book_id> --window <id>")
            return
        inspect_utils.inspect_window(args.book_id, args.window, config, section=args.chapter)
    elif args.what == "search":
        query = args.query or args.book_id
        if not query:
            print('Usage: inspect search "<query>" [--book <book_id>]')
            return
        inspect_utils.inspect_retrieval(query, config, book_id=args.book)


def main():
    parser = argparse.ArgumentParser(
        description="RAG book analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-V", "--version", action="version", version="rag-pipeline 1.0.0")
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
    p_sum.add_argument("--mode", default="default",
                       choices=["default", "body-only", "chapter-only", "full", "back-matter"],
                       help="Which sections to include")
    p_sum.add_argument("--quality", default="default",
                       choices=["fast", "default", "thorough"],
                       help="fast=3/section, default=6/section, thorough=all windows")
    p_sum.add_argument("--section", default=None,
                       help="Summarize only this one section (exact name)")
    p_sum.add_argument("--include", default=None,
                       help="Extra section_types to include (comma-separated)")
    p_sum.add_argument("--exclude", default=None,
                       help="Section_types to exclude (comma-separated)")
    p_sum.add_argument("--force", action="store_true",
                       help="Clear cache and recompute everything")
    p_sum.add_argument("--verify", action="store_true",
                       help="Run verification after summarising")
    p_sum.add_argument("--no-verify", action="store_true",
                       help="Skip verification even if config enables it")
    p_sum.set_defaults(func=cmd_summarize)

    # --- evaluate ---
    p_eval = sub.add_parser("evaluate", help="Evaluate summary quality against sources")
    p_eval.add_argument("book_id")
    p_eval.add_argument("--section", default=None,
                        help="Evaluate one section (exact name). Omit for book-level evaluation.")
    p_eval.set_defaults(func=cmd_evaluate)

    # --- explore ---
    p_expl = sub.add_parser("explore", help="Explore one section in structured detail")
    p_expl.add_argument("book_id")
    p_expl.add_argument("--section", required=True, help="Section name to explore")
    p_expl.add_argument("--show", default="all", choices=["all", "summary", "windows"],
                        help="Show summary subsections, window previews, or both")
    p_expl.add_argument("--windows", type=int, default=3,
                        help="Max selected windows to display (0 = all, default 3)")
    p_expl.set_defaults(func=cmd_explore)

    # --- trace ---
    p_trace = sub.add_parser("trace", help="Trace an idea through a book's summaries")
    p_trace.add_argument("book_id")
    p_trace.add_argument("--idea", required=True, help="Idea or concept to trace")
    p_trace.add_argument("--show", default="both", choices=["both", "sections", "windows"],
                         help="Show section summaries, window summaries, or both")
    p_trace.add_argument("--limit", type=int, default=20,
                         help="Max number of matching sections to display")
    p_trace.set_defaults(func=cmd_trace)

    # --- verify ---
    p_ver = sub.add_parser("verify", help="Verify a book summary against source chunks")
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

    # --- inspect-window ---
    p_iw = sub.add_parser("inspect-window", help="Zoom into a window: full summary + original text")
    p_iw.add_argument("book_id")
    p_iw.add_argument("--window", type=int, required=True, help="1-based window index")
    p_iw.add_argument("--section", default=None,
                      help="Section name (exact, normalized, or unambiguous partial)")
    p_iw.set_defaults(func=cmd_inspect_window)

    # --- inspect-structure ---
    p_struct = sub.add_parser("inspect-structure", help="Dry-run structure detection on a file")
    p_struct.add_argument("file", help="Path to book file (PDF, txt, md)")
    p_struct.add_argument("--debug", action="store_true", help="Show all candidates and rejections")
    p_struct.set_defaults(func=cmd_inspect_structure)

    # --- inspect ---
    p_ins = sub.add_parser("inspect", help="Inspect data and results")
    p_ins.add_argument("what", choices=[
        "books", "chunks", "subchunks", "windows", "window", "selection",
        "summary", "summary-meta", "search", "structure",
    ])
    p_ins.add_argument("book_id", nargs="?", default=None)
    p_ins.add_argument("--chapter", default=None)
    p_ins.add_argument("--query", default=None)
    p_ins.add_argument("--book", default=None, help="Scope search to a single book")
    p_ins.add_argument("--window", type=int, default=None,
                       help="1-based window index to zoom into (use with 'window')")
    p_ins.set_defaults(func=cmd_inspect)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
