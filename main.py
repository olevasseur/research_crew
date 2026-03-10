import inspect
import sys
import importlib
from pathlib import Path


RESULT_FILENAMES = {
    "startup_research": "final_ranked_ideas.md",
    "book_summary": "book_summary.md",
}


def run_crew(crew_name, extra_args):
    module = importlib.import_module(f"crews.{crew_name}.crew")

    build_sig = inspect.signature(module.build_crew)
    kwargs = {}
    for name, param in build_sig.parameters.items():
        if name in extra_args:
            kwargs[name] = extra_args[name]

    crew = module.build_crew(**kwargs)
    result = crew.kickoff()
    print("\nFINAL RESULT\n")
    print(result)

    results_dir = Path("crews") / crew_name / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    filename = RESULT_FILENAMES.get(crew_name, "result.md")
    (results_dir / filename).write_text(str(result))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <crew_name> [--key value ...]")
        print("Examples:")
        print("  python main.py startup_research")
        print("  python main.py book_summary --book_path /path/to/book.pdf")
        exit()

    crew_name = sys.argv[1]
    extra_args = {}
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i].startswith("--") and i + 1 < len(args):
            extra_args[args[i].lstrip("-")] = args[i + 1]
            i += 2
        else:
            i += 1

    run_crew(crew_name, extra_args)
