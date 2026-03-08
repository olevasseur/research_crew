import sys
import importlib
from pathlib import Path


def run_crew(crew_name):
    module = importlib.import_module(f"crews.{crew_name}.crew")
    crew = module.build_crew()
    result = crew.kickoff()
    print("\nFINAL RESULT\n")
    print(result)

    results_dir = Path("crews") / crew_name / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "final_ranked_ideas.md").write_text(str(result))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <crew_name>")
        exit()

    run_crew(sys.argv[1])
