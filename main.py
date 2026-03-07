import sys
import importlib


def run_crew(crew_name):
    module = importlib.import_module(f"crews.{crew_name}.crew")
    crew = module.build_crew()
    result = crew.kickoff()
    print("\nFINAL RESULT\n")
    print(result)
    with open("results.md", "w") as f:
        f.write(str(result))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <crew_name>")
        exit()

    run_crew(sys.argv[1])
