from pathlib import Path

from crewai import Agent, Task, Crew, Process, LLM

from tools import fetch_webpage, extract_links


BASE_DIR = Path(__file__).parent
# Relative to project root so paths are consistent (run from repo root)
RESULTS_DIR = Path("crews/startup_research/results")


def load_prompt(path):
    with open(path, "r") as f:
        return f.read()


def read_seed_urls():
    path = BASE_DIR / "seed_urls.txt"
    return [x.strip() for x in open(path) if x.strip()]


def build_crew():

    llm = LLM(
        model="ollama/llama3.1",
        base_url="http://localhost:11434"
    )

    # ---- load prompts ----

    researcher_prompt = load_prompt(BASE_DIR / "agents/researcher.md")
    problem_prompt = load_prompt(BASE_DIR / "agents/problem_analyst.md")
    founder_prompt = load_prompt(BASE_DIR / "agents/founder.md")
    evaluator_prompt = load_prompt(BASE_DIR / "agents/evaluator.md")

    find_sources_desc = load_prompt(BASE_DIR / "tasks/find_sources.md")
    extract_problems_desc = load_prompt(BASE_DIR / "tasks/extract_problems.md")
    generate_ideas_desc = load_prompt(BASE_DIR / "tasks/generate_ideas.md")
    rank_ideas_desc = load_prompt(BASE_DIR / "tasks/rank_ideas.md")

    seed_urls = read_seed_urls()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- agents ----

    researcher = Agent(
        role="Startup Pain Point Researcher",
        goal="Find real complaints and inefficiencies",
        backstory=researcher_prompt,
        tools=[fetch_webpage, extract_links],
        verbose=True,
        llm=llm,
    )

    problem_analyst = Agent(
        role="Problem Analyst",
        goal="Extract structured problems",
        backstory=problem_prompt,
        verbose=True,
        llm=llm,
    )

    founder = Agent(
        role="Solo Founder Product Designer",
        goal="Generate startup ideas",
        backstory=founder_prompt,
        verbose=True,
        llm=llm,
    )

    evaluator = Agent(
        role="Startup Opportunity Evaluator",
        goal="Rank startup opportunities",
        backstory=evaluator_prompt,
        verbose=True,
        llm=llm,
    )

    # ---- tasks ----

    find_sources = Task(
        description=f"{find_sources_desc}\n\nSeed URLs:\n{seed_urls}",
        expected_output=(
            "A structured research summary of the seed URLs, including source URL, "
            "main topic, notable complaints, inefficiencies, compliance burdens, "
            "manual workflows, and repeated pain points."
        ),
        agent=researcher,
    )

    extract_problems = Task(
        description=extract_problems_desc,
        expected_output=(
            "A deduplicated list of real problems extracted from the research, with "
            "for each problem: title, affected user, why it hurts, current workaround, "
            "urgency, frequency, and evidence from the sources."
        ),
        agent=problem_analyst,
        output_file=str(RESULTS_DIR / "raw_evidence.md"),
    )

    generate_ideas = Task(
        description=generate_ideas_desc,
        expected_output=(
            "A list of startup ideas mapped to the extracted problems. For each idea, "
            "include problem solved, target user, product concept, business model, "
            "why a user would pay, and why it fits a solo founder."
        ),
        agent=founder,
    )

    rank_ideas = Task(
        description=rank_ideas_desc,
        expected_output=(
            "A ranked list of the best startup ideas with scoring and justification. "
            "For each idea include: rank, idea name, a clear description of what the product is and what it does (so the reader understands without prior context), "
            "strengths, weaknesses, leverage, recurring revenue potential, founder fit, and a final recommendation."
        ),
        agent=evaluator,
        output_file=str(RESULTS_DIR / "final_ranked_ideas.md"),
    )

    # ---- crew ----

    crew = Crew(
        agents=[researcher, problem_analyst, founder, evaluator],
        tasks=[find_sources, extract_problems, generate_ideas, rank_ideas],
        process=Process.sequential,
        verbose=True,
    )

    return crew
