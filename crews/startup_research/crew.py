from pathlib import Path

from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama

from tools import fetch_webpage


BASE_DIR = Path(__file__).parent


def load_prompt(path):
    with open(path, "r") as f:
        return f.read()


def read_seed_urls():
    path = BASE_DIR / "seed_urls.txt"
    return [x.strip() for x in open(path) if x.strip()]


def build_crew():

    llm = Ollama(model="llama3")

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

    # ---- agents ----

    researcher = Agent(
        role="Startup Pain Point Researcher",
        goal="Find real complaints and inefficiencies",
        backstory=researcher_prompt,
        tools=[fetch_webpage],
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
        agent=researcher,
    )

    extract_problems = Task(
        description=extract_problems_desc,
        agent=problem_analyst,
    )

    generate_ideas = Task(
        description=generate_ideas_desc,
        agent=founder,
    )

    rank_ideas = Task(
        description=rank_ideas_desc,
        agent=evaluator,
    )

    # ---- crew ----

    crew = Crew(
        agents=[researcher, problem_analyst, founder, evaluator],
        tasks=[find_sources, extract_problems, generate_ideas, rank_ideas],
        process=Process.sequential,
        verbose=True,
    )

    return crew
