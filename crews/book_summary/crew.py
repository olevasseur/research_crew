from pathlib import Path

from crewai import Agent, Task, Crew, Process, LLM

from tools import read_book, get_book_info


BASE_DIR = Path(__file__).parent
RESULTS_DIR = Path("crews/book_summary/results")


def load_prompt(path):
    with open(path, "r") as f:
        return f.read()


def build_crew(book_path: str = ""):

    llm = LLM(
        model="ollama/llama3.1",
        base_url="http://localhost:11434"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- load prompts ----

    reader_prompt = load_prompt(BASE_DIR / "agents/reader.md")
    analyst_prompt = load_prompt(BASE_DIR / "agents/analyst.md")
    synthesizer_prompt = load_prompt(BASE_DIR / "agents/synthesizer.md")

    map_chapters_desc = load_prompt(BASE_DIR / "tasks/map_chapters.md")
    extract_insights_desc = load_prompt(BASE_DIR / "tasks/extract_insights.md")
    synthesize_desc = load_prompt(BASE_DIR / "tasks/synthesize_summary.md")

    # ---- agents ----

    reader = Agent(
        role="Book Reader & Chapter Mapper",
        goal="Read the book and produce an accurate chapter map",
        backstory=reader_prompt,
        tools=[get_book_info, read_book],
        verbose=True,
        llm=llm,
    )

    analyst = Agent(
        role="Book Insight Analyst",
        goal="Extract structured insights from every chapter",
        backstory=analyst_prompt,
        tools=[read_book],
        verbose=True,
        llm=llm,
    )

    synthesizer = Agent(
        role="Book Summary Synthesizer",
        goal="Produce a polished, cross-referenceable book summary",
        backstory=synthesizer_prompt,
        verbose=True,
        llm=llm,
    )

    # ---- tasks ----

    map_chapters = Task(
        description=map_chapters_desc.replace("{book_path}", book_path),
        expected_output=(
            "At the top: total_pages (from get_book_info). Then a numbered list of every "
            "chapter/section with: chapter number, title, page start, page end, one-sentence summary."
        ),
        agent=reader,
    )

    extract_insights = Task(
        description=extract_insights_desc.replace("{book_path}", book_path),
        expected_output=(
            "For each chapter: core ideas, key examples with context, "
            "frameworks/mental models, actionable takeaways, and notable quotes."
        ),
        agent=analyst,
        output_file=str(RESULTS_DIR / "chapter_insights.md"),
    )

    synthesize_summary = Task(
        description=synthesize_desc,
        expected_output=(
            "A structured book summary with: book metadata, chapter-by-chapter summary, "
            "cross-cutting themes, frameworks quick-reference, top actionable items, "
            "and connections/building blocks for cross-book synthesis."
        ),
        agent=synthesizer,
        output_file=str(RESULTS_DIR / "book_summary.md"),
    )

    # ---- crew ----

    crew = Crew(
        agents=[reader, analyst, synthesizer],
        tasks=[map_chapters, extract_insights, synthesize_summary],
        process=Process.sequential,
        verbose=True,
    )

    return crew
