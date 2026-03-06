from crewai import Agent, Task, Crew
from langchain_community.chat_models import ChatOllama
from tools import fetch_url_text

llm = ChatOllama(
    model="llama3.1",
    base_url="http://localhost:11434"
)

# Agent 1 — Source Finder
source_agent = Agent(
    role="Startup Researcher",
    goal="Find high quality sources discussing real problems in crypto infrastructure, DevOps security, compliance automation, wallet custody, and blockchain infrastructure.",
    backstory="You specialize in finding early signals of problems developers and startups complain about.",
    verbose=True,
    llm=llm
)

# Agent 2 — Problem Extractor
problem_agent = Agent(
    role="Problem Analyst",
    goal="Extract concrete business problems from technical discussions and articles.",
    backstory="You identify problems companies would pay to solve.",
    verbose=True,
    llm=llm
)

# Agent 3 — Product Ideator
idea_agent = Agent(
    role="Startup Founder",
    goal="Generate product ideas that could be built by a solo engineer and sold as SaaS.",
    backstory="You focus on products developers, crypto companies, or security teams would pay for.",
    verbose=True,
    llm=llm
)

# Agent 4 — Feasibility Evaluator
score_agent = Agent(
    role="Indie Hacker Advisor",
    goal="Evaluate which ideas are realistic for a solo founder seeking freedom and recurring revenue.",
    backstory="You prioritize small profitable SaaS opportunities.",
    verbose=True,
    llm=llm
)

task1 = Task(
    description="""
Find 10 URLs discussing real problems in:
- crypto custody
- wallet monitoring
- AML / compliance tooling
- blockchain node infrastructure
- DevOps security automation
- secrets rotation
- KMS management
- Terraform security

Return only URLs.
""",
    agent=source_agent
)

task2 = Task(
    description="""
From the URLs found earlier, identify the top real problems developers or companies face.

For each problem provide:
- problem description
- who experiences it
- why it matters
- link to evidence
""",
    agent=problem_agent
)

task3 = Task(
    description="""
Generate SaaS product ideas for the problems identified.

For each idea provide:
- product name
- one sentence description
- who pays
- rough pricing
- 2 week MVP scope
""",
    agent=idea_agent
)

task4 = Task(
    description="""
Score the ideas from best to worst for a solo founder.

Criteria:
- people willing to pay
- recurring revenue
- buildable part-time
- low regulatory burden
- reachable customers

Return a ranked list of the top 10.
""",
    agent=score_agent
)

crew = Crew(
    agents=[source_agent, problem_agent, idea_agent, score_agent],
    tasks=[task1, task2, task3, task4],
    verbose=True
)

result = crew.kickoff()

print(result)

