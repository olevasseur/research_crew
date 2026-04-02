# research_crew

AI research agents that discover startup ideas from real technical problems.

Runs locally using:
- CrewAI
- Ollama (local LLM)
- Python

No external APIs required.

---

# 1. Install Ollama

macOS:

brew install ollama

Start the server:

ollama serve

Download a model:

ollama pull llama3.1

Optional smaller/faster models:

ollama pull mistral
ollama pull qwen2.5

---

# 2. Create Python environment

cd research_crew

python3.13 -m venv .venv

Activate:

source .venv/bin/activate

Upgrade pip:

python -m pip install --upgrade pip setuptools wheel

---

# 3. Install dependencies

python -m pip install \
crewai \
crewai-tools \
requests \
beautifulsoup4 \
lxml

Save dependencies:

python -m pip freeze > requirements.txt

---

# 4. Project structure

research_crew/
│
├── crew.py
├── tools.py
├── seed_urls.txt
│
├── agents/
│   ├── researcher.md
│   ├── problem_analyst.md
│   ├── founder.md
│   └── evaluator.md
│
└── requirements.txt

---

# 5. Add seed URLs

Edit:

seed_urls.txt

Example:

https://news.ycombinator.com/
https://blog.cloudflare.com/
https://aws.amazon.com/blogs/security/
https://github.blog/engineering/
https://www.reddit.com/r/devops.json

---

# 6. Run the crew

Make sure Ollama is running.

Then:

source .venv/bin/activate

python crew.py

---

# 7. Expected output

The crew will:

1. Read seed URLs
2. Extract real technical problems
3. Generate product ideas
4. Rank ideas for solo founders

Example output:

Top Startup Opportunities

1. WalletWatch
   Monitoring for suspicious crypto wallet activity
   Target: crypto startups
   Price: $79/month

2. Terraform Guardrail
   Automated security policy enforcement for Terraform

3. NodeOps Monitor
   Monitoring for blockchain node reliability

---

# 8. Development tips

If you change dependencies:

python -m pip freeze > requirements.txt

If you recreate the environment:

python -m pip install -r requirements.txt

---

# 9. RAG pipeline — running tests

The RAG pipeline lives in `rag/` and `rag_cli.py`. Tests do not call the LLM or
read any book data; they run entirely offline.

**Preferred ingest format:** EPUB — gives exact chapter structure, clean text,
and auto-detected title/author. PDF is supported but best-effort (quality
depends on the PDF).

Bootstrap (if `.venv` does not exist yet):

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Run tests (works whether or not `.venv` is active — use the Python on `PATH`):

    python3 -m pytest tests/test_review_eval.py -v

Or, if a venv is active:

    python -m pytest tests/ -v

Do **not** call `.venv/bin/pytest` directly; use `python -m pytest` so that the
correct interpreter and installed packages are always used.

---

# 10. Future improvements

Possible upgrades:

- GitHub issue mining
- Hacker News scraping
- Reddit API integration
- clustering similar ideas
- automated landing page generation
