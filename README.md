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

# 9. Future improvements

Possible upgrades:

- GitHub issue mining
- Hacker News scraping
- Reddit API integration
- clustering similar ideas
- automated landing page generation
