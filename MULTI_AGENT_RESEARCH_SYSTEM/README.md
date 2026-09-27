# Aether — Autonomous Research Desk

Aether is a multi-agent research pipeline with a Streamlit front end. Give it a topic and four agents run in sequence — search, read, write, and critique — producing a structured report you can read, download, and review.

## How it works

1. **Scout** — searches the web for recent, credible sources on the topic (Tavily).
2. **Reader** — opens the strongest result and extracts its full text.
3. **Scribe** — drafts a structured report (introduction, key findings, conclusion, sources) from what was found.
4. **Critic** — reviews the report and returns a score out of 10 with strengths and areas to improve.

The Streamlit UI shows each stage as it completes, then displays the final report, the critic's review, and the sources consulted.

## Project structure

```
app.py                    Streamlit UI
src/Agents/agents.py      LLM setup, agent builders, writer/critic chains
src/Pipeline/pipeline.py  orchestrates the four stages in order
src/tools/tools.py        search_web and scrape_url tools
.streamlit/config.toml    theme colors
requirements.txt
.env.example
```

## Requirements

- Python 3.10+
- A Google Gemini API key
- A Tavily API key

## Setup

```bash
git clone <this-repo-url>
cd aether-research

conda create -n name python=3.11 -y
conda asctivate name

pip install -r requirements.txt

cp .env.example .env
# then edit .env and add your TAVILY_API_KEY and GEMINI_API_KEY
```

## Running

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

Without API keys configured, the app still loads. Use **Load Sample Dossier** in the sidebar to see a full example report, critique, and source list without making any live calls.

## Environment variables

| Variable          | Purpose                                   |
|-------------------|--------------------------------------------|
| `GEMINI_API_KEY`  | Used by the Scribe and Critic (and agents) |
| `TAVILY_API_KEY`  | Used by the Scout for web search           |

Set these in a `.env` file in the project root, or as real environment variables in your deployment platform's secrets manager.

## Notes

- The Gemini model name is set in `src/Agents/agents.py`. Confirm it matches a model currently available to your account before deploying — model names are renamed and retired independently of this code.
- The LLM client is constructed lazily, on first use, not at import time. This means the app (and the sample dossier path) still loads and runs correctly even if API keys are missing.
- Agent calls run synchronously, so a live research run blocks the Streamlit script thread until it completes. The sidebar tracker updates between stages, not mid-stage.
- On Windows, if you hit an `ssl.create_default_context` / `FileNotFoundError` on startup, it usually means a stale `SSL_CERT_FILE` or `REQUESTS_CA_BUNDLE` environment variable, or a corrupted `certifi` install. Reinstalling certifi (`pip install --upgrade --force-reinstall certifi`) resolves most cases.

## License

Add a license of your choice (for example MIT) before publishing this repository publicly.

ADITHYA UBALE