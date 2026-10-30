# AI Deep Research Agent

Multi-agent deep research with [Pydantic AI](https://ai.pydantic.dev/), **OpenAI gpt-5-mini**, **DuckDuckGo**, and a **Gradio** UI.

## What it does

1. **Classify** — ticker (e.g. `NVDA` → NVIDIA, semiconductors, GPUs, AI) vs general topic  
2. **Discover** — one DuckDuckGo overview search  
3. **Plan** — 3–4 non-overlapping angles (SWOT, performance, competition, earnings, …)  
4. **Deep dive (parallel)** — per angle: search, fetch pages, extract facts with **claim → source** tracking  
5. **Synthesize** — executive summary, key findings, cited evidence, risks, “what to watch next”  
   Prefers primary financial sources (earnings, filings, IR) and reputable news.

## Prerequisites

- Python 3.10+
- [OpenAI API key](https://platform.openai.com/api-keys)

## Setup

```bash
cd deep-research-agent
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Put your key in `.env`:

```env
OPENAI_API_KEY=sk-your-real-key-here
```

## Run

```bash
python main.py
```

Open **http://127.0.0.1:7860**

### Example queries

- `NVDA`, `TSLA`, `AAPL`
- `Future of solid-state batteries`
- `Impact of AI on healthcare`

## Project files

| File | Purpose |
|------|---------|
| `research_agent.py` | Classify → plan → parallel research → report |
| `main.py` | Gradio frontend |
| `.env` | OpenAI API key |
| `requirements.txt` | Dependencies |
| `README.md` | This guide |

Patterns follow `docs.md` in the sibling `Pydantic-AI-Deep-Research-Agent` folder (Agent, tools, structured `output_type`).
