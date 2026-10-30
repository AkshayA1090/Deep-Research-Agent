"""
AI Deep Research Agent — multi-agent orchestration with Pydantic AI.

Flow (from project prompt):
1. Classify ticker vs general query
2. Initial DuckDuckGo discovery search
3. Plan 3–4 non-overlapping research angles
4. Parallel deep dives (search + fetch pages + extract cited claims)
5. Synthesize a structured Markdown report
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import List

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

load_dotenv()

MODEL = "openai:gpt-5-mini"

# --- Data models ---


class ClassificationResult(BaseModel):
    is_ticker: bool
    subject: str = Field(
        description="Company name or clarified topic (e.g. 'NVIDIA Corporation')"
    )
    context: str = Field(
        description="Industry / topic context (e.g. 'semiconductors, GPUs, AI')"
    )


class ResearchAngle(BaseModel):
    title: str = Field(
        description="Research angle title, e.g. 'SWOT analysis'"
    )
    keywords: List[str] = Field(
        description="Search keywords for this angle"
    )
    description: str = Field(description="What to investigate for this angle")


class EvidenceItem(BaseModel):
    claim: str = Field(description="Fact, number, or claim extracted")
    source_title: str
    source_url: str


class ResearchSection(BaseModel):
    title: str
    key_findings: str = Field(description="Narrative summary of findings for this angle")
    evidence: List[EvidenceItem] = Field(
        description="Claims with the source that supports each one"
    )
    sources: List[str] = Field(description="Unique source URLs used")


class ResearchReport(BaseModel):
    executive_summary: str
    sections: List[ResearchSection]
    risks_uncertainties: str = Field(
        description="Risks, uncertainties, and conflicting information"
    )
    what_to_watch_next: List[str]


@dataclass
class ResearchDeps:
    client: httpx.AsyncClient


# --- Tools ---


async def search_duckduckgo(query: str, max_results: int = 5) -> List[dict]:
    """Run a DuckDuckGo text search; returns title / href / body dicts."""
    print(f"DDG search: {query}")
    results: List[dict] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    }
                )
    except Exception as e:  # noqa: BLE001
        print(f"Search error: {e}")
    return results


async def fetch_page(ctx: RunContext[ResearchDeps], url: str) -> str:
    """Fetch a URL and return cleaned text (capped for context size)."""
    print(f"Fetch: {url}")
    try:
        response = await ctx.deps.client.get(
            url, follow_redirects=True, timeout=12.0
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = (line.strip() for line in text.splitlines())
        text = "\n".join(line for line in lines if line)
        return text[:10_000]
    except Exception as e:  # noqa: BLE001
        return f"Error fetching {url}: {e}"


# --- Agents (lazy so missing API key fails at run time) ---

_classify_agent: Agent | None = None
_plan_agent: Agent | None = None
_worker_agent: Agent | None = None
_writer_agent: Agent | None = None


def _require_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )


def classify_agent() -> Agent:
    global _classify_agent
    if _classify_agent is None:
        _require_key()
        _classify_agent = Agent(
            MODEL,
            output_type=ClassificationResult,
            instructions=(
                "You classify research requests. "
                "Detect whether the input is a stock ticker (e.g. NVDA, AAPL, TSLA) "
                "or a general query. "
                "If a ticker: resolve to the company name and context "
                "(industry, products, themes). Example: NVDA → NVIDIA, "
                "semiconductors, GPUs, AI. "
                "If general: clarify the main topic and useful context."
            ),
        )
    return _classify_agent


def plan_agent() -> Agent:
    global _plan_agent
    if _plan_agent is None:
        _require_key()
        _plan_agent = Agent(
            MODEL,
            output_type=List[ResearchAngle],
            instructions=(
                "You are a research planner. Given a subject, context, and initial "
                "DuckDuckGo snippets, generate 3–4 distinct, non-overlapping research "
                "angles with keywords. "
                "For stocks, prefer angles like: SWOT analysis; last 12 months stock "
                "performance; competition and market positioning; latest quarterly "
                "results and forward guidance. "
                "For general topics, cover key trends, history, pros/cons, and outlook."
            ),
        )
    return _plan_agent


def worker_agent() -> Agent:
    global _worker_agent
    if _worker_agent is None:
        _require_key()
        _worker_agent = Agent(
            MODEL,
            deps_type=ResearchDeps,
            output_type=ResearchSection,
            instructions=(
                "You are a senior researcher doing a deep dive on ONE research angle. "
                "1) Use search_web (DuckDuckGo) for the angle keywords. "
                "2) Collect sources (title, url, snippet). "
                "3) Use fetch_page_content on promising URLs when snippets are thin. "
                "4) Extract key facts, numbers, and claims. "
                "5) Track which source supports which claim (fill evidence with "
                "claim + source_title + source_url). "
                "Prefer primary sources for financials (earnings releases, filings, "
                "investor relations) and reputable outlets for news. "
                "Return key_findings plus cited evidence."
            ),
        )

        @_worker_agent.tool
        async def search_web(
            ctx: RunContext[ResearchDeps], query: str
        ) -> List[dict]:
            """Search DuckDuckGo; returns title, url, snippet for each hit."""
            return await search_duckduckgo(query)

        @_worker_agent.tool
        async def fetch_page_content(
            ctx: RunContext[ResearchDeps], url: str
        ) -> str:
            """Fetch and extract readable text from a URL."""
            return await fetch_page(ctx, url)

    return _worker_agent


def writer_agent() -> Agent:
    global _writer_agent
    if _writer_agent is None:
        _require_key()
        _writer_agent = Agent(
            MODEL,
            output_type=ResearchReport,
            instructions=(
                "You are a professional research report writer. "
                "Synthesize the provided sections into one detailed report with: "
                "1) Executive summary "
                "2) Key findings per section (preserve evidence bullets with "
                "citations: url + source title) "
                "3) Risks / uncertainties and conflicting information "
                "4) A 'What to watch next' list. "
                "Keep citations intact. Prefer clarity over fluff."
            ),
        )
    return _writer_agent


# --- Orchestration ---


async def deep_research(user_input: str):
    """
    Yields status strings, then a final ResearchReport.
    """
    async with httpx.AsyncClient(
        headers={"User-Agent": "DeepResearchAgent/1.0"}
    ) as client:
        deps = ResearchDeps(client=client)

        yield "Analyzing request…"
        classification = (
            await classify_agent().run(f"Classify this input: {user_input!r}")
        ).output
        subject, context = classification.subject, classification.context
        kind = "ticker" if classification.is_ticker else "topic"
        yield f"Identified ({kind}): {subject} — {context}"

        yield "Running initial DuckDuckGo discovery search…"
        initial = await search_duckduckgo(
            f"{subject} {context} overview", max_results=5
        )
        initial_context = "\n".join(
            f"- {r['title']}: {r['snippet']} ({r['url']})" for r in initial
        )

        yield "Planning 3–4 research angles…"
        angles = (
            await plan_agent().run(
                f"Subject: {subject}\nContext: {context}\n"
                f"Initial search results:\n{initial_context}"
            )
        ).output
        yield f"Angles: {', '.join(a.title for a in angles)}"

        yield f"Starting parallel deep dives ({len(angles)} angles)…"

        async def process_angle(angle: ResearchAngle) -> ResearchSection:
            prompt = (
                f"Subject: {subject}\n"
                f"Angle: {angle.title}\n"
                f"Description: {angle.description}\n"
                f"Keywords: {', '.join(angle.keywords)}\n"
                "Search, fetch pages as needed, extract cited claims."
            )
            return (await worker_agent().run(prompt, deps=deps)).output

        sections = await asyncio.gather(*[process_angle(a) for a in angles])

        yield "Writing final report…"
        synthesis = f"Subject: {subject}\nContext: {context}\n\n"
        for section in sections:
            synthesis += f"## {section.title}\n{section.key_findings}\n"
            for ev in section.evidence:
                synthesis += (
                    f"- {ev.claim} "
                    f"[{ev.source_title}]({ev.source_url})\n"
                )
            synthesis += f"Sources: {', '.join(section.sources)}\n\n"

        report = (await writer_agent().run(synthesis)).output
        yield report


def report_to_markdown(query: str, report: ResearchReport) -> str:
    """Render ResearchReport as Markdown for the UI."""
    md = f"# Research Report: {query}\n\n"
    md += f"## Executive Summary\n{report.executive_summary}\n\n"
    for section in report.sections:
        md += f"## {section.title}\n\n"
        md += f"### Key findings\n{section.key_findings}\n\n"
        if section.evidence:
            md += "### Evidence\n"
            for ev in section.evidence:
                md += (
                    f"- {ev.claim} — "
                    f"[{ev.source_title}]({ev.source_url})\n"
                )
            md += "\n"
        if section.sources:
            md += (
                "**Sources:**\n"
                + "\n".join(f"- {s}" for s in section.sources)
                + "\n\n"
            )
    md += f"## Risks & Uncertainties\n{report.risks_uncertainties}\n\n"
    md += "## What to Watch Next\n"
    md += "\n".join(f"- {item}" for item in report.what_to_watch_next)
    return md
