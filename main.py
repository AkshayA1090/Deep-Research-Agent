"""Gradio frontend for the AI Deep Research Agent."""

import os

import gradio as gr
from dotenv import load_dotenv

from research_agent import ResearchReport, deep_research, report_to_markdown

load_dotenv()


def _require_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise gr.Error(
            "OPENAI_API_KEY is missing. Add it to your .env file and restart."
        )


async def run_research(user_message: str, history: list):
    """Stream progress updates, then the final cited report."""
    _require_api_key()

    if not user_message or not user_message.strip():
        yield history, ""
        return

    history = history or []
    history.append({"role": "user", "content": user_message.strip()})
    history.append({"role": "assistant", "content": "Starting research…"})
    yield history, ""

    try:
        async for update in deep_research(user_message.strip()):
            if isinstance(update, str):
                history[-1]["content"] = update
                yield history, ""
            elif isinstance(update, ResearchReport):
                history[-1]["content"] = report_to_markdown(
                    user_message.strip(), update
                )
                yield history, ""
    except Exception as exc:  # noqa: BLE001
        history[-1]["content"] = f"Error: {exc}"
        yield history, ""


with gr.Blocks(title="AI Deep Research Agent") as demo:
    gr.Markdown("# AI Deep Research Agent")
    gr.Markdown(
        "Enter a **stock ticker** (e.g. `NVDA`) or a **topic**. "
        "The agent classifies the query, searches DuckDuckGo, plans 3–4 angles, "
        "runs parallel deep dives, and writes a cited report."
    )

    chatbot = gr.Chatbot(label="Research", height=640)
    msg = gr.Textbox(
        placeholder="e.g. NVDA  or  Future of solid-state batteries",
        label="Research query",
        autofocus=True,
    )
    with gr.Row():
        send = gr.Button("Research", variant="primary")
        clear = gr.Button("Clear")

    send.click(run_research, inputs=[msg, chatbot], outputs=[chatbot, msg])
    msg.submit(run_research, inputs=[msg, chatbot], outputs=[chatbot, msg])
    clear.click(lambda: ([], ""), outputs=[chatbot, msg])

if __name__ == "__main__":
    demo.launch()
