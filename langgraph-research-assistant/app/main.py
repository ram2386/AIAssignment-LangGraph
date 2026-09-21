"""Command Line Interface (CLI) entrypoint for LangGraph Research Assistant.

Usage:
    Interactive mode:
        python -m app.main

    Direct question mode:
        python -m app.main --question "What are the benefits of Model Context Protocol?"
"""

import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

# Load environment variables from .env file before imports
load_dotenv()

from app.graph import create_research_graph
from app.state import create_initial_state

# Setup clean console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
# Silence verbose third-party loggers
for quiet_logger in ["httpx", "httpcore", "mcp", "google"]:
    logging.getLogger(quiet_logger).setLevel(logging.WARNING)


def print_banner():
    """Print welcoming CLI banner."""
    print("\n" + "=" * 60)
    print("        LANGGRAPH MULTI-AGENT RESEARCH ASSISTANT")
    print("=" * 60)
    provider = os.getenv("LLM_PROVIDER", "gemini")
    print(f"Provider: {provider.upper()} | Python: {sys.version.split()[0]}")
    print("=" * 60 + "\n")


def display_results(result: dict):
    """Format and display research workflow results."""
    question = result.get("question", "")
    plan = result.get("research_plan", [])
    summary = result.get("summary", "")
    final_answer = result.get("final_answer", "")
    search_results = result.get("search_results", [])
    retry_count = result.get("retry_count", 0)
    errors = result.get("errors", [])

    print("\n" + "=" * 60)
    print("RESEARCH RESULTS")
    print("=" * 60)

    print("\n[Question]:")
    print(question)

    print("\n[Research Plan]:")
    if plan:
        for idx, step in enumerate(plan, 1):
            print(f"  {idx}. {step}")
    else:
        print("  (No plan generated)")

    if retry_count > 0:
        print(f"\n[Retrieval Retries Executed]: {retry_count}")

    print("\n[Research Summary]:")
    print(summary.strip() if summary else "  (No summary generated)")

    print("\n[Final Answer]:")
    print(final_answer.strip() if final_answer else "  (No answer generated)")

    print("\n[Sources Retrieved]:")
    distinct_sources = {}
    for r in search_results:
        url = r.get("url", "")
        title = r.get("title", "")
        source = r.get("source", "").capitalize()
        if url and not title.lower().startswith("search error"):
            distinct_sources[url] = (title, source)

    if distinct_sources:
        for url, (title, source) in distinct_sources.items():
            print(f"  - [{source}] {title}: {url}")
    else:
        print("  - No external source links accessible.")

    if errors:
        print("\n[Warnings / Non-fatal Errors]:")
        for err in set(errors):
            print(f"  * {err}")

    print("\n" + "=" * 60)


async def run_research(question: str):
    """Compile graph, execute state workflow, and display output.

    Args:
        question: User query string.
    """
    initial_state = create_initial_state(question)
    graph = create_research_graph()

    print(f"\n[Starting Research Workflow for]: '{question}'\n")

    try:
        # Asynchronously invoke the LangGraph workflow
        final_state = await graph.ainvoke(initial_state)
        display_results(final_state)
    except Exception as exc:
        print(f"\n[ERROR] Workflow failed with exception: {exc}", file=sys.stderr)
        logging.exception("Workflow execution failure")


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="LangGraph Multi-Agent Research Assistant"
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        help="Research question to process directly",
    )
    args = parser.parse_args()

    print_banner()

    question = args.question
    if not question:
        try:
            print("Enter your research question:")
            question = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

    if not question:
        print("No question provided. Exiting.")
        sys.exit(0)

    asyncio.run(run_research(question))


if __name__ == "__main__":
    main()
