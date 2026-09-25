"""Command Line Interface (CLI) entrypoint for LangGraph Research Assistant.

Demonstrates:
1. Persistence with thread IDs.
2. Human-in-the-Loop interrupt detection and review presentation.
3. Resuming execution using `Command(resume=...)` on the same thread ID.
4. Adaptive revision loops when rejected.

Usage:
    Interactive mode:
        python -m app.main

    Direct question mode:
        python -m app.main --question "What are the benefits of Model Context Protocol?"

    Specify persistent thread ID:
        python -m app.main --question "..." --thread-id "my-session-01"
"""

import argparse
import asyncio
import itertools
import logging
import os
import sys
import uuid

from pathlib import Path
from dotenv import load_dotenv
from langgraph.types import Command

# Environment loading: loads .env (best practice), with fallback to .env.example
_project_root = Path(__file__).resolve().parent.parent
_env_path = _project_root / ".env"
_example_path = _project_root / ".env.example"

if _env_path.exists():
    load_dotenv(dotenv_path=_env_path, override=True)
else:
    load_dotenv(override=True)

# Seamless fallback if user adds their key directly into .env.example
if _example_path.exists():
    provider = os.getenv("LLM_PROVIDER", "mistral").lower()
    key_var = "MISTRAL_API_KEY" if provider == "mistral" else ("GOOGLE_API_KEY" if provider == "gemini" else "GROQ_API_KEY")
    val = os.getenv(key_var, "")
    if not val or "your_" in val:
        load_dotenv(dotenv_path=_example_path, override=True)

from app.graph import create_research_graph
from app.state import create_initial_state

# Setup clean console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("research_assistant.cli")

# Silence verbose third-party loggers
for quiet_logger in ["httpx", "httpcore", "mcp", "google"]:
    logging.getLogger(quiet_logger).setLevel(logging.WARNING)


class StatusSpinner:
    """Animated terminal spinner for asynchronous workflow tasks."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "Processing..."):
        self.message = message
        self._running = False
        self._task = None

    async def _spin(self):
        spinner = itertools.cycle(self.FRAMES)
        while self._running:
            frame = next(spinner)
            sys.stdout.write(f"\r\033[K\033[1;36m{frame}\033[0m {self.message}")
            sys.stdout.flush()
            try:
                await asyncio.sleep(0.08)
            except asyncio.CancelledError:
                break

    def set_message(self, new_message: str):
        self.message = new_message

    async def __aenter__(self):
        if sys.stdout.isatty():
            self._running = True
            self._task = asyncio.create_task(self._spin())
        else:
            print(f"... {self.message}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._running:
            self._running = False
            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()


def print_banner():
    """Print welcoming CLI banner."""
    print("\n" + "=" * 60)
    print("        LANGGRAPH MULTI-AGENT RESEARCH ASSISTANT")
    print("=" * 60)
    provider = os.getenv("LLM_PROVIDER", "gemini")
    model = (
        os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        if provider == "gemini"
        else os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        if provider == "groq"
        else os.getenv("MISTRAL_MODEL", "open-mistral-7b")
    )
    print(f"Provider: {provider.upper()} | Model: {model} | Python: {sys.version.split()[0]}")
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
    human_approved = result.get("human_approved")
    human_feedback = result.get("human_feedback")

    print("\n" + "=" * 60)
    print("FINAL RESEARCH RESULTS")
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

    if human_approved is not None:
        status = "Approved" if human_approved else "Rejected"
        print(f"\n[Human Review Status]: {status}")
        if human_feedback:
            print(f"[Latest Human Feedback]: {human_feedback}")

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

    print("\n" + "=" * 60 + "\n")


async def run_research(question: str, thread_id: str | None = None):
    """Compile graph, execute state workflow, handle HITL interrupt, and display output.

    Args:
        question: User query string.
        thread_id: Optional identifier for checkpoint persistence and session resumption.
    """
    thread_id = thread_id or f"session-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = create_initial_state(question)
    graph = create_research_graph()

    logger.info("[GRAPH] Workflow started")
    logger.info(f"[GRAPH] Checkpointing enabled (thread_id: '{thread_id}')")
    print(f"\n[Starting Research Workflow for]: '{question}'")
    print(f"[Session Thread ID]: {thread_id}\n")

    try:
        # 1. Initial asynchronous invocation of graph with animating loader
        async with StatusSpinner("Conducting research and synthesizing findings..."):
            final_state = await graph.ainvoke(initial_state, config=config)

        # 2. Check for human-in-the-loop interrupts
        snapshot = await graph.aget_state(config)
        while snapshot.next:
            interrupts = []
            for task in snapshot.tasks:
                interrupts.extend(task.interrupts)

            if not interrupts:
                # No interrupts pending, run to next step
                async with StatusSpinner("Running next workflow step..."):
                    final_state = await graph.ainvoke(None, config=config)
                snapshot = await graph.aget_state(config)
                continue

            # Graph is paused at an interrupt point
            logger.info("[GRAPH] Workflow interrupted")
            payload = interrupts[0].value
            data = payload.get("data", {}) if isinstance(payload, dict) else {}
            message = (
                payload.get("message", "Please review the generated result.")
                if isinstance(payload, dict)
                else "Please review the generated result."
            )
            summary_to_review = data.get("summary") or data.get("result", "")

            print("\n" + "=" * 60)
            print("HUMAN APPROVAL REQUIRED")
            print("=" * 60)
            print(f"{message}\n")
            print("Generated Result:")
            print("-" * 60)
            print(summary_to_review.strip() if summary_to_review else "(No result content)")
            print("-" * 60 + "\n")

            # Solicit human decision outside the graph
            while True:
                try:
                    choice = input("Approve? (y/n): ").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    print("\nExecution aborted by user.")
                    return
                if choice in ["y", "yes", "n", "no"]:
                    break
                print("Please enter 'y' to approve or 'n' to reject.")

            approved = choice in ["y", "yes"]
            feedback = None

            if not approved:
                try:
                    feedback = input("Feedback: ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\nExecution aborted by user.")
                    return
                if not feedback:
                    feedback = "Please improve and refine this result."

            # Resume using Command(resume=...) on the same thread_id with animating loader
            logger.info(f"[GRAPH] Resuming workflow (thread_id: '{thread_id}')")
            spinner_msg = (
                f"Reworking research summary based on '{feedback}'..."
                if not approved
                else "Compiling final answer and verified citations..."
            )
            async with StatusSpinner(spinner_msg):
                final_state = await graph.ainvoke(
                    Command(
                        resume={
                            "approved": approved,
                            "feedback": feedback,
                        }
                    ),
                    config=config,
                )
            snapshot = await graph.aget_state(config)

        # 3. Present finalized workflow results
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
    parser.add_argument(
        "--thread-id",
        "-t",
        type=str,
        default=None,
        help="Custom thread ID for checkpoint persistence and session resumption",
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

    asyncio.run(run_research(question, thread_id=args.thread_id))


if __name__ == "__main__":
    main()
