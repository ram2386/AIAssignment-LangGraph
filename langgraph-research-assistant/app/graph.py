"""LangGraph workflow definition for Multi-Agent Research Assistant.

LangGraph Concepts Demonstrated:
---------------------------------
1. State: Typed dictionary (`ResearchState`) tracking workflow data and approval flags.
2. Nodes: Specialized functional agents (`planner`, `summarizer`, `final_answer`),
   a modular compiled subgraph (`retrieval_subgraph`), a human approval interrupt gate
   (`human_approval`), and an adaptive revision agent (`revision`).
3. Subgraphs: Encapsulation of parallel retrieval (`duckduckgo` and `wikipedia`)
   and retry verification into an independent, reusable `retrieval_subgraph`.
4. Persistence & Checkpointing: Configured with `InMemorySaver` so execution states
   are associated with configurable `thread_id`s, enabling interrupt and resume.
5. Human-in-the-Loop (HITL): The `human_approval` node uses `interrupt()` to pause
   execution, awaiting human approval or rejection with feedback.
6. Conditional Routing:
   - Approval routing (`route_approval`): routes to `final_answer` if approved,
     or `revision` if rejected.
7. Rejection Loop: Revision refines the synthesized summary and routes back to
   `human_approval` for re-evaluation.
"""

import logging
import os
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents import (
    duckduckgo_retrieval_agent,
    evaluate_retrieval,
    final_answer_agent,
    human_approval_node,
    planning_agent,
    revision_agent,
    summarization_agent,
    wikipedia_retrieval_agent,
)
from app.retrieval_subgraph import (
    build_retrieval_subgraph,
    create_retrieval_subgraph,
    route_retrieval_subgraph,
)
from app.state import ResearchState

logger = logging.getLogger("research_assistant.graph")


def route_retrieval(state: ResearchState) -> list[str] | str:
    """Evaluate retrieval quality and route conditionally.

    Retained for standalone retrieval evaluation and backwards compatibility:
    - If valid results are empty AND retry_count is within MAX_RETRIES:
      returns ['duckduckgo', 'wikipedia'] to initiate parallel retry.
    - Otherwise (results present OR retry budget exhausted):
      returns 'summarizer' to proceed towards final synthesis.
    """
    search_results = state.get("search_results", [])
    valid_results = [
        r
        for r in search_results
        if r.get("content")
        and not r.get("title", "").lower().startswith("search error")
    ]
    retry_count = state.get("retry_count", 0)
    max_retries = int(os.getenv("MAX_RETRIES", "2"))

    if not valid_results and 1 <= retry_count <= max_retries:
        return ["duckduckgo", "wikipedia"]

    return "summarizer"


def route_approval(state: ResearchState) -> str:
    """Evaluate human approval decision and route conditionally.

    LangGraph Concept: Dynamic Routing
    ----------------------------------
    Inspects state to determine next destination:
    - If `human_approved` is True:
      returns 'final_answer' to compile the user-facing response.
    - Otherwise (human rejected or revision requested):
      returns 'revision' to incorporate feedback and re-synthesize summary.
    """
    if state.get("human_approved"):
        return "final_answer"
    return "revision"


def build_research_graph() -> StateGraph:
    """Construct and configure the uncompiled LangGraph StateGraph.

    Architecture:
    START -> planner -> retrieval_subgraph -> summarizer -> human_approval
      -> (if approved) -> final_answer -> END
      -> (if rejected) -> revision -> human_approval

    Returns:
        StateGraph initialized with ResearchState and fully wired nodes and edges.
    """
    # 1. Initialize StateGraph with typed schema
    builder = StateGraph(ResearchState)

    # 2. Register Nodes
    builder.add_node("planner", planning_agent)
    builder.add_node("retrieval_subgraph", create_retrieval_subgraph())
    builder.add_node("summarizer", summarization_agent)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("revision", revision_agent)
    builder.add_node("final_answer", final_answer_agent)

    # 3. Add Edges: Start -> Planning -> Subgraph -> Summarizer -> Human Approval
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "retrieval_subgraph")
    builder.add_edge("retrieval_subgraph", "summarizer")
    builder.add_edge("summarizer", "human_approval")

    # 4. Add Conditional Routing for Human Approval Decision
    builder.add_conditional_edges(
        "human_approval",
        route_approval,
        {
            "final_answer": "final_answer",
            "revision": "revision",
        },
    )

    # 5. Connect Revision Loop back to Human Approval
    builder.add_edge("revision", "human_approval")

    # 6. Add Terminal Edge
    builder.add_edge("final_answer", END)

    return builder


def create_research_graph(checkpointer: Any | None = None) -> CompiledStateGraph:
    """Build and compile the multi-agent research workflow graph with persistence.

    Args:
        checkpointer: Optional LangGraph checkpoint saver. Defaults to InMemorySaver()
            to support state persistence, interrupt, and resumption across threads.
            Pass `False` to compile without a checkpointer.

    Returns:
        A compiled, executable LangGraph instance with checkpointing enabled.
    """
    if checkpointer is None:
        checkpointer = InMemorySaver()
    elif checkpointer is False:
        checkpointer = None

    builder = build_research_graph()
    return builder.compile(checkpointer=checkpointer)


# Module-level compiled instance for reuse
research_graph = create_research_graph()
