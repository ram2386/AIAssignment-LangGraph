"""LangGraph workflow definition for Multi-Agent Research Assistant.

LangGraph Concepts Demonstrated:
---------------------------------
1. State: Typed dictionary (`ResearchState`) that tracks data across all steps.
2. Nodes: Specialized functional agents (`planner`, `duckduckgo`, `wikipedia`,
   `check_results`, `summarizer`, `final_answer`).
3. Edges: Directed links that define linear progression (`START -> planner`).
4. Parallel Branches (Fan-out): A single node (`planner`) points to multiple nodes
   (`duckduckgo` and `wikipedia`), triggering parallel asynchronous execution.
5. Synchronization Barrier (Fan-in): Multiple incoming edges into `check_results`
   cause LangGraph to wait until BOTH parallel branches have completed before proceeding.
6. Conditional Edges: Dynamic routing from `check_results` evaluating if useful
   results were obtained. If empty and within retry budget, routes back to both
   retrieval nodes in parallel. Otherwise transitions to `summarizer`.
7. End Node: Terminal state (`END`).
"""

import os
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents import (
    duckduckgo_retrieval_agent,
    evaluate_retrieval,
    final_answer_agent,
    planning_agent,
    summarization_agent,
    wikipedia_retrieval_agent,
)
from app.state import ResearchState


def route_retrieval(state: ResearchState) -> list[str] | str:
    """Evaluate retrieval quality and route conditionally.

    LangGraph Concept: Conditional Routing
    --------------------------------------
    Inspects state to determine next destination:
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

    # If no results found and we have retries remaining (note: evaluate_retrieval
    # increments retry_count when results are empty, so retry_count <= max_retries
    # indicates an active retry cycle)
    if not valid_results and 1 <= retry_count <= max_retries:
        return ["duckduckgo", "wikipedia"]

    return "summarizer"


def build_research_graph() -> StateGraph:
    """Construct and configure the uncompiled LangGraph StateGraph.

    Returns:
        StateGraph initialized with ResearchState and fully wired nodes and edges.
    """
    # 1. Initialize StateGraph with typed schema
    builder = StateGraph(ResearchState)

    # 2. Register Nodes (Specialized Agents)
    builder.add_node("planner", planning_agent)
    builder.add_node("duckduckgo", duckduckgo_retrieval_agent)
    builder.add_node("wikipedia", wikipedia_retrieval_agent)
    builder.add_node("check_results", evaluate_retrieval)
    builder.add_node("summarizer", summarization_agent)
    builder.add_node("final_answer", final_answer_agent)

    # 3. Add Edges: Start -> Planning
    builder.add_edge(START, "planner")

    # 4. Add Parallel Branches (Fan-out): Planner -> DuckDuckGo AND Wikipedia
    builder.add_edge("planner", "duckduckgo")
    builder.add_edge("planner", "wikipedia")

    # 5. Add Fan-in Synchronization Barrier: Both branches join at check_results
    builder.add_edge("duckduckgo", "check_results")
    builder.add_edge("wikipedia", "check_results")

    # 6. Add Conditional Edge: Retry loop or proceed to Summarizer
    builder.add_conditional_edges(
        "check_results",
        route_retrieval,
        {
            "duckduckgo": "duckduckgo",
            "wikipedia": "wikipedia",
            "summarizer": "summarizer",
        },
    )

    # 7. Add Linear Completion Edges: Summarizer -> Final Answer -> END
    builder.add_edge("summarizer", "final_answer")
    builder.add_edge("final_answer", END)

    return builder


def create_research_graph() -> CompiledStateGraph:
    """Build and compile the multi-agent research workflow graph.

    Returns:
        A compiled, executable LangGraph instance.
    """
    builder = build_research_graph()
    return builder.compile()


# Module-level compiled instance for reuse
research_graph = create_research_graph()
