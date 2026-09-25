"""Information Retrieval Subgraph for LangGraph Multi-Agent Research Assistant.

LangGraph Concept: Subgraph-Based Modularization
------------------------------------------------
Encapsulates the parallel information retrieval and validation loop as an
independent, reusable StateGraph.

The subgraph manages:
1. Parallel retrieval agents: DuckDuckGo (live web) and Wikipedia (encyclopedic).
2. Synchronization barrier: Merging parallel branch writes into `check_results`.
3. Conditional retry evaluation: Routing back to retrieval branches if initial
   results are insufficient, or completing to END when results are satisfactory.

The parent graph integrates this module as a single node (`retrieval_subgraph`).
"""

import logging
import os
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents import (
    duckduckgo_retrieval_agent,
    evaluate_retrieval,
    wikipedia_retrieval_agent,
)
from app.state import ResearchState

logger = logging.getLogger("research_assistant.retrieval_subgraph")


def route_retrieval_subgraph(state: ResearchState) -> list[str] | str:
    """Evaluate retrieval completeness and decide whether to retry or exit subgraph.

    Returns:
        ['duckduckgo', 'wikipedia'] if retry is needed, or END if retrieval is complete.
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
        logger.info(f"[GRAPH] Subgraph initiating retry cycle ({retry_count}/{max_retries})")
        return ["duckduckgo", "wikipedia"]

    logger.info("[GRAPH] Subgraph completed")
    return END


def build_retrieval_subgraph() -> StateGraph:
    """Construct the uncompiled information retrieval StateGraph.

    Returns:
        StateGraph initialized with ResearchState schema and retrieval nodes/edges.
    """
    builder = StateGraph(ResearchState)

    # 1. Register retrieval nodes
    builder.add_node("duckduckgo", duckduckgo_retrieval_agent)
    builder.add_node("wikipedia", wikipedia_retrieval_agent)
    builder.add_node("check_results", evaluate_retrieval)

    # 2. Parallel fan-out from START into both retrieval agents
    builder.add_edge(START, "duckduckgo")
    builder.add_edge(START, "wikipedia")

    # 3. Fan-in synchronization barrier at check_results
    builder.add_edge("duckduckgo", "check_results")
    builder.add_edge("wikipedia", "check_results")

    # 4. Conditional retry routing or finish subgraph to END
    builder.add_conditional_edges(
        "check_results",
        route_retrieval_subgraph,
        {
            "duckduckgo": "duckduckgo",
            "wikipedia": "wikipedia",
            END: END,
        },
    )

    return builder


def create_retrieval_subgraph() -> CompiledStateGraph:
    """Build and compile the retrieval subgraph.

    Returns:
        A compiled StateGraph instance ready for parent graph embedding or standalone execution.
    """
    builder = build_retrieval_subgraph()
    return builder.compile()
