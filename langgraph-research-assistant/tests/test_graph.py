"""Tests for LangGraph Multi-Agent Research Assistant.

Covers:
1. State creation and reducer behavior.
2. Planning agent execution.
3. Conditional routing (results found vs retry).
4. Retry count increment and MAX_RETRIES limit enforcement.
5. Graph compilation and node validation.
6. End-to-end execution of graph with mock components.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from langgraph.graph.state import CompiledStateGraph

from app.agents import (
    evaluate_retrieval,
    final_answer_agent,
    planning_agent,
    summarization_agent,
)
from app.graph import build_research_graph, create_research_graph, route_retrieval
from app.state import ResearchState, create_initial_state, merge_search_results


# ---------------------------------------------------------------------------
# 1. State Tests
# ---------------------------------------------------------------------------
def test_create_initial_state():
    """Verify that initial state contains all required fields and correct defaults."""
    state = create_initial_state("What is Model Context Protocol?")

    assert state["question"] == "What is Model Context Protocol?"
    assert state["research_plan"] == []
    assert state["duckduckgo_results"] == []
    assert state["wikipedia_results"] == []
    assert state["search_results"] == []
    assert state["summary"] == ""
    assert state["final_answer"] == ""
    assert state["retry_count"] == 0
    assert state["errors"] == []


def test_merge_search_results_deduplication():
    """Verify reducer correctly merges items and eliminates exact duplicates."""
    existing = [
        {"source": "duckduckgo", "url": "https://example.com/1", "title": "Doc 1", "content": "A"}
    ]
    new_items = [
        # Duplicate of existing
        {"source": "duckduckgo", "url": "https://example.com/1", "title": "Doc 1", "content": "A"},
        # New distinct item
        {"source": "wikipedia", "url": "https://wikipedia.org/wiki/Doc2", "title": "Doc 2", "content": "B"},
    ]

    merged = merge_search_results(existing, new_items)

    assert len(merged) == 2
    assert merged[0]["title"] == "Doc 1"
    assert merged[1]["title"] == "Doc 2"


# ---------------------------------------------------------------------------
# 2. Planning Agent Tests
# ---------------------------------------------------------------------------
def test_planning_agent_produces_non_empty_plan():
    """Verify that the planning agent produces a non-empty list of research sub-queries."""
    state = create_initial_state("How does quantum computing work?")
    result = asyncio.run(planning_agent(state))

    assert "research_plan" in result
    plan = result["research_plan"]
    assert isinstance(plan, list)
    assert len(plan) >= 2
    for step in plan:
        assert isinstance(step, str)
        assert len(step.strip()) > 0


# ---------------------------------------------------------------------------
# 3. Routing & Retry Tests
# ---------------------------------------------------------------------------
def test_route_retrieval_with_results():
    """Verify that routing directs to 'summarizer' when search results are available."""
    state: ResearchState = {
        "question": "What is Python?",
        "research_plan": ["Python overview"],
        "duckduckgo_results": [],
        "wikipedia_results": [],
        "search_results": [
            {
                "source": "wikipedia",
                "title": "Python Language",
                "url": "https://en.wikipedia.org/wiki/Python",
                "content": "High level programming language",
            }
        ],
        "summary": "",
        "final_answer": "",
        "retry_count": 0,
        "errors": [],
    }

    destination = route_retrieval(state)
    assert destination == "summarizer"


def test_route_retrieval_without_results_triggers_retry():
    """Verify that routing directs back to parallel retrieval when results are missing."""
    state: ResearchState = {
        "question": "Obscure topic",
        "research_plan": ["Search query"],
        "duckduckgo_results": [],
        "wikipedia_results": [],
        "search_results": [],
        "summary": "",
        "final_answer": "",
        "retry_count": 1,  # 1st retry active
        "errors": [],
    }

    destination = route_retrieval(state)
    # Should route to both retrieval branches in parallel
    assert destination == ["duckduckgo", "wikipedia"]


def test_route_retrieval_retry_limit_exceeded():
    """Verify that routing proceeds to 'summarizer' when max retries are exhausted."""
    state: ResearchState = {
        "question": "Topic with no results",
        "research_plan": [],
        "duckduckgo_results": [],
        "wikipedia_results": [],
        "search_results": [],
        "summary": "",
        "final_answer": "",
        "retry_count": 3,  # Exceeded MAX_RETRIES (2)
        "errors": [],
    }

    destination = route_retrieval(state)
    assert destination == "summarizer"


def test_evaluate_retrieval_increments_retry_count():
    """Verify that evaluate_retrieval node increments retry_count when results are empty."""
    state = create_initial_state("Test Question")
    state["search_results"] = []
    state["retry_count"] = 0

    updates = asyncio.run(evaluate_retrieval(state))
    assert updates.get("retry_count") == 1


# ---------------------------------------------------------------------------
# 4. Graph Compilation Tests
# ---------------------------------------------------------------------------
def test_graph_compiles_successfully():
    """Verify that the research graph compiles and contains all expected nodes."""
    graph = create_research_graph()
    assert isinstance(graph, CompiledStateGraph)

    expected_nodes = {
        "__start__",
        "planner",
        "duckduckgo",
        "wikipedia",
        "check_results",
        "summarizer",
        "final_answer",
    }
    actual_nodes = set(graph.nodes.keys())
    assert expected_nodes.issubset(actual_nodes)


# ---------------------------------------------------------------------------
# 5. Full End-to-End Graph Execution (Mocked)
# ---------------------------------------------------------------------------
def test_full_graph_execution_mocked():
    """Verify end-to-end execution of the graph from START to END with mocked agents."""
    builder = build_research_graph()

    # Define simple mock nodes to test the exact wiring and state flow
    async def mock_ddg(state: ResearchState):
        mock_doc = {
            "source": "duckduckgo",
            "query": "MCP test",
            "title": "Model Context Protocol Intro",
            "url": "https://example.com/mcp",
            "content": "MCP connects LLMs with tools.",
        }
        return {"duckduckgo_results": [mock_doc], "search_results": [mock_doc]}

    async def mock_wiki(state: ResearchState):
        mock_doc = {
            "source": "wikipedia",
            "query": "MCP test",
            "title": "Context Protocol Entry",
            "url": "https://wikipedia.org/wiki/MCP",
            "content": "An open standard for AI contexts.",
        }
        return {"wikipedia_results": [mock_doc], "search_results": [mock_doc]}

    # Swap in mocked retrieval nodes for predictable unit testing
    builder.nodes["duckduckgo"].runnable = mock_ddg
    builder.nodes["wikipedia"].runnable = mock_wiki

    compiled_graph = builder.compile()

    initial_state = create_initial_state("What are the advantages of MCP?")
    final_state = asyncio.run(compiled_graph.ainvoke(initial_state))

    # Validate state progression through the entire workflow
    assert len(final_state["research_plan"]) > 0
    assert len(final_state["search_results"]) >= 2
    assert len(final_state["summary"]) > 0
    assert "Answer" in final_state["final_answer"]
    assert "Sources" in final_state["final_answer"]
    assert "https://example.com/mcp" in final_state["final_answer"]
