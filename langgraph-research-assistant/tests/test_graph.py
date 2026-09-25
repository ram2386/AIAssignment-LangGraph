"""Tests for LangGraph Multi-Agent Research Assistant.

Covers:
1. State creation and reducer behavior (including HITL fields).
2. Planning agent execution.
3. Conditional routing (results found vs retry).
4. Retry count increment and MAX_RETRIES limit enforcement.
5. Graph compilation and node validation (parent graph and retrieval subgraph).
6. End-to-end execution of graph with mock components and HITL approval.
7. Persistence and checkpoints (InMemorySaver and thread ID resumption).
8. Interrupt handling and payload contract verification.
9. Approval routing to final answer.
10. Rejection routing to revision and feedback loop.
11. Multiple revision cycles followed by eventual approval.
12. Retrieval subgraph standalone execution and output validation.
13. Subgraph output propagation to parent graph.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.agents import (
    evaluate_retrieval,
    final_answer_agent,
    human_approval_node,
    planning_agent,
    revision_agent,
    summarization_agent,
)
from app.graph import (
    build_research_graph,
    create_research_graph,
    route_approval,
    route_retrieval,
)
from app.retrieval_subgraph import (
    build_retrieval_subgraph,
    create_retrieval_subgraph,
    route_retrieval_subgraph,
)
from app.state import ResearchState, create_initial_state, merge_search_results


# ---------------------------------------------------------------------------
# Test Environment Fixture
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def mock_llm_environment(monkeypatch):
    """Ensure all unit and integration tests run deterministically offline without network calls."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")


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
    assert state["human_approved"] is None
    assert state["human_feedback"] is None


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
        "human_approved": None,
        "human_feedback": None,
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
        "human_approved": None,
        "human_feedback": None,
    }

    destination = route_retrieval(state)
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
        "human_approved": None,
        "human_feedback": None,
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
    """Verify that the parent research graph and retrieval subgraph compile with expected nodes."""
    graph = create_research_graph()
    assert isinstance(graph, CompiledStateGraph)

    expected_parent_nodes = {
        "__start__",
        "planner",
        "retrieval_subgraph",
        "summarizer",
        "human_approval",
        "revision",
        "final_answer",
    }
    actual_parent_nodes = set(graph.nodes.keys())
    assert expected_parent_nodes.issubset(actual_parent_nodes)

    subgraph = create_retrieval_subgraph()
    assert isinstance(subgraph, CompiledStateGraph)
    expected_subgraph_nodes = {
        "__start__",
        "duckduckgo",
        "wikipedia",
        "check_results",
    }
    actual_subgraph_nodes = set(subgraph.nodes.keys())
    assert expected_subgraph_nodes.issubset(actual_subgraph_nodes)


# ---------------------------------------------------------------------------
# 5. Full End-to-End Graph Execution (Mocked)
# ---------------------------------------------------------------------------
def test_full_graph_execution_mocked():
    """Verify end-to-end execution of the graph from START to END with mocked agents and HITL approval."""
    sub_builder = build_retrieval_subgraph()

    # Define mock retrieval nodes
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

    sub_builder.nodes["duckduckgo"].runnable = mock_ddg
    sub_builder.nodes["wikipedia"].runnable = mock_wiki
    mocked_subgraph = sub_builder.compile()

    builder = build_research_graph()
    builder.nodes["retrieval_subgraph"].runnable = mocked_subgraph

    checkpointer = InMemorySaver()
    compiled_graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "test-mocked-e2e"}}

    initial_state = create_initial_state("What are the advantages of MCP?")

    # Initial invoke runs until interrupt at human_approval
    asyncio.run(compiled_graph.ainvoke(initial_state, config=config))

    # Verify execution paused at human_approval
    snapshot = asyncio.run(compiled_graph.aget_state(config))
    assert snapshot.next == ("human_approval",)

    # Resume with approval
    final_state = asyncio.run(
        compiled_graph.ainvoke(
            Command(resume={"approved": True, "feedback": None}),
            config=config,
        )
    )

    # Validate state progression through the entire workflow
    assert len(final_state["research_plan"]) > 0
    assert len(final_state["search_results"]) >= 2
    assert len(final_state["summary"]) > 0
    assert "Answer" in final_state["final_answer"]
    assert "Sources" in final_state["final_answer"]
    assert "https://example.com/mcp" in final_state["final_answer"]
    assert final_state["human_approved"] is True


# ---------------------------------------------------------------------------
# 6. Persistence & Checkpoint Tests
# ---------------------------------------------------------------------------
def test_persistence_and_checkpoints():
    """Verify checkpointer stores state snapshots and associates them with thread IDs."""
    checkpointer = InMemorySaver()
    graph = create_research_graph(checkpointer=checkpointer)
    thread_id = "persistence-thread-101"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = create_initial_state("Explain state persistence in LangGraph")
    asyncio.run(graph.ainvoke(initial_state, config=config))

    # Inspect checkpointed state
    snapshot = asyncio.run(graph.aget_state(config))
    assert snapshot is not None
    assert snapshot.values.get("question") == "Explain state persistence in LangGraph"
    assert len(snapshot.values.get("research_plan", [])) > 0
    assert snapshot.next == ("human_approval",)

    # Verify checkpoint history contains steps
    history = [s for s in graph.get_state_history(config)]
    assert len(history) >= 2


# ---------------------------------------------------------------------------
# 7. Interrupt Handling & Payload Contract Tests
# ---------------------------------------------------------------------------
def test_interrupt_pauses_execution_and_contains_payload():
    """Verify interrupt pauses execution and exposes structured approval payload."""
    graph = create_research_graph()
    thread_id = "interrupt-thread-202"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = create_initial_state("What is interrupt in LangGraph?")
    asyncio.run(graph.ainvoke(initial_state, config=config))

    snapshot = asyncio.run(graph.aget_state(config))
    assert snapshot.next == ("human_approval",)

    # Verify interrupt payload structure
    tasks = snapshot.tasks
    assert len(tasks) > 0
    interrupts = tasks[0].interrupts
    assert len(interrupts) > 0

    payload = interrupts[0].value
    assert isinstance(payload, dict)
    assert payload.get("type") == "approval"
    assert "message" in payload
    assert "data" in payload
    assert "summary" in payload["data"]
    assert "question" in payload["data"]

    # Verify downstream node (final_answer) has NOT run yet
    assert snapshot.values.get("final_answer") == ""


# ---------------------------------------------------------------------------
# 8. Approval Routing Tests
# ---------------------------------------------------------------------------
def test_approval_resumes_to_final_answer():
    """Verify resuming with approval proceeds to final_answer and completes the graph."""
    graph = create_research_graph()
    thread_id = "approval-thread-303"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = create_initial_state("Evaluate approval routing")
    asyncio.run(graph.ainvoke(initial_state, config=config))

    # Resume with approval
    final_state = asyncio.run(
        graph.ainvoke(
            Command(resume={"approved": True, "feedback": None}),
            config=config,
        )
    )

    assert final_state["human_approved"] is True
    assert final_state["human_feedback"] is None
    assert len(final_state["final_answer"]) > 0
    assert "Answer" in final_state["final_answer"]

    # Graph reached terminal END
    snapshot = asyncio.run(graph.aget_state(config))
    assert snapshot.next == ()


# ---------------------------------------------------------------------------
# 9. Rejection & Revision Routing Tests
# ---------------------------------------------------------------------------
def test_rejection_routes_to_revision_and_reapproval():
    """Verify resuming with rejection executes revision and returns to human_approval."""
    graph = create_research_graph()
    thread_id = "rejection-thread-404"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = create_initial_state("Review rejection and revision loop")
    asyncio.run(graph.ainvoke(initial_state, config=config))

    first_snapshot = asyncio.run(graph.aget_state(config))
    initial_summary = first_snapshot.values.get("summary")

    # Resume with rejection and specific feedback
    feedback_text = "Please emphasize latency and checkpoint serialization trade-offs."
    asyncio.run(
        graph.ainvoke(
            Command(resume={"approved": False, "feedback": feedback_text}),
            config=config,
        )
    )

    # Workflow must pause again at human_approval
    second_snapshot = asyncio.run(graph.aget_state(config))
    assert second_snapshot.next == ("human_approval",)

    # Check that feedback reached state and revised summary
    revised_summary = second_snapshot.values.get("summary")
    assert feedback_text in revised_summary or len(revised_summary) > len(initial_summary)

    # Verify second interrupt payload contains updated summary
    second_payload = second_snapshot.tasks[0].interrupts[0].value
    assert second_payload["type"] == "approval"
    assert second_payload["data"]["summary"] == revised_summary


# ---------------------------------------------------------------------------
# 10. Multiple Rejection Cycles Followed by Approval
# ---------------------------------------------------------------------------
def test_multiple_rejections_then_approval():
    """Verify the workflow supports repeated rejection and revision cycles before final approval."""
    graph = create_research_graph()
    thread_id = "multi-round-thread-505"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = create_initial_state("Test multi-round revision")
    asyncio.run(graph.ainvoke(initial_state, config=config))

    # Round 1 Rejection
    asyncio.run(
        graph.ainvoke(
            Command(resume={"approved": False, "feedback": "Revision 1: add benchmarks"}),
            config=config,
        )
    )
    snap1 = asyncio.run(graph.aget_state(config))
    assert snap1.next == ("human_approval",)

    # Round 2 Rejection
    asyncio.run(
        graph.ainvoke(
            Command(resume={"approved": False, "feedback": "Revision 2: add architecture diagram"}),
            config=config,
        )
    )
    snap2 = asyncio.run(graph.aget_state(config))
    assert snap2.next == ("human_approval",)

    # Round 3 Approval
    final_state = asyncio.run(
        graph.ainvoke(
            Command(resume={"approved": True, "feedback": None}),
            config=config,
        )
    )
    assert final_state["human_approved"] is True
    assert len(final_state["final_answer"]) > 0

    snap3 = asyncio.run(graph.aget_state(config))
    assert snap3.next == ()


# ---------------------------------------------------------------------------
# 11. Subgraph Execution Tests
# ---------------------------------------------------------------------------
def test_retrieval_subgraph_execution_standalone():
    """Verify that retrieval subgraph executes independently and returns valid search results."""
    subgraph = create_retrieval_subgraph()
    state = create_initial_state("What is FastMCP?")
    state["research_plan"] = ["FastMCP architecture", "FastMCP Python tools"]

    final_sub_state = asyncio.run(subgraph.ainvoke(state))

    assert "search_results" in final_sub_state
    assert isinstance(final_sub_state["search_results"], list)


def test_subgraph_output_reaches_parent_graph():
    """Verify that data produced by retrieval subgraph is propagated to parent graph state."""
    graph = create_research_graph()
    thread_id = "subgraph-prop-thread-606"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = create_initial_state("Test subgraph propagation")
    asyncio.run(graph.ainvoke(initial_state, config=config))

    snapshot = asyncio.run(graph.aget_state(config))
    # Subgraph results should be accessible in parent graph state at interrupt point
    assert "search_results" in snapshot.values
    assert "summary" in snapshot.values
    assert len(snapshot.values["summary"]) > 0


# ---------------------------------------------------------------------------
# 12. Feedback Alignment & Supplementary Retrieval Tests
# ---------------------------------------------------------------------------
def test_revision_incorporates_targeted_feedback():
    """Verify revision agent executes targeted supplementary retrieval and aligns summary to feedback."""
    from unittest.mock import patch
    from app.agents import perform_supplementary_retrieval, revision_agent

    mock_records = [
        {
            "source": "duckduckgo",
            "query": "Donald Trump Ukraine support war Russia",
            "title": "Trump Policy on Ukraine Aid and Russian Conflict",
            "url": "https://example.com/trump-ukraine",
            "content": "Analysis of Donald Trump's statements on military aid to Ukraine, diplomatic peace proposals, and relations with Russia.",
        }
    ]

    state: ResearchState = {
        "question": "Who is Donald Trump",
        "research_plan": ["Donald Trump career"],
        "duckduckgo_results": [],
        "wikipedia_results": [],
        "search_results": [
            {
                "source": "wikipedia",
                "title": "Donald Trump",
                "url": "https://en.wikipedia.org/wiki/Donald_Trump",
                "content": "Donald Trump was the 45th president of the United States.",
            }
        ],
        "summary": "Donald Trump was the 45th president of the United States.",
        "final_answer": "",
        "retry_count": 0,
        "errors": [],
        "human_approved": False,
        "human_feedback": "add the information about the Ukraine support for war against Russia",
    }

    with patch("app.agents.perform_supplementary_retrieval", AsyncMock(return_value=mock_records)):
        result = asyncio.run(revision_agent(state))

        assert "summary" in result
        assert "search_results" in result
        assert result["human_approved"] is None

        # Verify the new targeted records were produced
        assert len(result["search_results"]) == 1
        assert result["search_results"][0]["title"] == "Trump Policy on Ukraine Aid and Russian Conflict"

        # Verify the revised summary directly incorporates the feedback topic and evidence
        revised = result["summary"]
        assert "Ukraine" in revised
        assert "Russia" in revised
        assert "Trump Policy on Ukraine Aid" in revised or "military aid" in revised.lower()
