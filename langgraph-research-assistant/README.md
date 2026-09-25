# LangGraph Multi-Agent Research Assistant

A production-structured, beginner-friendly **Multi-Agent Research Assistant** built in Python using **LangGraph**, **LangChain**, and the **Model Context Protocol (MCP)** via `langchain-mcp-adapters`.

This project demonstrates core foundational and advanced concepts of agentic graph orchestration:
- **Shared State Management** with conflict-free reducers (`typing.Annotated`).
- **Specialized Agent Nodes** for decomposition, retrieval, synthesis, and presentation.
- **Subgraph Modularization** encapsulating parallel retrieval and retry logic into an independent reusable graph.
- **State Persistence & Checkpointing** using `InMemorySaver` with configurable `thread_id` sessions.
- **Human-in-the-Loop (HITL)** decision gating via native LangGraph `interrupt()` and `Command(resume=...)`.
- **Dynamic Conditional Routing** for approval, rejection, and iterative revision loops.

---

## 1. Architecture

### Enhanced Modular Architecture

```text
                               ┌──────────────┐
                               │    START     │
                               └──────┬───────┘
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │   Planning Agent    │
                           │                     │
                           │  question → plan    │
                           └──────────┬──────────┘
                                      │
                                      ▼
                      ┌─────────────────────────────────┐
                      │    Retrieval Subgraph (MCP)     │
                      │  ┌───────────────────────────┐  │
                      │  │ START                     │  │
                      │  │   ├──→ DuckDuckGo Agent   │  │
                      │  │   └──→ Wikipedia Agent    │  │
                      │  │           │               │  │
                      │  │           ▼ (fan-in)      │  │
                      │  │      check_results        │  │
                      │  │           │               │  │
                      │  │      [retry loop]         │  │
                      │  │      ├── retry (retries<2)│  │
                      │  │      └── complete ───────→│  │
                      │  │                          END │
                      │  └───────────────────────────┘  │
                      └───────────────┬─────────────────┘
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │  Summarizer Agent   │
                           │                     │
                           │ synthesizes summary │
                           └──────────┬──────────┘
                                      │
                                      ▼
                           ┌─────────────────────┐  ◄───────────────────┐
                           │   Human Approval    │                      │
                           │    (HITL Gate)      │                      │
                           │                     │                      │
                           │     interrupt()     │                      │
                           └──────────┬──────────┘                      │
                                      │                                 │
                         [Approval Decision Route]                      │
                                      │                                 │
                       ┌──────────────┴──────────────┐                  │
          (Rejected: human_approved == False)        │ (Approved)       │
                       │                             │                  │
                       ▼                             ▼                  │
            ┌──────────────────┐           ┌──────────────────┐         │
            │  Revision Agent  │           │   Final Answer   │         │
            │                  │           │      Agent       │         │
            │ applies feedback │           │                  │         │
            └──────────┬───────┘           │ compiles answer  │         │
                       │                   └─────────┬────────┘         │
                       │                             │                  │
                       └─────────────────────────────┼──────────────────┘
                                                     │
                                                     ▼
                                                    END
```

---

## 2. Project Structure

```text
langgraph-research-assistant/
│
├── app/
│   ├── __init__.py              # Package marker
│   ├── state.py                 # Typed shared state schema and merge reducers
│   ├── agents.py                # Specialized agents, HITL approval node & revision agent
│   ├── retrieval_subgraph.py    # Subgraph encapsulating parallel retrieval & retry loop
│   ├── tools.py                 # FastMCP servers & MultiServerMCPClient loader
│   ├── graph.py                 # Parent LangGraph definition, checkpointer & routing
│   └── main.py                  # Interactive CLI with HITL prompt & thread persistence
│
├── tests/
│   ├── __init__.py              # Test package marker
│   └── test_graph.py            # Unit & integration tests (16 tests covering all features)
│
├── .env.example                 # Environment variable template
├── .gitignore                   # Git exclusions
├── requirements.txt             # Pinned project dependencies
└── README.md                    # Complete documentation
```

### Module Responsibilities

* **`app/state.py`**: Defines `ResearchState(TypedDict)` tracking `question`, `research_plan`, `search_results`, `duckduckgo_results`, `wikipedia_results`, `summary`, `final_answer`, `retry_count`, `errors`, `human_approved`, and `human_feedback`. Uses conflict-free reducers (`operator.add`, `merge_search_results`).
* **`app/retrieval_subgraph.py`**: Encapsulates `duckduckgo`, `wikipedia`, and `check_results` evaluation into an independent reusable `StateGraph`. Routes back for retry or exits to `END`.
* **`app/agents.py`**: Implements specialized functional nodes:
  * **Planning Agent**: Deconstructs questions into structured research tasks.
  * **DuckDuckGo Retrieval Agent**: Runs live web queries via DuckDuckGo FastMCP server.
  * **Wikipedia Retrieval Agent**: Queries encyclopedia articles via Wikipedia FastMCP server.
  * **Evaluate Retrieval Node**: Checks document presence and manages retry budget.
  * **Summarizer Agent**: Synthesizes and cross-references evidence across sources.
  * **Human Approval Node**: Invokes `interrupt()` with structured review data and captures feedback.
  * **Revision Agent**: Refines the research summary using reviewer feedback.
  * **Final Answer Agent**: Formats user response into Answer, Key Findings, and Sources.
* **`app/graph.py`**: Assembles the parent `StateGraph`, mounts the compiled `retrieval_subgraph`, connects the human approval gate and revision loop, and attaches `InMemorySaver` checkpointer.
* **`app/main.py`**: Command-line application managing execution threads, detecting interrupts, rendering human review prompts, and resuming via `Command(resume=...)`.
* **`tests/test_graph.py`**: 16 automated tests covering state reducers, planner, retry routing, subgraph execution, persistence, interrupts, approvals, and iterative revisions.

---

## 3. Core Concepts & Contracts

### 3.1 Persistence & Checkpoints

LangGraph checkpointing captures execution snapshots at each superstep. This project attaches `InMemorySaver` to the compiled graph:

```python
from langgraph.checkpoint.memory import InMemorySaver

builder = build_research_graph()
graph = builder.compile(checkpointer=InMemorySaver())
```

#### Thread ID Contract
Every execution is associated with a configurable `thread_id`:

```python
config = {"configurable": {"thread_id": "session-1234"}}
```

- **Initial Execution**: `await graph.ainvoke(initial_state, config=config)` runs until completion or until an `interrupt()` is encountered.
- **Resume Execution**: `await graph.ainvoke(Command(resume=resume_data), config=config)` resumes execution from the exact checkpoint on that thread.
- **Node Idempotency**: Resumed workflows do not re-execute previously completed upstream nodes (e.g. `planner`, `retrieval_subgraph`, `summarizer`).

---

### 3.2 Human-in-the-Loop (HITL)

Human review is positioned after the Summarizer Agent produces a research synthesis, before the Final Answer Agent compiles the public response.

#### Interrupt Contract (Graph → Application)
Inside `human_approval_node`:

```python
response = interrupt({
    "type": "approval",
    "message": "Please review the generated research summary.",
    "data": {
        "result": state["summary"],
        "summary": state["summary"],
        "question": state["question"],
    },
})
```

- When `interrupt()` is called, LangGraph suspends the thread, writes a checkpoint, and halts execution before any downstream nodes run.
- The payload is strictly serializable JSON-compatible data.
- **No blocking `input()` calls exist inside graph nodes**, preserving cross-platform reusability (CLI, FastAPI, Web UI).

#### Resume Contract (Application → Graph)
The application layer prompts the human and resumes the thread using `Command`:

```python
from langgraph.types import Command

command = Command(
    resume={
        "approved": True,       # True to accept, False to reject
        "feedback": None        # Optional revision instructions
    }
)
await graph.ainvoke(command, config=config)
```

The resume dictionary becomes the return value of `interrupt()` inside `human_approval_node`.

---

### 3.3 Approval & Rejection Routing

The parent graph routes dynamically based on `state["human_approved"]`:

```python
def route_approval(state: ResearchState) -> str:
    if state.get("human_approved"):
        return "final_answer"
    return "revision"
```

1. **Approval**:
   - `human_approved` is `True`.
   - Routes to `final_answer` agent.
   - Final response is formatted and graph terminates at `END`.
2. **Rejection**:
   - `human_approved` is `False`.
   - Routes to `revision` agent with `human_feedback`.
   - Summary is re-synthesized taking the reviewer's instructions into account.
   - Routes back to `human_approval_node` for re-evaluation.
   - Supports unlimited revision cycles until approved.

---

### 3.4 Subgraph Modularization

The parallel search and retry logic is encapsulated in `app/retrieval_subgraph.py`:
- **Why a Subgraph?** DuckDuckGo and Wikipedia retrieval, synchronization, and retry counting form a self-contained, cohesive subsystem.
- **Clean Parent Boundary**: The parent graph treats the entire search subsystem as a single modular node (`retrieval_subgraph`), improving maintainability and testability.
- **Encapsulated Retries**: Retries loop internally within the subgraph, avoiding cluttered edges in the top-level orchestration graph.

---

## 4. Installation & Setup

### Prerequisites

* Python 3.11 or higher
* Git

### Step-by-Step Setup

1. **Clone the repository and enter the directory:**
   ```bash
   git clone <repo_url>
   cd langgraph-research-assistant
   ```

2. **Create and activate a virtual environment:**
   * **macOS / Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   * **Windows:**
     ```cmd
     python -m venv .venv
     .venv\Scripts\activate.bat
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to configure your LLM provider (`gemini`, `groq`, `mistral`, or `mock`). If left blank or set to `mock`, the system operates in graceful offline mode.

---

## 5. Running the Application

### Interactive CLI Mode

```bash
python -m app.main
```

### Direct Question Mode with Custom Session Thread ID

```bash
python -m app.main --question "What are the benefits of Model Context Protocol?" --thread-id "mcp-session-01"
```

### Demonstration of Human-in-the-Loop Workflow

When executed, the system pauses at the human approval gate:

```text
[GRAPH] Workflow started
[GRAPH] Checkpointing enabled (thread_id: 'mcp-session-01')
[GRAPH] Subgraph started
[GRAPH] Subgraph completed
[Summarizer] Combining research and synthesizing summary...
[GRAPH] Waiting for human approval
[GRAPH] Workflow interrupted

============================================================
HUMAN APPROVAL REQUIRED
============================================================
Please review the generated research summary.

Generated Result:
------------------------------------------------------------
### Research Synthesis for: 'What are the benefits of Model Context Protocol?'
Synthesized evidence from DuckDuckGo web results and Wikipedia articles...
------------------------------------------------------------

Approve? (y/n): n
Feedback: Please emphasize the stdio transport security model.

[GRAPH] Resuming workflow (thread_id: 'mcp-session-01')
[GRAPH] Human rejected result
[GRAPH] Human feedback received: Please emphasize the stdio transport security model.
[GRAPH] Starting revision
[Revision] Summary revision completed.
[GRAPH] Waiting for human approval
[GRAPH] Workflow interrupted

============================================================
HUMAN APPROVAL REQUIRED
============================================================
...
**[Revision Applied - Human Feedback]:** Please emphasize the stdio transport security model.
------------------------------------------------------------

Approve? (y/n): y
[GRAPH] Resuming workflow (thread_id: 'mcp-session-01')
[GRAPH] Human approved
[Final Answer] Generating response with sources...

============================================================
FINAL RESEARCH RESULTS
============================================================
[Human Review Status]: Approved
[Latest Human Feedback]: Please emphasize the stdio transport security model.
...
```

---

## 6. Running Tests

Run the full test suite with `pytest`:

```bash
pytest tests/ -v
```

All 16 tests pass deterministically:

```text
tests/test_graph.py::test_create_initial_state PASSED                    [  6%]
tests/test_graph.py::test_merge_search_results_deduplication PASSED      [ 12%]
tests/test_graph.py::test_planning_agent_produces_non_empty_plan PASSED  [ 18%]
tests/test_graph.py::test_route_retrieval_with_results PASSED            [ 25%]
tests/test_graph.py::test_route_retrieval_without_results_triggers_retry PASSED [ 31%]
tests/test_graph.py::test_route_retrieval_retry_limit_exceeded PASSED    [ 37%]
tests/test_graph.py::test_evaluate_retrieval_increments_retry_count PASSED [ 43%]
tests/test_graph.py::test_graph_compiles_successfully PASSED             [ 50%]
tests/test_graph.py::test_full_graph_execution_mocked PASSED             [ 56%]
tests/test_graph.py::test_persistence_and_checkpoints PASSED             [ 62%]
tests/test_graph.py::test_interrupt_pauses_execution_and_contains_payload PASSED [ 68%]
tests/test_graph.py::test_approval_resumes_to_final_answer PASSED        [ 75%]
tests/test_graph.py::test_rejection_routes_to_revision_and_reapproval PASSED [ 81%]
tests/test_graph.py::test_multiple_rejections_then_approval PASSED       [ 87%]
tests/test_graph.py::test_retrieval_subgraph_execution_standalone PASSED [ 93%]
tests/test_graph.py::test_subgraph_output_reaches_parent_graph PASSED    [100%]

============================= 16 passed in 12.65s ==============================
```

---

## 7. MCP Server Setup Details

This project uses the official Python MCP SDK with `langchain-mcp-adapters`:

1. **Built-in FastMCP Servers**:
   The file [`app/tools.py`](app/tools.py) defines two FastMCP servers:
   - `DuckDuckGoSearch`: Uses `ddgs` to execute real web queries.
   - `WikipediaSearch`: Uses `wikipedia` with a Wikimedia-compliant User-Agent header.
2. **Subprocess stdio Transport**:
   When `load_research_tools()` is called, `MultiServerMCPClient` spawns:
   - `python -m app.tools --server duckduckgo`
   - `python -m app.tools --server wikipedia`
   These processes communicate using standard JSON-RPC over stdin/stdout.
3. **Custom External MCP Servers**:
   You can easily configure external MCP servers in `.env`:
   ```env
   DUCKDUCKGO_MCP_COMMAND=uvx
   DUCKDUCKGO_MCP_ARGS=duckduckgo-mcp-server
   WIKIPEDIA_MCP_COMMAND=npx
   WIKIPEDIA_MCP_ARGS=-y wikipedia-mcp
   ```
