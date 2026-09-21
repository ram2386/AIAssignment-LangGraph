# LangGraph Multi-Agent Research Assistant

A production-structured, beginner-friendly **Multi-Agent Research Assistant** built in Python using **LangGraph**, **LangChain**, and the **Model Context Protocol (MCP)** via `langchain-mcp-adapters`.

This project demonstrates the core foundational concepts of agentic graph orchestration: shared state, specialized agent nodes, fan-out parallel branches, fan-in synchronization barriers, dynamic conditional routing, automatic retry handling, and grounded final answer synthesis.

---

## 1. What is LangGraph?

**LangGraph** is an open-source orchestration framework designed for building stateful, multi-actor applications with LLMs. Unlike traditional sequential chains or monolithic agent loops:

* Applications are modeled explicitly as a **graph** composed of **Nodes** (agent tasks or functions) and **Edges** (transitions).
* A **Shared State** dictionary is passed from node to node. When a node executes, it returns partial updates that are applied to the state.
* **Parallel Execution** is a first-class citizen: branching a graph triggers concurrent async operations, automatically synchronizing at downstream join points.
* **Cycles and Conditional Edges** allow agents to inspect output, evaluate quality, and loop back (e.g. for retries or refinement) without infinite loops.

---

## 2. Architecture

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
                           Parallel branches (fan-out)
                             ┌───────┴───────┐
                             │               │
                             ▼               ▼
                    ┌────────────────┐ ┌────────────────┐
                    │ DuckDuckGo     │ │ Wikipedia      │
                    │ Retrieval      │ │ Retrieval      │
                    │ Agent (MCP)    │ │ Agent (MCP)    │
                    └───────┬────────┘ └───────┬────────┘
                            │                  │
                            └────────┬─────────┘
                                     │ Fan-in synchronization barrier
                                     ▼
                          ┌─────────────────────┐
                          │  Conditional Check  │
                          │                     │
                          │ Results available?  │
                          └──────────┬──────────┘
                                     │
                          ┌──────────┴──────────┐
                          │                     │
                        Retry                Continue
                 (retry_count < 2)       (results found or
                          │               max retries hit)
                          │                     │
                          │                     ▼
                          │           ┌──────────────────┐
                          │           │ Summarizer Agent │
                          │           └────────┬─────────┘
                          │                    │
                          │                    ▼
                          │           ┌──────────────────┐
                          │           │ Final Answer     │
                          │           │ Agent            │
                          │           └────────┬─────────┘
                          │                    │
                          │                    ▼
                          │                  END
                          │
                          └─> [Parallel Retry Retrieval]
```

---

## 3. Project Structure

```text
langgraph-research-assistant/
│
├── app/
│   ├── __init__.py           # Package marker
│   ├── state.py              # Typed shared state schema and merge reducers
│   ├── agents.py             # 5 specialized agents + evaluation node
│   ├── tools.py              # FastMCP servers & MultiServerMCPClient loader
│   ├── graph.py              # LangGraph definition, edges, conditional routing
│   └── main.py               # Interactive CLI interface
│
├── tests/
│   ├── __init__.py           # Test package marker
│   └── test_graph.py         # Unit tests for state, routing, retry, and graph
│
├── .env.example              # Environment variable template
├── .gitignore                # Git exclusions
├── requirements.txt          # Minimal, pinned dependencies
└── README.md                 # Complete documentation
```

### Module Responsibilities

* **`app/state.py`**: Defines `ResearchState(TypedDict)` containing `question`, `research_plan`, `search_results`, `duckduckgo_results`, `wikipedia_results`, `summary`, `final_answer`, `retry_count`, and `errors`. Implements `merge_search_results` reducer for conflict-free parallel updates.
* **`app/tools.py`**: Embeds two open-source `FastMCP` servers (`DuckDuckGo` and `Wikipedia`) that communicate via `stdio`. Connects agents to these tools dynamically using `MultiServerMCPClient`.
* **`app/agents.py`**: Implements specialized functional nodes:
  * **Planning Agent**: Deconstructs questions into structured research tasks.
  * **DuckDuckGo Retrieval Agent**: Runs live web queries using DuckDuckGo MCP tool.
  * **Wikipedia Retrieval Agent**: Queries encyclopedia articles using Wikipedia MCP tool.
  * **Evaluate Retrieval Node**: Tracks retrieval success and increments retry counters.
  * **Summarizer Agent**: Synthesizes and cross-references evidence from both sources.
  * **Final Answer Agent**: Formats user response with Answer, Key Findings, and Sources.
* **`app/graph.py`**: Assembles the `StateGraph`, registers nodes, links parallel edges, configures the conditional retry route, and compiles the workflow.
* **`app/main.py`**: Provides the interactive CLI and direct `--question` command-line flag.
* **`tests/test_graph.py`**: Validates state reducers, planner output, routing logic, retry bounds, and end-to-end graph compilation.

---

## 4. LangGraph Concepts Demonstrated

| Concept | Location in Code | How It Works |
|---|---|---|
| **State Management** | [`app/state.py`](app/state.py) | Defines `ResearchState(TypedDict)`. Uses `Annotated[list[dict], merge_search_results]` so parallel branches can update search results simultaneously without `InvalidUpdateError`. |
| **Nodes** | [`app/agents.py`](app/agents.py) | Each agent is an async function that receives `state: ResearchState` and returns a dictionary of state updates. |
| **Edges** | [`app/graph.py`](app/graph.py) | `builder.add_edge(START, "planner")` and `builder.add_edge("summarizer", "final_answer")` establish deterministic transitions. |
| **Parallel Execution (Fan-out)** | [`app/graph.py`](app/graph.py) | `planner` has outgoing edges to both `duckduckgo` and `wikipedia`. LangGraph executes both nodes concurrently. |
| **Synchronization Barrier (Fan-in)** | [`app/graph.py`](app/graph.py) | Both `duckduckgo` and `wikipedia` connect into `check_results`. LangGraph automatically waits for **both** to finish before executing the check node. |
| **Conditional Routing** | [`app/graph.py`](app/graph.py) | `add_conditional_edges("check_results", route_retrieval, ...)` evaluates whether search documents were found. |
| **Retry Handling** | [`app/graph.py`](app/graph.py), [`app/agents.py`](app/agents.py) | If results are empty and `retry_count < MAX_RETRIES` (default 2), it routes back to `['duckduckgo', 'wikipedia']` with broadened search terms. When max retries are reached, it gracefully moves forward to summarization. |
| **Final Answer Synthesis** | [`app/agents.py`](app/agents.py) | Grounded strictly on retrieved documents, formatting Answer, Key Findings, and clickable Source citations. |

---

## 5. Installation

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
   * **Windows (Command Prompt):**
     ```cmd
     python -m venv .venv
     .venv\Scripts\activate.bat
     ```
   * **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 6. Configuration

Create your local `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` and set your credentials:

```env
# Choose LLM Provider: gemini (default), groq, mistral, or mock
LLM_PROVIDER=gemini

# Google Gemini API Key (Get at: https://aistudio.google.com/)
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# Maximum retrieval retries
MAX_RETRIES=2
```

> **Note on Offline / Mock Mode**:
> If you do not have an API key or wish to run tests without internet LLM charges, leave `LLM_PROVIDER=mock` (or leave `GOOGLE_API_KEY` blank). The assistant will operate using deterministic heuristic synthesis.

---

## 7. Running the Application

### Interactive CLI Mode

```bash
python -m app.main
```

You will see:
```text
============================================================
        LANGGRAPH MULTI-AGENT RESEARCH ASSISTANT
============================================================
Provider: GEMINI | Python: 3.14.6
============================================================

Enter your research question:
> What are the advantages of Model Context Protocol for AI agents?
```

### Direct Question Mode (One-Shot)

```bash
python -m app.main --question "What are the benefits of Model Context Protocol for AI agents?"
```

---

## 8. Example Output

```text
============================================================
RESEARCH RESULTS
============================================================

[Question]:
What are the benefits of Model Context Protocol for AI agents?

[Research Plan]:
  1. Understand what Model Context Protocol (MCP) is
  2. Identify the main benefits and architectural advantages of MCP
  3. Identify limitations or security considerations of MCP
  4. Compare MCP with traditional ad-hoc tool integrations

[Research Summary]:
Model Context Protocol (MCP) is an open-source standard created by Anthropic to unify
how AI models connect to external tools, databases, and APIs.
- DuckDuckGo live sources highlight its role as the 'USB-C for AI', replacing fragmented
  custom API integrations with standardized stdio and HTTP JSON-RPC transports.
- Wikipedia and documentation articles emphasize security isolation, client-host-server
  architecture, and seamless tool portability across different AI agents.

[Final Answer]:
## Answer
The Model Context Protocol (MCP) provides a standardized, open specification that
transforms how AI agents interact with external data and tools. By creating a unified
client-server abstraction over standard transports (stdio and streamable HTTP), MCP replaces
bespoke API wrappers with reusable, portable tool servers.

## Key Findings
- **Standardized Integration**: Eliminates repetitive custom connectors for each LLM provider.
- **Enhanced Security**: Servers run in isolated subprocesses with explicit boundary controls.
- **Ecosystem Portability**: Tools built for MCP can be shared across Claude, LangGraph, Cursor, and custom agents.
- **Dynamic Discovery**: Agents query tools, resources, and prompts at runtime.

## Sources
- [Model Context Protocol Specification](https://modelcontextprotocol.io) (Duckduckgo)
- [Anthropic Introduces MCP](https://www.anthropic.com/news/model-context-protocol) (Duckduckgo)
- [Model Context Protocol](https://en.wikipedia.org/wiki/Model_Context_Protocol) (Wikipedia)

============================================================
```

---

## 9. Running Tests

Run the comprehensive test suite with `pytest`:

```bash
pytest tests/ -v
```

All 9 unit and integration tests execute offline in milliseconds:
```text
tests/test_graph.py::test_create_initial_state PASSED                    [ 11%]
tests/test_graph.py::test_merge_search_results_deduplication PASSED      [ 22%]
tests/test_graph.py::test_planning_agent_produces_non_empty_plan PASSED  [ 33%]
tests/test_graph.py::test_route_retrieval_with_results PASSED            [ 44%]
tests/test_graph.py::test_route_retrieval_without_results_triggers_retry PASSED [ 55%]
tests/test_graph.py::test_route_retrieval_retry_limit_exceeded PASSED    [ 66%]
tests/test_graph.py::test_evaluate_retrieval_increments_retry_count PASSED [ 77%]
tests/test_graph.py::test_graph_compiles_successfully PASSED             [ 88%]
tests/test_graph.py::test_full_graph_execution_mocked PASSED             [100%]
```

---

## 10. MCP Server Setup Details

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
   If you wish to point to external MCP servers (such as `uvx duckduckgo-mcp-server` or `npx @modelcontextprotocol/server-wikipedia`), you can simply specify them in `.env`:
   ```env
   DUCKDUCKGO_MCP_COMMAND=uvx
   DUCKDUCKGO_MCP_ARGS=duckduckgo-mcp-server
   WIKIPEDIA_MCP_COMMAND=npx
   WIKIPEDIA_MCP_ARGS=-y wikipedia-mcp
   ```

---

## 11. Known Limitations & Extensibility

* **Web Search Rate Limits**: DuckDuckGo's public endpoint may occasionally throttle high-frequency requests. The retry mechanism automatically broadens search terms when zero items are returned.
* **Wikipedia Disambiguation**: Some broad search queries may map to disambiguation pages; the agent automatically selects the first primary page candidate.
* **Extensibility**: You can easily add more specialized retrieval agents (e.g. ArXiv, GitHub, PubMed) by adding an MCP server in `tools.py`, creating a specialized agent in `agents.py`, and linking it as a parallel branch in `graph.py`.
