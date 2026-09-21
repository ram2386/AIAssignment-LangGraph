"""Model Context Protocol (MCP) server definitions and client integration.

LangGraph & MCP Concept: Dynamic Tool Discovery
------------------------------------------------
The Model Context Protocol (MCP) standardizes how LLM applications interact with
external data sources and tools. Rather than binding monolithic SDKs directly inside
agents, agents connect to lightweight MCP servers over standard transports (stdio or HTTP).

This module implements:
1. Two open-source FastMCP servers (DuckDuckGo web search and Wikipedia encyclopedia),
   executable as standalone stdio processes (`python -m app.tools --server <name>`).
2. A `MultiServerMCPClient` manager (via `langchain-mcp-adapters`) that connects to
   both MCP servers over stdio, discovers tools dynamically, and exposes LangChain-ready
   BaseTool instances to our specialized retrieval agents.
"""

import argparse
import logging
import os
import sys
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.server.fastmcp import FastMCP

# Setup logger for tools module
logger = logging.getLogger("research_assistant.tools")

# ---------------------------------------------------------------------------
# 1. DuckDuckGo FastMCP Server Definition
# ---------------------------------------------------------------------------
ddg_mcp = FastMCP("DuckDuckGoSearch")


@ddg_mcp.tool()
def search(query: str, max_results: int = 4) -> list[dict[str, str]]:
    """Search DuckDuckGo for live web information.

    Args:
        query: The search query string.
        max_results: Maximum number of search results to return (default: 4).

    Returns:
        A list of dictionaries with keys: title, url, content.
    """
    results: list[dict[str, str]] = []
    try:
        # Prefer modern ddgs package, fallback to duckduckgo_search if present
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS  # type: ignore

        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))

        for item in raw_results:
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "content": item.get("body", ""),
                }
            )
    except Exception as exc:
        logger.warning(f"DuckDuckGo search failed for '{query}': {exc}")
        results.append(
            {
                "title": f"Search Error for '{query}'",
                "url": "",
                "content": f"DuckDuckGo query encountered an error: {str(exc)}",
            }
        )
    return results


# ---------------------------------------------------------------------------
# 2. Wikipedia FastMCP Server Definition
# ---------------------------------------------------------------------------
wiki_mcp = FastMCP("WikipediaSearch")


@wiki_mcp.tool()
def search(query: str, max_results: int = 3) -> list[dict[str, str]]:
    """Search Wikipedia for encyclopedic articles and summaries.

    Args:
        query: The article or concept name to look up.
        max_results: Maximum number of Wikipedia pages to summarize (default: 3).

    Returns:
        A list of dictionaries with keys: title, url, content.
    """
    results: list[dict[str, str]] = []
    try:
        import wikipedia

        # Set compliant User-Agent to satisfy Wikimedia policy
        wikipedia.set_user_agent(
            "LangGraphResearchAssistant/1.0 (https://github.com/langgraph-research-assistant)"
        )

        search_titles = wikipedia.search(query, results=max_results)
        for title in search_titles:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                summary_text = (
                    page.summary[:1500] if hasattr(page, "summary") else ""
                )
                results.append(
                    {
                        "title": page.title,
                        "url": getattr(page, "url", ""),
                        "content": summary_text,
                    }
                )
            except wikipedia.DisambiguationError as dis_err:
                # If ambiguous, pick the first option
                if dis_err.options:
                    try:
                        sub_page = wikipedia.page(
                            dis_err.options[0], auto_suggest=False
                        )
                        results.append(
                            {
                                "title": sub_page.title,
                                "url": getattr(sub_page, "url", ""),
                                "content": sub_page.summary[:1500],
                            }
                        )
                    except Exception:
                        pass
            except wikipedia.PageError:
                continue
            except Exception as page_exc:
                logger.debug(f"Error fetching page '{title}': {page_exc}")
                continue
    except Exception as exc:
        logger.warning(f"Wikipedia search failed for '{query}': {exc}")
        results.append(
            {
                "title": f"Wikipedia Error for '{query}'",
                "url": "",
                "content": f"Wikipedia search encountered an error: {str(exc)}",
            }
        )
    return results


# ---------------------------------------------------------------------------
# 3. Client Configuration & Dynamic Tool Loading
# ---------------------------------------------------------------------------
def build_mcp_server_config() -> dict[str, dict[str, Any]]:
    """Build connection configuration dictionary for MultiServerMCPClient.

    Allows overriding server launch commands via environment variables:
    - DUCKDUCKGO_MCP_COMMAND and DUCKDUCKGO_MCP_ARGS
    - WIKIPEDIA_MCP_COMMAND and WIKIPEDIA_MCP_ARGS
    """
    python_bin = sys.executable

    # DuckDuckGo MCP server configuration
    ddg_cmd = os.getenv("DUCKDUCKGO_MCP_COMMAND", python_bin)
    ddg_args_str = os.getenv("DUCKDUCKGO_MCP_ARGS", "")
    if ddg_args_str:
        ddg_args = ddg_args_str.split()
    elif ddg_cmd == python_bin:
        ddg_args = ["-m", "app.tools", "--server", "duckduckgo"]
    else:
        ddg_args = []

    # Wikipedia MCP server configuration
    wiki_cmd = os.getenv("WIKIPEDIA_MCP_COMMAND", python_bin)
    wiki_args_str = os.getenv("WIKIPEDIA_MCP_ARGS", "")
    if wiki_args_str:
        wiki_args = wiki_args_str.split()
    elif wiki_cmd == python_bin:
        wiki_args = ["-m", "app.tools", "--server", "wikipedia"]
    else:
        wiki_args = []

    return {
        "duckduckgo": {
            "transport": "stdio",
            "command": ddg_cmd,
            "args": ddg_args,
        },
        "wikipedia": {
            "transport": "stdio",
            "command": wiki_cmd,
            "args": wiki_args,
        },
    }


def get_mcp_client() -> MultiServerMCPClient:
    """Create a configured MultiServerMCPClient instance.

    Returns:
        An initialized MultiServerMCPClient with duckduckgo and wikipedia servers.
    """
    config = build_mcp_server_config()
    return MultiServerMCPClient(config, tool_name_prefix=True)


async def load_research_tools() -> dict[str, BaseTool]:
    """Dynamically load and index MCP tools for each retrieval agent.

    Returns:
        A dictionary mapping server keys ('duckduckgo', 'wikipedia') to their
        respective LangChain BaseTool instances.
    """
    client = get_mcp_client()
    tools = await client.get_tools()

    tool_map: dict[str, BaseTool] = {}
    for tool in tools:
        if "duckduckgo" in tool.name.lower():
            tool_map["duckduckgo"] = tool
        elif "wikipedia" in tool.name.lower():
            tool_map["wikipedia"] = tool

    if "duckduckgo" not in tool_map or "wikipedia" not in tool_map:
        # Fallback by server name if prefix wasn't applied
        ddg_tools = await client.get_tools(server_name="duckduckgo")
        if ddg_tools:
            tool_map["duckduckgo"] = ddg_tools[0]
        wiki_tools = await client.get_tools(server_name="wikipedia")
        if wiki_tools:
            tool_map["wikipedia"] = wiki_tools[0]

    return tool_map


# ---------------------------------------------------------------------------
# 4. CLI Execution Entrypoint for Stdio Subprocesses
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LangGraph Research MCP Server")
    parser.add_argument(
        "--server",
        choices=["duckduckgo", "wikipedia"],
        required=True,
        help="Which MCP server to execute over stdio transport",
    )
    args = parser.parse_args()

    if args.server == "duckduckgo":
        ddg_mcp.run(transport="stdio")
    elif args.server == "wikipedia":
        wiki_mcp.run(transport="stdio")
