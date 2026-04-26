"""web_search — searches the web and returns real results (titles, URLs, snippets).

Uses DuckDuckGo — no API key required.

TOOL ARCHITECT SCOPE: This file may be rewritten by the Tool Architect.
Contract: function signature `web_search(query: str, max_results: int = 5) -> str` must be preserved.
"""

from duckduckgo_search import DDGS


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return titles, URLs, and snippets for the top results.

    Args:
        query: Search query string.
        max_results: Number of results to return (default 5).

    Returns:
        Formatted search results with title, URL, and snippet per result.
    """
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    f"### {r['title']}\n{r['href']}\n{r['body']}"
                )
        if not results:
            return f"No results found for: {query}"
        return "\n\n".join(results)
    except Exception as exc:
        return f"Search failed: {exc}"
