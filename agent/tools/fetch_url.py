"""fetch_url — fetches the content of a URL and returns the readable text.

Useful for reading documentation pages, Stack Overflow answers, GitHub files,
or any public URL relevant to the task.

TOOL ARCHITECT SCOPE: This file may be rewritten by the Tool Architect.
Contract: function signature `fetch_url(url: str, max_chars: int = 3000) -> str` must be preserved.
"""

import re
import urllib.request


def fetch_url(url: str, max_chars: int = 3000) -> str:
    """Fetch a URL and return its readable text content.

    Args:
        url: The URL to fetch (must be http/https).
        max_chars: Maximum characters to return (default 3000).

    Returns:
        Cleaned text content of the page, truncated to max_chars.
    """
    if not url.startswith(("http://", "https://")):
        return f"Invalid URL — must start with http:// or https://"

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; GenesisAgent/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="replace")

        # Strip HTML tags
        text = re.sub(r"<style[^>]*>.*?</style>", "", raw, flags=re.DOTALL)
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&[a-z]+;", " ", text)
        text = re.sub(r"\s{2,}", "\n", text).strip()

        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[truncated — {len(text)} chars total]"

        return text if text else "No readable content found."

    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code}: {exc.reason} — {url}"
    except urllib.error.URLError as exc:
        return f"Failed to fetch URL: {exc.reason}"
    except Exception as exc:
        return f"Fetch error: {exc}"
