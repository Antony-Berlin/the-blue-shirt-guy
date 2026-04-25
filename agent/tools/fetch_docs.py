"""fetch_docs — retrieves Python documentation for a library or symbol.

Initial quality: MEDIUM. Returns unfiltered pydoc dump, no relevance trimming.

TOOL ARCHITECT SCOPE: This file may be rewritten by the Tool Architect.
Contract: function signature `fetch_docs(library: str, symbol: str = "") -> str` must be preserved.
"""

import subprocess
import sys


def fetch_docs(library: str, symbol: str = "") -> str:
    """Retrieve documentation for a Python library or specific symbol.

    Args:
        library: The module name (e.g. "collections", "itertools").
        symbol: Optional attribute/function within the module (e.g. "Counter").

    Returns:
        Raw pydoc output — unfiltered, potentially very long (improvable).
    """
    target = f"{library}.{symbol}" if symbol else library

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pydoc", target],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout.strip()
        if not output:
            output = result.stderr.strip()
        if not output:
            return f"No documentation found for: {target}"
        return output[:2000]
    except subprocess.TimeoutExpired:
        return f"Docs fetch timed out for: {target}"
    except Exception as exc:
        return f"Error fetching docs for {target}: {exc}"
