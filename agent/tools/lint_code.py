"""lint_code — static analysis via pyflakes.

Initial quality: LOW. Too verbose, unfiltered, no severity ranking.

TOOL ARCHITECT SCOPE: This file may be rewritten by the Tool Architect.
Contract: function signature `lint_code(code: str) -> str` must be preserved.
"""

import subprocess
import sys
import tempfile
import os


def lint_code(code: str) -> str:
    """Run static analysis on Python code.

    Uses pyflakes. Returns raw output — verbose and unfiltered.

    Args:
        code: Python source code to analyze.

    Returns:
        Raw pyflakes output, or "No issues found." if clean.
    """
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pyflakes", tmp_path],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = (result.stdout + result.stderr).strip()
        output = output.replace(tmp_path + ":", "line ")
        if not output:
            return "No issues found."
        return output
    except subprocess.TimeoutExpired:
        return "Lint timed out."
    except Exception as exc:
        return f"Lint error: {exc}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
