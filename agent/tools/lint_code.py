"""lint_code — static analysis via pyflakes.

TOOL ARCHITECT SCOPE: This file may be rewritten by the Tool Architect.
Contract: function signature `lint_code(code: str) -> str` must be preserved.
"""

import subprocess
import sys
import tempfile
import os


def lint_code(code: str) -> str:
    """Run static analysis on Python code using pyflakes.

    Filters out less severe issues and ranks remaining issues by severity.

    Args:
        code: Python source code to analyze.

    Returns:
        A formatted string containing ranked issues, or "No issues found." if clean.
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
        if not output:
            return "No issues found."

        lines = output.split('\n')
        issue_counts = {}
        for line in lines:
            parts = line.split(': ')
            if len(parts) == 2:
                message, count = parts[1].split(' ', 1)
                issue_counts[count] = issue_counts.get(count, []) + [message]

        sorted_issues = sorted(issue_counts.items(), key=lambda x: int(x[0]), reverse=True)
        formatted_output = "\n".join(f"{count}: {', '.join(issues)}" for count, issues in sorted_issues)
        return formatted_output
    except subprocess.TimeoutExpired:
        return "Lint timed out."
    except Exception as exc:
        return f"Lint error: {exc}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass