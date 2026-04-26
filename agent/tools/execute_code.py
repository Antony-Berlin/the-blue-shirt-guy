"""execute_code — runs arbitrary Python code in a subprocess and returns stdout/stderr.

Different from run_tests — no test assertions required. Use this to experiment,
verify logic, print intermediate values, or prototype an approach.

TOOL ARCHITECT SCOPE: This file may be rewritten by the Tool Architect.
Contract: function signature `execute_code(code: str) -> str` must be preserved.
"""

import os
import subprocess
import sys
import tempfile


def execute_code(code: str) -> str:
    """Execute arbitrary Python code and return its output.

    Args:
        code: Python code to run. Use print() to see values.

    Returns:
        stdout + stderr from the execution, or an error message.
    """
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout.strip()
        stderr = result.stderr.strip()
        if output and stderr:
            return f"{output}\n[stderr]: {stderr}"
        if stderr:
            return f"[stderr]: {stderr}"
        if output:
            return output
        return "(no output)"
    except subprocess.TimeoutExpired:
        return "Execution timed out (>10s)."
    except Exception as exc:
        return f"Execution error: {exc}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
