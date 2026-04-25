"""run_tests — executes test cases against agent-provided Python code.

Initial quality: HIGH (objective). Minimal diagnostic output on failure.

TOOL ARCHITECT SCOPE: This file may be rewritten by the Tool Architect.
Contract: function signature `run_tests(code: str, test_cases: str) -> str` must be preserved.
"""

import subprocess
import sys
import tempfile
import os


def run_tests(code: str, test_cases: str) -> str:
    """Run test cases against provided Python code.

    Args:
        code: The Python function/solution to test.
        test_cases: Newline-separated assert statements or a test script.

    Returns:
        A string reporting PASSED / FAILED with any error output.
    """
    combined = f"{code}\n\n{test_cases}\n"

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
        tmp.write(combined)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return "PASSED: All test cases passed."
        else:
            stderr = result.stderr.strip() or result.stdout.strip()
            return f"FAILED:\n{stderr}"
    except subprocess.TimeoutExpired:
        return "FAILED: Test execution timed out (>10s)."
    except Exception as exc:
        return f"FAILED: Unexpected error: {exc}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
