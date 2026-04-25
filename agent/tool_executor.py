"""ToolExecutor — loads tool functions from agent/tools/ and runs them locally.

Builds the tool_usage_log that gets POSTed to the env server for reward attribution.
The Tool Architect may rewrite any file in agent/tools/ — the executor imports
them fresh on each episode so updated tools are picked up without restart.
"""

import importlib
import importlib.util
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

_TOOLS_DIR = Path(__file__).parent / "tools"

_TOOL_NAMES = [
    "search_code_examples",
    "run_tests",
    "lint_code",
    "fetch_docs",
    "explain_error",
]


def _load_tool(tool_name: str):
    """Dynamically import a tool function from agent/tools/<tool_name>.py.

    Uses importlib so updated files are picked up without restarting the process.
    """
    module_path = _TOOLS_DIR / f"{tool_name}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Tool file not found: {module_path}")

    spec = importlib.util.spec_from_file_location(f"agent.tools.{tool_name}", module_path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, tool_name)


class ToolExecutor:
    """Executes agent tools locally and records a tool_usage_log.

    Each call to ``call()`` appends an entry to the log:
        {"tool": str, "args": dict, "result": str, "error": str|None}

    The log is passed to the env server in ``SubmitCodeAction.tool_usage_log``
    so the env can attribute reward back to each tool's EMA weight.
    """

    def __init__(self) -> None:
        self._log: List[Dict[str, Any]] = []

    def call(self, tool_name: str, **kwargs: Any) -> str:
        """Execute a tool and record the call in the log.

        Args:
            tool_name: One of the 5 tool names.
            **kwargs: Arguments forwarded to the tool function.

        Returns:
            Tool output as a string.
        """
        result_str: str
        error: Optional[str] = None

        try:
            fn = _load_tool(tool_name)
            result = fn(**kwargs)
            result_str = str(result)
        except Exception as exc:
            result_str = f"[Tool error] {exc}"
            error = traceback.format_exc()

        self._log.append(
            {
                "tool": tool_name,
                "args": kwargs,
                "result": result_str,
                "error": error,
            }
        )
        return result_str

    def get_log(self) -> List[Dict[str, Any]]:
        """Return the accumulated tool usage log for this episode."""
        return list(self._log)

    def reset_log(self) -> None:
        """Clear the log at the start of a new episode."""
        self._log = []

    def available_tools(self) -> List[str]:
        return [t for t in _TOOL_NAMES if (_TOOLS_DIR / f"{t}.py").exists()]
