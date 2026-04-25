"""ToolArchitect — LLM-driven tool rewriter and creator.

Uses the same HuggingFace router as inference.py to rewrite underperforming
tools or create brand-new ones. Works by direct file editing — no OpenCode
needed.

Decision logic:
  - REPLACE  → full rewrite of the existing tool file
  - REVIEW   → targeted improvement patch to the existing tool file
  - NEW       → create a new tool file (architect decides when existing tools
                 are insufficient for the task domain)

The architect receives:
  - The current tool source code
  - Its EMA weight history and KEEP/REVIEW/REPLACE flag
  - Natural language feedback from the LLM judge
  - Example tool calls + results from recent episodes
  - The full list of existing tools (to avoid duplicates and enable new tool decisions)

Contract enforced on every generated file:
  - Valid Python syntax (ast.parse check before writing)
  - Function signature matches the declared contract (or defines a new contract for new tools)
  - Has a module-level docstring with "TOOL ARCHITECT SCOPE" marker
  - Exports exactly one callable with the same name as the file stem
"""

import ast
import os
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_TOOLS_DIR = Path(__file__).parent.parent / "agent" / "tools"
_API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
_API_BASE = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
_ARCHITECT_MODEL = os.getenv("ARCHITECT_MODEL", os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-7B-Instruct"))

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_REWRITE = textwrap.dedent("""
    You are an expert Python tool engineer. You will rewrite or improve a Python tool
    used by a coding agent to solve programming challenges.

    RULES:
    1. Output ONLY the complete Python file content — no prose, no markdown fences.
    2. Preserve the exact function signature (name and parameter names) of the main function.
    3. The file must define exactly one public function whose name matches the file stem.
    4. Include a module-level docstring with "TOOL ARCHITECT SCOPE" in it.
    5. Use only Python standard library — no third-party imports.
    6. Make the tool meaningfully better based on the feedback provided.
    7. The function must return a plain string.
""").strip()

_SYSTEM_CREATE = textwrap.dedent("""
    You are an expert Python tool engineer. You will create a brand-new Python tool
    for a coding agent that solves programming challenges.

    RULES:
    1. Output ONLY the complete Python file content — no prose, no markdown fences.
    2. The file stem (filename without .py) must be the function name.
    3. Define exactly one public function. It must accept keyword arguments only
       and return a plain string.
    4. Include a module-level docstring with "TOOL ARCHITECT SCOPE" in it.
    5. Use only Python standard library — no third-party imports.
    6. The tool must be genuinely useful for solving Python programming tasks.
    7. Choose a clear, snake_case name that describes what the tool does.
""").strip()

_SYSTEM_DECIDE = textwrap.dedent("""
    You are a tool portfolio manager for a coding agent. Given performance data,
    decide what action to take.

    Respond with a JSON object (no prose, no fences):
    {
      "action": "rewrite" | "improve" | "create_new" | "skip",
      "target_tool": "<tool_name or null for create_new>",
      "new_tool_name": "<snake_case name, only for create_new>",
      "new_tool_purpose": "<one sentence, only for create_new>",
      "reasoning": "<one sentence why>"
    }

    - "rewrite": completely replace a REPLACE-flagged tool
    - "improve": targeted improvement to a REVIEW-flagged tool
    - "create_new": add a new tool that fills a gap (use sparingly — only when existing tools
       are clearly insufficient for the recurring task patterns)
    - "skip": no action needed this cycle
""").strip()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _read_tool(tool_name: str) -> str:
    path = _TOOLS_DIR / f"{tool_name}.py"
    return path.read_text() if path.exists() else ""


def _list_tools() -> List[str]:
    return sorted(p.stem for p in _TOOLS_DIR.glob("*.py") if p.stem != "__init__")


def _validate_python(code: str) -> Tuple[bool, str]:
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, str(e)


def _extract_function_name(code: str) -> Optional[str]:
    """Return the name of the first top-level function defined in the code."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_"):
                    return node.name
    except SyntaxError:
        pass
    return None


def _extract_code_from_response(text: str) -> str:
    """Strip markdown fences if the model wraps output despite instructions."""
    # Remove ```python ... ``` or ``` ... ```
    text = re.sub(r"^```(?:python)?\s*\n?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text.strip())
    return text.strip()


def _build_perf_summary(
    tool_name: str,
    ema_weight: float,
    flag: str,
    nl_feedback: str,
    recent_calls: List[Dict],
) -> str:
    calls_summary = ""
    if recent_calls:
        samples = recent_calls[-3:]
        calls_summary = "\n".join(
            f"  call {i+1}: args={s.get('args',{})} → result={str(s.get('result',''))[:120]!r}"
            for i, s in enumerate(samples)
        )

    return textwrap.dedent(f"""
        TOOL: {tool_name}
        EMA WEIGHT: {ema_weight:.3f}  FLAG: {flag}
        RECENT JUDGE FEEDBACK: {nl_feedback or '(none)'}
        RECENT CALLS (last 3):
        {calls_summary or '  (no recent calls)'}
    """).strip()


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def _llm(system: str, user: str, max_tokens: int = 2048) -> str:
    client = OpenAI(base_url=_API_BASE, api_key=_API_KEY)
    response = client.chat.completions.create(
        model=_ARCHITECT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return (response.choices[0].message.content or "").strip()


def decide_action(
    tool_weights: Dict[str, float],
    tool_flags: Dict[str, str],
    nl_feedback: str,
    recent_tool_calls: Dict[str, List[Dict]],
) -> dict:
    """Ask the LLM to decide what improvement action to take this cycle."""
    existing = _list_tools()
    perf_lines = []
    for name, weight in tool_weights.items():
        flag = tool_flags.get(name, "REVIEW")
        calls = recent_tool_calls.get(name, [])
        perf_lines.append(_build_perf_summary(name, weight, flag, nl_feedback, calls))

    user_msg = (
        f"EXISTING TOOLS: {existing}\n\n"
        + "\n\n---\n\n".join(perf_lines)
        + f"\n\nOVERALL JUDGE FEEDBACK: {nl_feedback}"
    )

    import json
    raw = _llm(_SYSTEM_DECIDE, user_msg, max_tokens=256)
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
    try:
        return json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group())
    except Exception:
        return {"action": "skip", "reasoning": f"could not parse decision: {raw[:80]}"}


def rewrite_tool(
    tool_name: str,
    ema_weight: float,
    flag: str,
    nl_feedback: str,
    recent_calls: List[Dict],
    mode: str = "rewrite",
) -> Tuple[bool, str, str]:
    """Rewrite or improve an existing tool. Returns (success, new_code, message)."""
    current_source = _read_tool(tool_name)
    perf = _build_perf_summary(tool_name, ema_weight, flag, nl_feedback, recent_calls)

    verb = "completely rewrite" if mode == "rewrite" else "improve"

    user_msg = textwrap.dedent(f"""
        {verb.upper()} this tool to make it meaningfully better.

        PERFORMANCE DATA:
        {perf}

        CURRENT SOURCE:
        {current_source}

        Write the improved complete Python file now.
    """).strip()

    raw = _llm(_SYSTEM_REWRITE, user_msg)
    new_code = _extract_code_from_response(raw)

    ok, err = _validate_python(new_code)
    if not ok:
        return False, new_code, f"Syntax error in generated code: {err}"

    fn_name = _extract_function_name(new_code)
    if fn_name != tool_name:
        return False, new_code, f"Function name mismatch: expected '{tool_name}', got '{fn_name}'"

    return True, new_code, "ok"


def create_new_tool(
    new_tool_name: str,
    purpose: str,
    nl_feedback: str,
) -> Tuple[bool, str, str]:
    """Create a brand-new tool file. Returns (success, code, message)."""
    existing = _list_tools()
    user_msg = textwrap.dedent(f"""
        Create a new Python tool named `{new_tool_name}`.

        PURPOSE: {purpose}

        CONTEXT: This tool is for a coding agent solving Python programming tasks.
        Existing tools already available: {existing}
        Do NOT duplicate functionality already covered by existing tools.

        JUDGE FEEDBACK that motivated this new tool:
        {nl_feedback}

        Write the complete Python file for `{new_tool_name}.py` now.
    """).strip()

    raw = _llm(_SYSTEM_CREATE, user_msg)
    new_code = _extract_code_from_response(raw)

    ok, err = _validate_python(new_code)
    if not ok:
        return False, new_code, f"Syntax error in generated code: {err}"

    fn_name = _extract_function_name(new_code)
    if fn_name != new_tool_name:
        return False, new_code, f"Function name mismatch: expected '{new_tool_name}', got '{fn_name}'"

    return True, new_code, "ok"


# ---------------------------------------------------------------------------
# Main entry: apply one improvement cycle
# ---------------------------------------------------------------------------

def apply_improvement(
    tool_weights: Dict[str, float],
    tool_flags: Dict[str, str],
    nl_feedback: str,
    recent_tool_calls: Dict[str, List[Dict]],
    dry_run: bool = False,
) -> Dict:
    """Decide what to improve and write the new/updated tool file.

    Returns a result dict with keys:
        action, target_tool, file_written, success, message
    """
    decision = decide_action(tool_weights, tool_flags, nl_feedback, recent_tool_calls)
    print(f"[ARCHITECT] Decision: {decision}", flush=True)

    action = decision.get("action", "skip")
    result = {"action": action, "target_tool": None, "file_written": None, "success": False, "message": ""}

    if action == "skip":
        result["message"] = decision.get("reasoning", "no action needed")
        result["success"] = True
        return result

    if action in ("rewrite", "improve"):
        tool_name = decision.get("target_tool", "")
        if not tool_name or not (_TOOLS_DIR / f"{tool_name}.py").exists():
            result["message"] = f"Target tool '{tool_name}' not found"
            return result

        weight = tool_weights.get(tool_name, 0.5)
        flag = tool_flags.get(tool_name, "REVIEW")
        calls = recent_tool_calls.get(tool_name, [])

        ok, new_code, msg = rewrite_tool(tool_name, weight, flag, nl_feedback, calls, mode=action)
        result["target_tool"] = tool_name
        result["success"] = ok
        result["message"] = msg

        if ok:
            dest = _TOOLS_DIR / f"{tool_name}.py"
            if not dry_run:
                # Keep one backup
                backup = _TOOLS_DIR / f"{tool_name}.py.bak"
                backup.write_text(dest.read_text())
                dest.write_text(new_code)
                result["file_written"] = str(dest)
            else:
                result["file_written"] = f"(dry_run) {dest}"
            print(f"[ARCHITECT] {'(dry_run) ' if dry_run else ''}Wrote {dest}", flush=True)
        else:
            print(f"[ARCHITECT] Validation failed for '{tool_name}': {msg}", flush=True)

    elif action == "create_new":
        new_name = decision.get("new_tool_name", "")
        purpose = decision.get("new_tool_purpose", "")

        if not new_name or not re.match(r"^[a-z][a-z0-9_]+$", new_name):
            result["message"] = f"Invalid tool name '{new_name}'"
            return result

        dest = _TOOLS_DIR / f"{new_name}.py"
        if dest.exists():
            result["message"] = f"Tool '{new_name}' already exists"
            return result

        ok, new_code, msg = create_new_tool(new_name, purpose, nl_feedback)
        result["target_tool"] = new_name
        result["success"] = ok
        result["message"] = msg

        if ok:
            if not dry_run:
                dest.write_text(new_code)
                result["file_written"] = str(dest)
            else:
                result["file_written"] = f"(dry_run) {dest}"
            print(f"[ARCHITECT] {'(dry_run) ' if dry_run else ''}Created new tool {dest}", flush=True)
        else:
            print(f"[ARCHITECT] Validation failed for new tool '{new_name}': {msg}", flush=True)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    # Quick test: run one improvement cycle with synthetic data
    weights = {"search_code_examples": 0.35, "run_tests": 0.72, "lint_code": 0.31, "fetch_docs": 0.55, "explain_error": 0.28}
    flags = {k: ("REPLACE" if v < 0.4 else ("KEEP" if v > 0.7 else "REVIEW")) for k, v in weights.items()}
    feedback = (
        "The explain_error tool returned generic messages without referencing specific lines. "
        "The lint_code output was too verbose and the agent ignored it."
    )
    result = apply_improvement(weights, flags, feedback, {}, dry_run=False)
    print(json.dumps(result, indent=2))
