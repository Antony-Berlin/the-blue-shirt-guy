"""RubricJudge — LLM-as-judge scorer that returns (float_score, nl_feedback).

Falls back to a heuristic scorer when no API key is available so the
environment works without credentials (useful for smoke tests).
"""

import os
import re
import textwrap
from typing import Tuple


_SYSTEM_PROMPT = textwrap.dedent("""
    You are a rigorous code-quality judge. Given a programming task, the agent's
    code solution, and the tool usage log, produce:
    1. A score between 0.0 and 1.0 assessing reasoning quality and tool use
       (0 = no useful reasoning, 1 = exemplary step-by-step reasoning with effective tools).
    2. A short natural-language critique (1–3 sentences) identifying the most important
       weakness in tool usage or reasoning.

    Respond ONLY in this exact format:
    SCORE: <float>
    FEEDBACK: <one-to-three sentence critique>
""").strip()


def _heuristic_score(code: str, tool_log: list) -> Tuple[float, str]:
    """Simple heuristic fallback when no LLM is available."""
    score = 0.3
    feedback = "No LLM judge configured; heuristic score assigned."

    if tool_log:
        score += 0.1 * min(len(tool_log), 3)
        tools_used = {entry.get("tool") for entry in tool_log}
        if "run_tests" in tools_used:
            score += 0.1
        if "explain_error" in tools_used:
            score += 0.05
    if "def " in code and "return" in code:
        score += 0.1
    if "import" in code:
        score += 0.05

    return min(score, 1.0), feedback


def score_reasoning(
    task_description: str,
    code: str,
    tool_log: list,
) -> Tuple[float, str]:
    """Return (reasoning_quality_score 0-1, nl_feedback string).

    Uses Claude claude-haiku-4-5 via the Anthropic SDK when ANTHROPIC_API_KEY is set.
    Falls back to a heuristic scorer otherwise.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return _heuristic_score(code, tool_log)

    try:
        import anthropic

        tool_summary = "\n".join(
            f"- {e.get('tool', '?')}({e.get('args', {})})" for e in tool_log[:10]
        )
        user_msg = (
            f"TASK:\n{task_description}\n\n"
            f"AGENT CODE:\n```python\n{code}\n```\n\n"
            f"TOOL CALLS:\n{tool_summary or '(none)'}"
        )

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = response.content[0].text.strip()

        score_match = re.search(r"SCORE:\s*([0-9.]+)", text)
        feedback_match = re.search(r"FEEDBACK:\s*(.+)", text, re.DOTALL)

        score = float(score_match.group(1)) if score_match else 0.3
        feedback = feedback_match.group(1).strip() if feedback_match else text
        return min(max(score, 0.0), 1.0), feedback

    except Exception as exc:
        return _heuristic_score(code, tool_log)[0], f"Judge error: {exc}"
