"""RubricJudge — LLM-as-judge scorer that returns (float_score, nl_feedback).

Judge priority:
  1. HF router (OpenAI-compatible) — uses HF_TOKEN + RUBRIC_MODEL env vars
  2. Anthropic Claude — uses ANTHROPIC_API_KEY env var
  3. Heuristic fallback — no API key required

Set RUBRIC_MODEL to override the judge model (default: Qwen/Qwen2.5-Coder-7B-Instruct).
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

_RUBRIC_MODEL = os.environ.get("RUBRIC_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
_HF_API_BASE = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")


def _build_user_msg(task_description: str, code: str, tool_log: list) -> str:
    tool_summary = "\n".join(
        f"- {e.get('tool', '?')}({e.get('args', {})})" for e in tool_log[:10]
    )
    return (
        f"TASK:\n{task_description}\n\n"
        f"AGENT CODE:\n```python\n{code}\n```\n\n"
        f"TOOL CALLS:\n{tool_summary or '(none)'}"
    )


def _parse_response(text: str) -> Tuple[float, str]:
    score_match = re.search(r"SCORE:\s*([0-9.]+)", text)
    feedback_match = re.search(r"FEEDBACK:\s*(.+)", text, re.DOTALL)
    score = float(score_match.group(1)) if score_match else 0.3
    feedback = feedback_match.group(1).strip() if feedback_match else text
    return min(max(score, 0.0), 1.0), feedback


def _heuristic_score(code: str, tool_log: list) -> Tuple[float, str]:
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


def _judge_via_hf(task_description: str, code: str, tool_log: list) -> Tuple[float, str]:
    """OpenAI-compatible judge via HuggingFace router."""
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise ValueError("HF_TOKEN not set")

    from openai import OpenAI

    client = OpenAI(base_url=_HF_API_BASE, api_key=hf_token)
    response = client.chat.completions.create(
        model=_RUBRIC_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_msg(task_description, code, tool_log)},
        ],
        max_tokens=256,
        temperature=0.0,
    )
    text = (response.choices[0].message.content or "").strip()
    return _parse_response(text)


def _judge_via_anthropic(task_description: str, code: str, tool_log: list) -> Tuple[float, str]:
    """Anthropic Claude judge fallback."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_msg(task_description, code, tool_log)}],
    )
    text = response.content[0].text.strip()
    return _parse_response(text)


def score_reasoning(
    task_description: str,
    code: str,
    tool_log: list,
) -> Tuple[float, str]:
    """Return (reasoning_quality_score 0-1, nl_feedback string).

    Tries HF router first, then Anthropic, then heuristic fallback.
    """
    # 1. HF router (primary)
    if os.environ.get("HF_TOKEN"):
        try:
            return _judge_via_hf(task_description, code, tool_log)
        except Exception as exc:
            print(f"[DEBUG] HF judge failed: {exc}", flush=True)

    # 2. Anthropic (secondary)
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _judge_via_anthropic(task_description, code, tool_log)
        except Exception as exc:
            print(f"[DEBUG] Anthropic judge failed: {exc}", flush=True)

    # 3. Heuristic fallback
    return _heuristic_score(code, tool_log)
