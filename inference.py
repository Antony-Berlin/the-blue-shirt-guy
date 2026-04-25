"""
Genesis Environment Inference Script
===========================
MANDATORY env vars (loaded from .env):
    API_BASE_URL   LLM endpoint (OpenAI-compatible)
    MODEL_NAME     Model identifier
    HF_TOKEN       HuggingFace API key

STDOUT FORMAT (one line each):
    [START] task=<task_id> env=gen_env model=<model>
    [STEP]  step=<n> action=<summary> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,...>
"""

import json
import os
import re
import textwrap
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from envs.gen_env.models import GenEnvAction
from envs.gen_env.server.gen_env_environment import GenesisEnvironment
from agent.tool_executor import ToolExecutor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
SEED = int(os.getenv("GEN_ENV_SEED", "0")) or None
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
SUCCESS_THRESHOLD = float(os.getenv("SUCCESS_THRESHOLD", "0.5"))

BENCHMARK = "gen_env"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""
    You are a Python coding assistant solving programming challenges.
    You have access to these tools:
      - search_code_examples(query) — find similar code in a local corpus
      - run_tests(code, test_cases) — run assert statements against your code
      - lint_code(code) — check for syntax/style issues
      - fetch_docs(library, symbol="") — get Python library documentation
      - explain_error(traceback_text, code="") — diagnose an error traceback

    Strategy:
    1. Search for related examples first
    2. Write your solution
    3. Run tests to verify
    4. Fix errors using explain_error and lint_code
    5. Submit your final code

    When calling a tool, respond ONLY with JSON:
    {"action": "call_tool", "tool": "<name>", "args": {<kwargs>}}

    When you have a final answer, respond with:
    {"action": "submit", "code": "<your complete Python solution>"}
""").strip()


def _make_user_prompt(description: str, starter_code: str) -> str:
    return (
        f"TASK:\n{description}\n\n"
        f"STARTER CODE:\n```python\n{starter_code}\n```\n\n"
        "Use tools to research and test your solution, then submit."
    )


def _extract_json(text: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Logging helpers (mandatory stdout format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    # Sanitise action — strip newlines so it stays on one line
    action_safe = action.replace("\n", " ")[:120]
    print(
        f"[STEP] step={step} action={action_safe!r} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM tool-use loop
# ---------------------------------------------------------------------------

def run_tool_loop(
    client: OpenAI,
    executor: ToolExecutor,
    description: str,
    starter_code: str,
) -> str:
    """Run the iterative tool-use loop; return final code string."""
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _make_user_prompt(description, starter_code)},
    ]

    final_code = starter_code

    for _ in range(MAX_STEPS):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = (completion.choices[0].message.content or "").strip()
        except Exception as exc:
            print(f"[DEBUG] LLM call failed: {exc}", flush=True)
            break

        messages.append({"role": "assistant", "content": response_text})

        action = _extract_json(response_text)
        if action is None:
            break

        if action.get("action") == "submit":
            final_code = action.get("code", starter_code)
            break

        elif action.get("action") == "call_tool":
            tool_name = action.get("tool", "")
            args = action.get("args", {})
            tool_result = executor.call(tool_name, **args)
            messages.append({"role": "user", "content": f"Tool result:\n{tool_result}"})

    return final_code


# ---------------------------------------------------------------------------
# Main episode loop
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    executor = ToolExecutor()
    env = GenesisEnvironment()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    task_id = "unknown"

    try:
        # ── 1. Reset: get task ──────────────────────────────────────────────
        obs = env.reset(seed=SEED)

        task_id = obs.task_id
        description = obs.task_description
        starter_code = obs.starter_code
        difficulty = obs.difficulty

        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
        print(
            f"[DEBUG] difficulty={difficulty} task={task_id!r} "
            f"starter_lines={len(starter_code.splitlines())}",
            flush=True,
        )

        # ── 2. Tool-use loop (local) ────────────────────────────────────────
        executor.reset_log()
        final_code = run_tool_loop(client, executor, description, starter_code)
        tool_log = executor.get_log()

        # ── 3. Submit to env ────────────────────────────────────────────────
        action = GenEnvAction(
            code=final_code,
            task_id=task_id,
            tool_usage_log=tool_log,
        )
        step_obs = env.step(action)

        reward = step_obs.reward or 0.0
        done = step_obs.done
        rewards.append(reward)
        steps_taken = 1

        tools_used = [e["tool"] for e in tool_log]
        action_summary = (
            f"submit(task={task_id}, tools={tools_used}, "
            f"tests={step_obs.tests_passed}/{step_obs.tests_total})"
        )
        log_step(step=1, action=action_summary, reward=reward, done=done, error=None)

        print(f"[DEBUG] nl_feedback={step_obs.nl_feedback!r}", flush=True)
        print(f"[DEBUG] tool_weights={step_obs.tool_weights}", flush=True)

        score = min(max(reward, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
