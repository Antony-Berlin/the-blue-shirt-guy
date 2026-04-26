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

from envs.gen_env.models import GenEnvAction, GenEnvToolAction
from envs.gen_env.server.gen_env_environment import GenesisEnvironment
from agent.tool_executor import ToolExecutor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-7B-Instruct")
SEED = int(os.getenv("GEN_ENV_SEED", "0")) or None
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
SUCCESS_THRESHOLD = float(os.getenv("SUCCESS_THRESHOLD", "0.5"))

BENCHMARK = "gen_env"

# ---------------------------------------------------------------------------
# Prompts — built dynamically from the tool files, no manual updates needed
# ---------------------------------------------------------------------------

def _get_tool_actions() -> set:
    from agent.tool_executor import _discover_tools
    return set(_discover_tools())


def _build_system_prompt() -> str:
    """Introspect every tool file to build the system prompt dynamically.

    Pulls the one-line description from the module docstring and generates
    an example JSON call from the function signature. Adding or removing a
    tool file is all that's needed — no edits here required.
    """
    import importlib.util
    import inspect
    from pathlib import Path

    tools_dir = Path(__file__).parent / "agent" / "tools"
    tool_files = sorted(p for p in tools_dir.glob("*.py") if p.stem != "__init__")

    tool_lines = []    # "  - tool_name(sig) — description"
    example_lines = [] # {"action": "tool_name", "param": "..."}

    # Example values for common parameter names
    _example_vals = {
        "query": "python Counter most frequent element",
        "url": "https://docs.python.org/3/library/collections.html",
        "code": "def foo(x): return x * 2",
        "test_cases": "assert foo(2) == 4",
        "traceback_text": "TypeError: unsupported operand...",
        "library": "collections",
        "symbol": "Counter",
        "max_results": 5,
        "max_chars": 3000,
    }

    for tool_path in tool_files:
        spec = importlib.util.spec_from_file_location(tool_path.stem, tool_path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            continue

        fn = getattr(mod, tool_path.stem, None)
        if fn is None or not callable(fn):
            continue

        # Description from module docstring first line after the dash
        mod_doc = (mod.__doc__ or "").strip().splitlines()[0]
        description = mod_doc.split("—", 1)[-1].strip() if "—" in mod_doc else mod_doc

        # Signature for the tool line
        sig = inspect.signature(fn)
        sig_str = ", ".join(
            f"{p.name}={p.default!r}" if p.default is not inspect.Parameter.empty else p.name
            for p in sig.parameters.values()
        )
        tool_lines.append(f"  - {tool_path.stem}({sig_str}) — {description}")

        # Example JSON call using known example values, skip defaults
        example = {"action": tool_path.stem}
        for name, param in sig.parameters.items():
            if param.default is inspect.Parameter.empty:
                example[name] = _example_vals.get(name, f'"<{name}>"')
            # skip optional params with defaults to keep example short
        example_lines.append(json.dumps(example))

    tool_block = "\n".join(tool_lines)
    example_block = "\n    ".join(example_lines)
    submit_example = json.dumps({"action": "submit", "code": "def foo(x): return x"})

    return textwrap.dedent(f"""
        You are a Python coding assistant solving programming challenges.
        You have access to these tools:
        {tool_block}

        Strategy:
        1. Use web_search or fetch_url to find real examples and documentation
        2. Use execute_code to experiment before writing the full solution
        3. Write your solution and run_tests to verify it passes
        4. Fix errors with explain_error and lint_code
        5. Submit your final code

        To call a tool respond ONLY with a JSON object where "action" is the tool name:
        {example_block}

        To submit your final answer:
        {submit_example}

        Respond ONLY with a single JSON object. No prose, no markdown fences.
    """).strip()


def _make_user_prompt(description: str, starter_code: str, tool_actions: set) -> str:
    tool_list = "\n".join(f"  - {t}" for t in sorted(tool_actions))
    return (
        f"TASK:\n{description}\n\n"
        f"STARTER CODE:\n```python\n{starter_code}\n```\n\n"
        f"Available tools:\n{tool_list}\n\n"
        "Use tools to research and test your solution, then submit."
    )


def _extract_json(text: str) -> Optional[dict]:
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    # Replace triple-quoted strings with single-quoted equivalents so json.loads works
    text = re.sub(r'"""(.*?)"""', lambda m: json.dumps(m.group(1)), text, flags=re.DOTALL)
    text = re.sub(r"'''(.*?)'''", lambda m: json.dumps(m.group(1)), text, flags=re.DOTALL)
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
    env=None,
) -> tuple:
    """Run the iterative tool-use loop; return (final_code, step_log).

    Each tool call is sent to env.step_tool() as a separate env step,
    graded immediately, and the grade fed back to the agent.
    step_log: [{step, action, reward, done, error}] — one entry per tool call.
    The final submit step is appended by main() after env.step().
    """
    tool_actions = _get_tool_actions()
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": _make_user_prompt(description, starter_code, tool_actions)},
    ]

    final_code = starter_code
    step_log: List[Dict[str, Any]] = []
    step_num = 0

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

        action = _extract_json(response_text)
        if action is None:
            # Don't append the bad response — add a corrective user turn instead
            messages.append({"role": "user", "content": "Respond ONLY with a single JSON object. No prose, no markdown. Example: {\"action\": \"submit\", \"code\": \"...\"}"})
            continue

        messages.append({"role": "assistant", "content": response_text})

        action_name = action.get("action", "")

        if action_name == "submit":
            final_code = action.get("code", starter_code)
            break

        elif action_name in tool_actions or action_name == "call_tool":
            if action_name == "call_tool":
                tool_name = action.get("tool", "")
                args = action.get("args", {})
            else:
                tool_name = action_name
                args = {k: v for k, v in action.items() if k != "action"}

            if tool_name not in tool_actions:
                messages.append({"role": "user", "content": f"Unknown tool '{tool_name}'. Available: {sorted(tool_actions)}"})
                continue

            # Execute tool locally
            tool_result = executor.call(tool_name, **args)
            log_entry = executor.get_log()[-1]
            step_num += 1

            # Send this single tool call to the env as its own step
            if env is not None:
                tool_action = GenEnvToolAction(
                    tool=tool_name,
                    args=log_entry["args"],
                    result=log_entry["result"],
                    error=log_entry.get("error"),
                )
                tool_obs = env.step_tool(tool_action)
                grade = tool_obs.reward
                # Sync the grade back into the executor's log entry
                log_entry["graded"] = True
                log_entry["grade"] = grade
                grade_hint = f"\n[Tool quality score: {grade:.2f}/1.0]"
                error_str = log_entry.get("error")
                step_log.append({
                    "step": step_num,
                    "action": f"{tool_name}({', '.join(f'{k}={str(v)[:40]!r}' for k, v in args.items())})",
                    "reward": grade,
                    "done": False,
                    "error": str(error_str)[:80] if error_str else None,
                })
            else:
                grade_hint = ""
                step_log.append({
                    "step": step_num,
                    "action": f"{tool_name}(...)",
                    "reward": 0.0,
                    "done": False,
                    "error": None,
                })

            messages.append({"role": "user", "content": f"Tool result:\n{tool_result}{grade_hint}"})

        else:
            messages.append({"role": "user", "content": f"Unknown action '{action_name}'. Use a tool action or 'submit'."})

    return final_code, step_log


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
        final_code, step_log = run_tool_loop(client, executor, description, starter_code, env=env)
        tool_log = executor.get_log()

        # Emit [STEP] for every tool call with its grade as the per-step reward
        for s in step_log:
            log_step(step=s["step"], action=s["action"], reward=s["reward"], done=False, error=s["error"])

        # ── 3. Submit to env ────────────────────────────────────────────────
        action = GenEnvAction(
            code=final_code,
            task_id=task_id,
            tool_usage_log=tool_log,
        )
        step_obs = env.step(action)

        reward = step_obs.reward or 0.0
        done = step_obs.done
        rewards.extend([s["reward"] for s in step_log])
        rewards.append(reward)
        submit_step = len(step_log) + 1
        steps_taken = submit_step

        tools_used = [e["tool"] for e in tool_log]
        action_summary = (
            f"submit(task={task_id}, tools={tools_used}, "
            f"tests={step_obs.tests_passed}/{step_obs.tests_total})"
        )
        log_step(step=submit_step, action=action_summary, reward=reward, done=done, error=None)

        print(f"[DEBUG] nl_feedback={step_obs.nl_feedback!r}", flush=True)
        print(f"[DEBUG] tool_weights={step_obs.tool_weights}", flush=True)
        print(f"[DEBUG] tool_grades={step_obs.tool_grades}", flush=True)

        score = min(max(reward, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
