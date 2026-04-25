"""ResearchAgent — LLM agent that solves coding tasks using local tools.

The agent:
  1. Calls POST /reset to receive a task from the env server
  2. Uses tools (search, lint, run_tests, fetch_docs, explain_error) locally
  3. Builds code iteratively using LLM reasoning + tool feedback
  4. Calls POST /step with final code + tool_usage_log
  5. Returns the reward + tool_weights for the training pipeline

The LLM is Qwen2.5-7B-Instruct by default (configurable). During GRPO training,
the Research Agent is frozen — only the Tool Architect's target tools change.
"""

import json
import re
import textwrap
from typing import Any, Dict, List, Optional

from agent.env_http_client import GenEnvHTTPClient
from agent.tool_executor import ToolExecutor

# ---------------------------------------------------------------------------
# Prompt templates
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

    When calling a tool, respond ONLY with JSON in this format:
    {"action": "call_tool", "tool": "<name>", "args": {<kwargs>}}

    When you have a final answer, respond with:
    {"action": "submit", "code": "<your complete Python solution>"}
""").strip()


def _make_user_prompt(task_description: str, starter_code: str) -> str:
    return (
        f"TASK:\n{task_description}\n\n"
        f"STARTER CODE:\n```python\n{starter_code}\n```\n\n"
        "Use tools to research and test your solution, then submit."
    )


def _extract_json(text: str) -> Optional[dict]:
    """Extract first JSON object from LLM output."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ResearchAgent:
    """Runs one episode: reset → tool loop → submit → return result.

    Can operate in two modes:
      - "llm": uses a real LLM (Qwen2.5-7B-Instruct via transformers/vllm)
      - "heuristic": rule-based agent for smoke testing without GPU
    """

    def __init__(
        self,
        env_url: str,
        mode: str = "heuristic",
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        max_steps: int = 8,
    ) -> None:
        self.client = GenEnvHTTPClient(env_url)
        self.executor = ToolExecutor()
        self.mode = mode
        self.model_name = model_name
        self.max_steps = max_steps
        self._llm = None  # lazy-loaded

    def _load_llm(self):
        if self._llm is not None:
            return self._llm
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self._llm = (tokenizer, model)
        return self._llm

    def _llm_step(self, messages: List[Dict]) -> str:
        tokenizer, model = self._load_llm()
        import torch

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    def _heuristic_episode(self, task: Dict) -> str:
        """Rule-based agent: search → write → run_tests → submit. No LLM needed."""
        description = task.get("task_description", "")
        starter_code = task.get("starter_code", "")
        task_id = task.get("task_id", "")

        # Step 1: search for examples
        query = " ".join(description.split()[:6])
        self.executor.call("search_code_examples", query=query)

        # Step 2: attempt a simple heuristic solution based on starter_code
        fn_match = re.search(r"def (\w+)\(", starter_code)
        fn_name = fn_match.group(1) if fn_match else "solution"

        # Very naive: wrap starter in pass; tests will likely fail — that's fine
        # (the point is to generate a realistic tool_usage_log)
        candidate_code = starter_code.replace("    pass", "    return None")

        # Step 3: run tests with dummy test cases
        self.executor.call(
            "run_tests",
            code=candidate_code,
            test_cases=f"# testing {fn_name}",
        )

        # Step 4: lint
        lint_result = self.executor.call("lint_code", code=candidate_code)
        if "No issues" not in lint_result:
            self.executor.call("explain_error", traceback_text=lint_result, code=candidate_code)

        return candidate_code

    def run_episode(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Run one complete episode.

        Returns dict with:
          reward, done, tests_passed, tests_total, nl_feedback, tool_weights
        """
        self.executor.reset_log()

        # 1. Get task from env
        task = self.client.reset(seed=seed)
        task_id = task.get("task_id", "")
        description = task.get("task_description", "")
        starter_code = task.get("starter_code", "")

        if self.mode == "llm":
            final_code = self._llm_episode(task_id, description, starter_code)
        else:
            final_code = self._heuristic_episode(task)

        # 2. Submit to env
        result = self.client.step(
            code=final_code,
            task_id=task_id,
            tool_usage_log=self.executor.get_log(),
        )
        return result

    def _llm_episode(self, task_id: str, description: str, starter_code: str) -> str:
        """Full LLM-driven tool-use loop."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _make_user_prompt(description, starter_code)},
        ]

        final_code = starter_code
        for _ in range(self.max_steps):
            response_text = self._llm_step(messages)
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
                tool_result = self.executor.call(tool_name, **args)
                messages.append({"role": "user", "content": f"Tool result:\n{tool_result}"})

        return final_code


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run one Genesis episode")
    parser.add_argument("--env-url", default="http://localhost:7860")
    parser.add_argument("--mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    agent = ResearchAgent(
        env_url=args.env_url,
        mode=args.mode,
        model_name=args.model,
    )
    result = agent.run_episode(seed=args.seed)

    print(f"Reward:        {result['reward']:.4f}")
    print(f"Tests:         {result['tests_passed']}/{result['tests_total']}")
    print(f"Feedback:      {result['nl_feedback']}")
    print(f"Tool weights:  {result['tool_weights']}")
