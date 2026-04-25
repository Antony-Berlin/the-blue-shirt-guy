"""GenesisEnvironment — pure evaluator deployed on HuggingFace Spaces.

The server holds benchmark tasks and hidden tests. It receives agent submissions
(code + tool usage log) via POST /step, evaluates them, and returns a dual reward.

Tools are NOT here. They live on the agent/training side in agent/tools/.
The Tool Architect rewrites those files locally — no server restart required.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action

try:
    from ..models import GenEnvObservation, GenEnvState, GenEnvAction
    from .rubric import score_reasoning
    from .tool_registry import ToolRegistry
    from .tool_graders import GraderRouter
except ImportError:
    from models import GenEnvObservation, GenEnvState, GenEnvAction
    from server.rubric import score_reasoning
    from server.tool_registry import ToolRegistry
    from server.tool_graders import GraderRouter

_BENCHMARK_PATH = Path(
    os.environ.get(
        "BENCHMARK_PATH",
        Path(__file__).parent.parent.parent.parent / "tasks" / "benchmark.json",
    )
)
MAX_STEPS_PER_EPISODE = int(os.environ.get("MAX_STEPS_PER_EPISODE", "10"))


def _load_benchmark() -> list:
    if _BENCHMARK_PATH.exists():
        with open(_BENCHMARK_PATH) as f:
            return json.load(f)
    return []


def _run_tests_against_code(code: str, tests: list[str]) -> tuple[int, int]:
    """Execute hidden test suite against submitted code. Returns (passed, total)."""
    if not tests:
        return 0, 0

    total = len(tests)
    passed = 0

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
        tmp.write(code + "\n\n")
        for test in tests:
            tmp.write(test + "\n")
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            passed = total
        else:
            # Try each test individually to get partial credit
            for test in tests:
                with tempfile.NamedTemporaryFile(
                    suffix=".py", mode="w", delete=False
                ) as t:
                    t.write(code + "\n\n" + test + "\n")
                    tpath = t.name
                res = subprocess.run(
                    [sys.executable, tpath],
                    capture_output=True,
                    timeout=5,
                )
                if res.returncode == 0:
                    passed += 1
                try:
                    os.unlink(tpath)
                except OSError:
                    pass
    except Exception:
        pass
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return passed, total


class GenesisEnvironment(Environment):
    """Pure evaluator environment — holds tasks, runs tests, computes rewards.

    The 5 coding tools live on the agent side (agent/tools/). The agent calls
    them locally, logs the calls, then submits code + log here for evaluation.

    POST /reset  → task description + starter code
    POST /step   → evaluate GenEnvAction, return dual reward + tool weights
    GET  /state  → current episode metadata + tool weight snapshot
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._benchmark: list = _load_benchmark()
        self._registry = ToolRegistry()
        self._grader = GraderRouter()
        self._current_task: Optional[dict] = None
        self._episode_id: Optional[str] = None
        self._step_count: int = 0
        self._tool_log: list = []

    # ------------------------------------------------------------------
    # Gym-style API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> GenEnvObservation:
        import random

        rng = random.Random(seed)
        tasks = self._benchmark or [_fallback_task()]
        self._current_task = rng.choice(tasks)
        self._episode_id = episode_id or str(uuid4())
        self._step_count = 0
        self._tool_log = []

        return GenEnvObservation(
            task_id=self._current_task["id"],
            task_description=self._current_task["description"],
            starter_code=self._current_task.get("starter_code", ""),
            difficulty=self._current_task.get("difficulty", "easy"),
            tests_passed=0,
            tests_total=len(self._current_task.get("tests", [])),
            reward=0.0,
            done=False,
            tool_weights=self._registry.snapshot(),
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> GenEnvObservation:
        if not isinstance(action, GenEnvAction):
            return GenEnvObservation(
                task_id=self._current_task["id"] if self._current_task else "",
                task_description="",
                starter_code="",
                difficulty="easy",
                reward=0.0,
                done=False,
                nl_feedback=f"Unknown action type: {type(action).__name__}. Expected GenEnvAction.",
                tool_weights=self._registry.snapshot(),
            )

        self._step_count += 1
        self._tool_log = action.tool_usage_log

        task = self._current_task or _fallback_task()
        tests = task.get("tests", [])

        passed, total = _run_tests_against_code(action.code, tests)
        pass_score = passed / total if total > 0 else 0.0

        # Per-tool-call grades (skips already-graded entries)
        tool_grades = self._grader.grade_log(action.tool_usage_log)
        tool_usage_score = sum(tool_grades) / len(tool_grades) if tool_grades else 0.0

        reasoning_score, nl_feedback = score_reasoning(
            task["description"], action.code, self._tool_log
        )

        reward = pass_score * 0.6 + tool_usage_score * 0.2 + reasoning_score * 0.2

        # Attribute reward to tools using per-entry grades
        tools_used = [entry.get("tool", "") for entry in self._tool_log if entry.get("tool")]
        entry_grades = {
            entry.get("tool", ""): entry.get("grade", 0.0)
            for entry in self._tool_log
            if entry.get("tool")
        }
        self._registry.update(reward, tools_used, entry_grades)

        return GenEnvObservation(
            task_id=task["id"],
            task_description=task["description"],
            starter_code=task.get("starter_code", ""),
            difficulty=task.get("difficulty", "easy"),
            tests_passed=passed,
            tests_total=total,
            reward=reward,
            done=True,
            nl_feedback=nl_feedback,
            tool_weights=self._registry.snapshot(),
            tool_grades=tool_grades,
            metadata={
                "pass_score": pass_score,
                "tool_usage_score": tool_usage_score,
                "reasoning_score": reasoning_score,
                "episode_id": self._episode_id,
            },
        )

    @property
    def state(self) -> GenEnvState:
        return GenEnvState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._current_task["id"] if self._current_task else None,
            difficulty=(
                self._current_task.get("difficulty", "easy")
                if self._current_task
                else "easy"
            ),
            tool_usage_log=self._tool_log,
            tool_weights=self._registry.snapshot(),
            last_reward=None,
        )


def _fallback_task() -> dict:
    return {
        "id": "fallback_001",
        "description": "Write a function most_frequent(lst) that returns the most frequently occurring element.",
        "starter_code": "def most_frequent(lst):\n    pass\n",
        "difficulty": "easy",
        "tests": [
            "assert most_frequent([1, 2, 2, 3]) == 2",
            "assert most_frequent(['a', 'b', 'a']) == 'a'",
        ],
    }
