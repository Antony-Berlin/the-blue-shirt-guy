"""CodeForge environment client — connects to the running server via WebSocket."""

from typing import Dict, List, Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import CodeForgeObservation, SubmitCodeAction, CodeForgeState
except ImportError:
    from models import CodeForgeObservation, SubmitCodeAction, CodeForgeState


class CodeForgeEnv(EnvClient[SubmitCodeAction, CodeForgeObservation, CodeForgeState]):
    """Async client for the CodeForge environment server.

    Example::

        async with CodeForgeEnv(base_url="http://localhost:7860") as env:
            result = await env.reset(seed=42)
            print(result.observation.task_description)

            action = SubmitCodeAction(
                code="def most_frequent(lst): ...",
                task_id=result.observation.task_id,
                tool_usage_log=[],
            )
            result = await env.step(action)
            print(result.reward)  # composite float 0-1
    """

    def _step_payload(self, action: SubmitCodeAction) -> Dict[str, Any]:
        return {
            "code": action.code,
            "task_id": action.task_id,
            "tool_usage_log": action.tool_usage_log,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CodeForgeObservation]:
        obs_data = payload.get("observation", payload)
        observation = CodeForgeObservation(
            task_id=obs_data.get("task_id", ""),
            task_description=obs_data.get("task_description", ""),
            starter_code=obs_data.get("starter_code", ""),
            difficulty=obs_data.get("difficulty", "easy"),
            tests_passed=obs_data.get("tests_passed", 0),
            tests_total=obs_data.get("tests_total", 0),
            nl_feedback=obs_data.get("nl_feedback", ""),
            tool_weights=obs_data.get("tool_weights", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> CodeForgeState:
        return CodeForgeState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id"),
            difficulty=payload.get("difficulty", "easy"),
            tool_usage_log=payload.get("tool_usage_log", []),
            tool_weights=payload.get("tool_weights", {}),
            last_reward=payload.get("last_reward"),
        )
