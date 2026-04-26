"""Genesis environment client — connects to the running server via WebSocket.

Use GenesisEnvClient (the sync wrapper at the bottom) in all client-side code.
It mirrors the GenesisEnvironment interface exactly so no call-site changes are needed.

Server URL is read from ENV_SERVER_URL (default http://localhost:7860).
For the deployed HuggingFace Space set:
    ENV_SERVER_URL=https://berlin1906-genesis-env.hf.space
"""

import os
from typing import Any, Dict, Optional

from openenv.core import EnvClient, SyncEnvClient
from openenv.core.client_types import StepResult

try:
    from .models import GenEnvObservation, GenEnvAction, GenEnvState, GenEnvToolAction
except ImportError:
    from models import GenEnvObservation, GenEnvAction, GenEnvState, GenEnvToolAction

_DEFAULT_URL = "https://berlin1906-genesis-env.hf.space"


class GenEnvClient(EnvClient[GenEnvAction, GenEnvObservation, GenEnvState]):
    """Async client for the Genesis environment server.

    Example::

        async with GenEnvClient(base_url="http://localhost:7860") as env:
            result = await env.reset(seed=42)
            print(result.observation.task_description)

            action = GenEnvAction(
                code="def most_frequent(lst): ...",
                task_id=result.observation.task_id,
                tool_usage_log=[],
            )
            result = await env.step(action)
            print(result.reward)
    """

    def _step_payload(self, action: GenEnvAction) -> Dict[str, Any]:
        return {
            "code": action.code,
            "task_id": action.task_id,
            "tool_usage_log": action.tool_usage_log,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[GenEnvObservation]:
        obs_data = payload.get("observation", payload)
        observation = GenEnvObservation(
            task_id=obs_data.get("task_id", ""),
            task_description=obs_data.get("task_description", ""),
            starter_code=obs_data.get("starter_code", ""),
            difficulty=obs_data.get("difficulty", "easy"),
            tests_passed=obs_data.get("tests_passed", 0),
            tests_total=obs_data.get("tests_total", 0),
            nl_feedback=obs_data.get("nl_feedback", ""),
            tool_weights=obs_data.get("tool_weights", {}),
            tool_grades=obs_data.get("tool_grades", []),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> GenEnvState:
        return GenEnvState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id"),
            difficulty=payload.get("difficulty", "easy"),
            tool_usage_log=payload.get("tool_usage_log", []),
            tool_weights=payload.get("tool_weights", {}),
            last_reward=payload.get("last_reward"),
        )


class GenesisEnvClient:
    """Synchronous drop-in replacement for GenesisEnvironment.

    Wraps GenEnvClient (async) via SyncEnvClient so all existing call sites
    work unchanged — no async/await needed.

    Usage::

        env = GenesisEnvClient()          # reads ENV_SERVER_URL
        obs = env.reset(seed=42)          # returns GenEnvObservation
        step_obs = env.step(action)       # returns GenEnvObservation
        tool_obs = env.step_tool(ta)      # returns GenEnvObservation
    """

    def __init__(self, base_url: Optional[str] = None) -> None:
        url = base_url or os.getenv("ENV_SERVER_URL", _DEFAULT_URL)
        self._sync: SyncEnvClient = GenEnvClient(base_url=url).sync()
        self._sync.connect()

    def reset(self, seed: Optional[int] = None, **kwargs: Any) -> GenEnvObservation:
        result = self._sync.reset(seed=seed, **kwargs)
        return result.observation

    def step(self, action: GenEnvAction, **kwargs: Any) -> GenEnvObservation:
        result = self._sync.step(action, **kwargs)
        obs = result.observation
        # surface reward and done onto the observation so call sites work unchanged
        if result.reward is not None:
            obs.reward = result.reward
        obs.done = result.done
        return obs

    def step_tool(self, action: GenEnvToolAction) -> GenEnvObservation:
        # step_tool is a custom mid-episode endpoint — sent as a regular step
        # with the tool action serialised; the server distinguishes by action type
        result = self._sync.step(action)  # type: ignore[arg-type]
        obs = result.observation
        if result.reward is not None:
            obs.reward = result.reward
        obs.done = result.done
        return obs

    @property
    def state(self) -> GenEnvState:
        return self._sync.state()

    def close(self) -> None:
        self._sync.close()

    def __enter__(self) -> "GenesisEnvClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
