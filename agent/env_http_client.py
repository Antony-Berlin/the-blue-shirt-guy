"""Thin HTTP client for the Genesis environment server.

Calls POST /reset, POST /step, GET /state using plain requests.
No WebSocket, no openenv-core dependency required on the agent/training side.
"""

from typing import Any, Dict, Optional

import requests


class GenEnvHTTPClient:
    """HTTP client for the Genesis evaluation server.

    Usage::

        client = GenEnvHTTPClient("http://localhost:7860")
        task = client.reset(seed=42)
        print(task["task_description"])

        result = client.step(
            code="def missing_number(nums): ...",
            task_id=task["task_id"],
            tool_usage_log=[{"tool": "run_tests", "args": {...}, "result": "PASSED"}],
        )
        print(result["reward"])
        print(result["tool_weights"])
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers["Content-Type"] = "application/json"

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None) -> Dict[str, Any]:
        """POST /reset — returns task dict with task_id, task_description, starter_code, etc."""
        payload: Dict[str, Any] = {}
        if seed is not None:
            payload["seed"] = seed
        if episode_id is not None:
            payload["episode_id"] = episode_id

        resp = self._session.post(
            f"{self.base_url}/reset",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        # openenv wraps the observation under "observation" key
        return data.get("observation", data)

    def step(
        self,
        code: str,
        task_id: str,
        tool_usage_log: list,
    ) -> Dict[str, Any]:
        """POST /step — submit code + tool log, receive reward + feedback + tool weights."""
        payload = {
            "action": {
                "code": code,
                "task_id": task_id,
                "tool_usage_log": tool_usage_log,
            }
        }
        resp = self._session.post(
            f"{self.base_url}/step",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        obs = data.get("observation", data)
        return {
            "reward": data.get("reward"),
            "done": data.get("done", True),
            "task_id": obs.get("task_id", task_id),
            "tests_passed": obs.get("tests_passed", 0),
            "tests_total": obs.get("tests_total", 0),
            "nl_feedback": obs.get("nl_feedback", ""),
            "tool_weights": obs.get("tool_weights", {}),
            "metadata": obs.get("metadata", {}),
        }

    def get_state(self) -> Dict[str, Any]:
        """GET /state — returns current episode state + tool weight snapshot."""
        resp = self._session.get(
            f"{self.base_url}/state",
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def health(self) -> bool:
        """GET /health — returns True if server is up."""
        try:
            resp = self._session.get(f"{self.base_url}/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "GenEnvHTTPClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
