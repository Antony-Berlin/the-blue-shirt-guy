"""Data models for the Genesis environment."""

from typing import Any, Dict, List, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class GenEnvAction(Action):
    """Agent submits a final code solution for evaluation."""

    code: str = Field(..., description="The Python code solution to evaluate")
    task_id: str = Field(..., description="ID of the task being solved")
    tool_usage_log: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Log of tool calls made during the episode [{tool, args, result}]",
    )


class GenEnvObservation(Observation):
    """Observation returned after each episode step."""

    task_id: str = Field(default="", description="Current task ID")
    task_description: str = Field(default="", description="Problem statement")
    starter_code: str = Field(default="", description="Starter code scaffold")
    difficulty: str = Field(default="easy", description="easy | medium | hard")
    tests_passed: int = Field(default=0, description="Number of tests passed")
    tests_total: int = Field(default=0, description="Total number of tests")
    nl_feedback: str = Field(
        default="",
        description="LLM-as-judge natural language critique of tool usage and reasoning",
    )
    tool_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="EMA performance weight per tool name",
    )
    tool_grades: List[float] = Field(
        default_factory=list,
        description="Per-tool-call grade [0.0, 1.0] in submission order",
    )


class GenEnvState(State):
    """Internal environment state for the current episode."""

    task_id: Optional[str] = Field(default=None, description="Active task ID")
    difficulty: str = Field(default="easy")
    tool_usage_log: List[Dict[str, Any]] = Field(default_factory=list)
    tool_weights: Dict[str, float] = Field(default_factory=dict)
    last_reward: Optional[float] = Field(default=None)
