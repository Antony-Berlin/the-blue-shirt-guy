"""ToolRegistry — tracks per-tool EMA reward weights and emits KEEP/REVIEW/REPLACE flags."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class ToolFlag(str, Enum):
    KEEP = "KEEP"
    REVIEW = "REVIEW"
    REPLACE = "REPLACE"


_EMA_ALPHA = 0.3  # smoothing factor; higher = more reactive to recent episodes

TOOL_NAMES = [
    "search_code_examples",
    "run_tests",
    "lint_code",
    "fetch_docs",
    "explain_error",
]


@dataclass
class ToolRegistry:
    """Maintains an Exponential Moving Average of rewards attributed to each tool."""

    ema_weights: Dict[str, float] = field(
        default_factory=lambda: {name: 0.5 for name in TOOL_NAMES}
    )
    usage_counts: Dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in TOOL_NAMES}
    )

    def update(self, episode_reward: float, tools_used: List[str]) -> None:
        """Update EMA weights for tools that were used in this episode.

        Attribution is proportional to usage frequency within the trajectory.
        """
        if not tools_used:
            return

        counts: Dict[str, int] = {}
        for t in tools_used:
            counts[t] = counts.get(t, 0) + 1
        total = sum(counts.values())

        for tool_name, count in counts.items():
            if tool_name not in self.ema_weights:
                self.ema_weights[tool_name] = 0.5
                self.usage_counts[tool_name] = 0

            # Attribution: fraction of tool usage × episode reward
            attributed_reward = episode_reward * (count / total)
            self.ema_weights[tool_name] = (
                _EMA_ALPHA * attributed_reward
                + (1 - _EMA_ALPHA) * self.ema_weights[tool_name]
            )
            self.usage_counts[tool_name] += count

    def flag(self, tool_name: str) -> ToolFlag:
        weight = self.ema_weights.get(tool_name, 0.5)
        if weight > 0.7:
            return ToolFlag.KEEP
        if weight >= 0.4:
            return ToolFlag.REVIEW
        return ToolFlag.REPLACE

    def underperforming(self) -> List[str]:
        """Return tool names flagged REPLACE."""
        return [t for t in self.ema_weights if self.flag(t) == ToolFlag.REPLACE]

    def snapshot(self) -> Dict[str, float]:
        return dict(self.ema_weights)
