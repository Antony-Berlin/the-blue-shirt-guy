"""ToolRegistry — tracks per-tool EMA reward weights and emits KEEP/REVIEW/REPLACE flags."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ToolFlag(str, Enum):
    KEEP = "KEEP"
    REVIEW = "REVIEW"
    REPLACE = "REPLACE"


_EMA_ALPHA = 0.3  # smoothing factor; higher = more reactive to recent episodes


@dataclass
class ToolRegistry:
    """Maintains an Exponential Moving Average of rewards attributed to each tool."""

    ema_weights: Dict[str, float] = field(default_factory=dict)
    usage_counts: Dict[str, int] = field(default_factory=dict)

    def update(
        self,
        episode_reward: float,
        tools_used: List[str],
        entry_grades: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update EMA weights for tools used in this episode.

        When entry_grades is provided, each tool's attributed reward is its
        per-entry grade (last grade seen for that tool name). This gives finer
        signal than distributing the aggregate episode reward by usage count.

        Falls back to usage-proportional attribution when entry_grades is absent.
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

            if entry_grades and tool_name in entry_grades:
                # Per-entry grade is a direct quality signal for this tool
                attributed_reward = entry_grades[tool_name]
            else:
                # Fallback: proportional share of episode reward
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
