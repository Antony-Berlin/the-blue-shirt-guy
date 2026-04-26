# Reward Mechanism — Future Improvement

## The Problem with the Current Design

Right now the EMA tracker attributes reward to tools based on **usage count**.

If `run_tests` was called 3 times and `web_search` once, `run_tests` gets 75% of the episode reward — regardless of whether those calls were actually good.

That's wrong. A single brilliant `web_search` that unlocked the entire solution deserves more credit than three mediocre `run_tests` calls.

---

## The Better Idea: Return Decomposition

Instead of splitting reward by count, attribute it **backwards through the call sequence** based on each tool's actual contribution. This is the credit assignment problem — the same thing backpropagation solves in neural networks.

**Example:**

```
Tool calls:   web_search → run_tests → explain_error → run_tests
Grades:          0.6          0.4           0.9            1.0
Final reward: 0.85
```

The question becomes: *how much did each tool call cause the final reward?*

---

## How It Works: Discounted Return Attribution

**Step 1 — Compute a discounted return for each step** (like Q-values in RL):

```
G_i = grade_i + γ × grade_{i+1} + γ² × grade_{i+2} + ...
```

With discount factor `γ = 0.9`. This gives early tool calls that *set up* later success proportional credit — not just the last call before submit.

**Step 2 — Normalize and scale by final reward:**

```
attributed_reward_i = (G_i / sum(G)) × final_reward
```

Each tool gets a share of the final reward proportional to its discounted contribution through the sequence.

**Step 3 — Update EMA with attributed reward instead of raw grade.**

---

## The Code Change (3 lines in `tool_registry.update()`)

```python
γ = 0.9
G = 0.0
returns = []
for grade in reversed(tool_grades):
    G = grade + γ * G
    returns.insert(0, G)

total = sum(returns) or 1.0
attributed = [(r / total) * final_reward for r in returns]
# then: update each tool's EMA with attributed[i] instead of grade[i]
```

Everything needed is already in the system — `tool_grades`, `final_reward`, and the sequence order in `tool_usage_log`. This is a drop-in improvement to `tool_registry.update()`.

---

## Why This Matters

| | Current (count-based) | Improved (return decomposition) |
|---|---|---|
| `run_tests` called 3×, mediocre | gets 75% of reward | gets proportional to actual grades |
| `web_search` called once, brilliant | gets 25% of reward | gets higher share if it enabled success |
| Early calls that set up later wins | undervalued | credited via discount factor |

The Tool Architect makes better rewrite decisions when the signal it receives actually reflects which tool was responsible for the outcome — not just which tool was called the most.
