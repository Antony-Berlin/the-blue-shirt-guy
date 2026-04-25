  ---
    The Core Idea

    Instead of splitting the final reward by usage count (what the current EMA does), you want to attribute the final reward backwards through the tool call sequence based on each tool's actual contribution.

    This is called Return Decomposition or Reward Redistribution.

    ---
    How It Would Work Here

    Tool calls:  [search] → [run_tests] → [explain_error] → [run_tests]
    Grades:      [0.6]       [0.4]          [0.9]             [1.0]
    Final reward: 0.85

    Question: how much did each tool *cause* the final reward?

    Step 1 — Compute a "causal contribution" score

    For each tool call at position i, ask: "How much did the trajectory improve after this call?"

    contribution_i = grade_i × (final_reward - baseline_reward_without_this_call)

    You can approximate baseline_reward_without_this_call using the running average reward before step i — the counterfactual "what would the reward have been if the agent stopped here."

    Step 2 — Use cumulative discounted grades as the attribution

    Similar to how Q-values work in RL:

    G_i = grade_i + γ × grade_{i+1} + γ² × grade_{i+2} + ...

    With discount factor γ = 0.9. This makes early tool calls that set up later success get credit, not just the last call before submit.

    Step 3 — Rescale by final reward

    attributed_reward_i = (G_i / sum(G)) × final_reward

    Now each tool gets a share of the final reward proportional to its discounted contribution through the sequence.

    ---
    The Backpropagation Analogy

    What you're describing is essentially the credit assignment problem — the same thing backprop solves in neural networks. The difference:

    - Backprop has a differentiable computation graph → exact gradients
    - Tool call sequences are discrete and non-differentiable → must approximate

    The approximations used in practice:

    ┌─────────────────────────────┬─────────────────────────────────────────────────────────────┬───────────────────────────┐
    │           Method            │                             How                             │            Fit            │
    ├─────────────────────────────┼─────────────────────────────────────────────────────────────┼───────────────────────────┤
    │ REINFORCE with baseline     │ (G_i - mean(G)) × log_prob(action_i)                        │ Good for GRPO             │
    ├─────────────────────────────┼─────────────────────────────────────────────────────────────┼───────────────────────────┤
    │ Advantage estimation (GAE)  │ Weighted blend of n-step returns                            │ Best credit assignment    │
    ├─────────────────────────────┼─────────────────────────────────────────────────────────────┼───────────────────────────┤
    │ Attention-based attribution │ Train a small model to predict which step caused the reward │ Complex but most accurate │
    └─────────────────────────────┴─────────────────────────────────────────────────────────────┴───────────────────────────┘

    ---
    What This Gives You Over Current Design

    Current EMA: run_tests called 3×, search called 1× → run_tests gets 75% of reward regardless of quality.

    With return decomposition: if search at step 1 had grade 0.9 and directly enabled the agent to write correct code, it gets higher attribution even though it was only called once.
  ---
    Practical Recommendation

    The current architecture already has everything needed to implement GAE-style attribution:

    1. tool_grades[i] — per-step quality signal (already computed)
    2. final_reward — outcome signal (already returned)
    3. Tool sequence order — preserved in tool_usage_log

    The formula to add to tool_registry.update():

    γ = 0.9
    G = 0.0
    returns = []
    for grade in reversed(tool_grades):
        G = grade + γ * G
        returns.insert(0, G)

    # Normalize by sum so attribution sums to final_reward
    total = sum(returns) or 1.0
    attributed = [(r / total) * final_reward for r in returns]

    Then update each tool's EMA with attributed[i] instead of grade[i]. This is the closest practical approximation to "backpropagating through the tool sequence" without needing a differentiable model over the
    tool calls.