# CodeForge Self-Programmer

## Problem & Motivation

LLM agents are equipped with static tool sets. When a tool is poorly designed — a broken test runner, a weak code search utility, an unhelpful error explainer — the agent learns to work around its limitations rather than fix them. This permanently caps the agent's potential.

**CodeForge Self-Programmer** breaks this ceiling. It is a coding agent that:
1. Solves programming tasks using a suite of tools
2. Receives graded feedback identifying which tools are underperforming
3. Commissions AI-driven rewrites of those tools via **OpenCode**
4. Measures the improvement, and uses the improvement delta to train the tool-improving system itself

The theme is maximally literal — a **coding agent that improves its own coding tools** through reinforcement learning.

---

## The Self-Improvement Loop

```
┌────────────────────────────────────────────────────────────────┐
│                  SELF-IMPROVEMENT CYCLE                         │
│                                                                  │
│  1. Research Agent uses tools to solve programming tasks         │
│  2. Environment scores: decimal reward + natural language        │
│     feedback + per-tool EMA performance weights                  │
│  3. ToolRegistry flags underperforming tools (weight < 0.4)     │
│  4. Tool Architect LLM reads weights + feedback + tool source    │
│  5. Tool Architect generates precise coding instruction          │
│  6. OpenCode executes instruction → new tool implementation      │
│  7. New agent (new tools) vs old agent (old tools) evaluated     │
│  8. Improvement delta (r_new - r_old) trains Tool Architect      │
│     via GRPO                                                     │
│  9. Loop continues with improved tools                           │
└────────────────────────────────────────────────────────────────┘
```

The key insight: the **GRPO reward teaches the Tool Architect not just to generate code, but to generate code that produces measurable improvement in a downstream RL agent**. This is meta-learning in its most literal form.

---

## Environment Design

### Task Domain: Programming Problems

The Research Agent is given programming tasks drawn from a benchmark of 50 episodes (HumanEval subset + original tasks):

- *"Implement a function `most_frequent(lst)` that returns the most frequently occurring element. Use the available tools to search for reference implementations and validate your solution."*
- *"Write a function that detects if a linked list has a cycle. Fetch documentation for Python's `sys` module if needed, and run tests to verify."*

Each task has a problem description, starter code, a difficulty label (easy/medium/hard), and a hidden test suite.

### Dual Reward Signal

Each episode's `step()` returns an `Observation` with:

| Field | Type | Description |
|-------|------|-------------|
| `reward` | `float` (0–1) | `pass@1` test score × 0.7 + LLM reasoning quality × 0.3 |
| `metadata["nl_feedback"]` | `str` | LLM-as-judge critique of the agent's tool usage and reasoning |
| `metadata["tool_weights"]` | `dict[str, float]` | EMA of rewards when each tool was active during the episode |

**Example NL feedback:** *"The `explain_error` tool returned a generic message that didn't reference the agent's specific traceback. The agent was forced to hallucinate the fix rather than reason from the actual error context."*

### Tool Weight Attribution (ToolRegistry)

The `ToolRegistry` tracks an Exponential Moving Average of rewards for each tool across episodes. Attribution is proportional to usage frequency within the trajectory.

After each evaluation round:

| EMA Weight | Flag | Action |
|------------|------|--------|
| > 0.7 | `KEEP` | No change |
| 0.4–0.7 | `REVIEW` | Candidate for improvement |
| < 0.4 | `REPLACE` | Primary target for Tool Architect |

### The 5 Evolvable MCP Tools

| Tool | Purpose | Initial Quality |
|------|---------|-----------------|
| `search_code_examples` | Searches a local corpus for similar implementations | Medium (improvable: naive BM25 search) |
| `run_tests` | Executes the test suite against agent-generated code | High (objective) |
| `lint_code` | Static analysis and style feedback | Low (improvable: too verbose, low signal) |
| `fetch_docs` | Retrieves documentation for a library/function | Medium (improvable: no context filtering) |
| `explain_error` | Takes a traceback + code, returns diagnosis | Low (improvable: generic messages) |

Tools are implemented as standalone Python files in `server/tools/` so they can be rewritten by OpenCode without touching the environment framework.

---

## System Architecture

### Two Separate LLMs

**Research Agent (Primary LLM)** — frozen during Tool Architect training
- Role: solves coding tasks using the available tools
- Model: Qwen2.5-7B-Instruct (or any capable instruction model)
- Frozen during Tool Architect training; re-evaluated before/after tool improvements

**Tool Architect (Secondary LLM)** — trained with GRPO
- Role: reads tool performance data and generates OpenCode instructions
- Model: Qwen2.5-7B-Instruct, fine-tuned with Unsloth + TRL GRPOTrainer
- Reward: improvement delta `r_new - r_old` clipped to `[-1, 1]`

### OpenEnv Integration

`CodeForgeEnvironment` extends `MCPEnvironment` from `openenv-core`:

```python
class CodeForgeEnvironment(MCPEnvironment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def reset(self, seed=None, **kwargs) -> CodeForgeObservation:
        """Sample a programming task and reset episode state."""

    def _step_impl(self, action, **kwargs) -> CodeForgeObservation:
        """Handle SubmitCodeAction: run tests, score with rubric,
        update ToolRegistry, return observation with dual reward."""

    @property
    def state(self) -> CodeForgeState:
        """Return episode metadata including tool usage log."""
```

MCP tools are registered via FastMCP `@mcp.tool()` decorators. No tool uses reserved names (`reset`, `step`, `state`, `close`).

### OpenCode Integration

The Tool Architect generates a structured coding instruction. The `CodingAgentClient` passes this to OpenCode:

```
opencode run --prompt "<instruction>" --file server/tools/<tool_name>.py
```

The resulting modified file is captured and passed to the `ToolPatcher`, which injects it into a sandboxed environment copy for evaluation.

**Example Tool Architect instruction:**
> *"Rewrite `explain_error.py`. The current implementation returns a generic error description. Instead: (1) parse the traceback to identify the exact line and exception type, (2) include the relevant code snippet in context, (3) return a diagnosis that references the specific variable or expression causing the failure. Expected improvement: agent should be able to fix errors in 1–2 attempts instead of 3–4."*

### GRPO Training Loop

```
For each training step:
  prompt = {tool_weights, nl_feedback, tool_name, current_tool_source_code}
  completions = Tool Architect generates G=8 coding instructions
  
  for each completion:
    new_code = OpenCode(completion)           # execute instruction
    r_new    = evaluate(new_code, n=5)        # run 5 episodes
    r_old    = evaluate(current_code, n=5)    # baseline
    reward_i = clip(r_new - r_old, -1, 1)
  
  advantages = (rewards - mean(rewards)) / std(rewards)
  update Tool Architect policy via GRPO gradient
```

**Trainer config:** Qwen2.5-7B-Instruct, Unsloth 4-bit quantization (T4 GPU), TRL `GRPOTrainer`, G=8 completions, lr=5e-6, 3 epochs.

---

## Repository Structure

```
scaler-fin/
├── envs/codeforge_env/
│   ├── openenv.yaml                      ← spec_version 1 manifest
│   ├── models.py                         ← Action / Observation / State models
│   ├── client.py                         ← OpenEnv client
│   └── server/
│       ├── app.py                        ← FastAPI app via create_app()
│       ├── codeforge_environment.py      ← MCPEnvironment subclass
│       ├── tool_registry.py              ← EMA weight tracker + KEEP/REVIEW/REPLACE
│       ├── rubric.py                     ← LLM-as-judge scorer
│       └── tools/                        ← evolvable tool implementations
│           ├── search_code_examples.py
│           ├── run_tests.py
│           ├── lint_code.py
│           ├── fetch_docs.py
│           └── explain_error.py
├── tasks/
│   └── benchmark.json                    ← 50 programming tasks with test suites
├── training/
│   ├── train_tool_architect.py           ← GRPOTrainer + tool_improvement_reward
│   ├── coding_agent.py                   ← OpenCode CLI wrapper
│   ├── tool_patcher.py                   ← sandboxed env with injected tool
│   ├── evaluation.py                     ← episode runner returning mean reward
│   ├── reward.py                         ← tool_improvement_reward function
│   └── dataset.py                        ← GRPO training dataset builder
├── notebooks/
│   └── train_tool_architect_colab.ipynb  ← end-to-end training + plots
├── PROJECT.md
├── README.md
└── Hackathon.md
```

---

## Hackathon Requirements

| Requirement | Implementation |
|-------------|----------------|
| **Use OpenEnv** | `CodeForgeEnvironment(MCPEnvironment)` from `openenv-core` |
| **Gym-style API** | `reset()`, `step()`, `state` property all implemented |
| **Valid `openenv.yaml`** | `spec_version: 1`, name, type, runtime, app, port, variables |
| **No reserved tool names** | Tools: `search_code_examples`, `run_tests`, `lint_code`, `fetch_docs`, `explain_error` |
| **Training script** | `train_tool_architect_colab.ipynb` using Unsloth + TRL `GRPOTrainer` |
| **Evidence of training** | Per-tool improvement delta curves, loss plots, before/after reward histograms |
| **Short writeup** | HuggingFace blog post with loop diagram, reward curves, example tool evolution |

---

## Evidence of Self-Improvement

**Primary metric: Tool Improvement Delta** — the average reward gain when the Research Agent uses a Tool Architect-improved tool vs the original, measured across 50 benchmark tasks. Positive and increasing delta over GRPO training steps = self-programming loop working.

**Secondary evidence:**
- NL feedback shifts character over training rounds: from *"the explain_error tool returned a generic message"* to *"the improved error diagnostic tool correctly identified the off-by-one in the loop boundary"*
- Tool EMA weight distribution shifts right (more tools in KEEP zone) after training
- Before/after reward histogram: research agent scores higher on the full benchmark after tool improvements

---

## Quickstart

```bash
pip install openenv-core trl unsloth
openenv validate envs/codeforge_env/
openenv serve envs/codeforge_env/
jupyter notebook notebooks/train_tool_architect_colab.ipynb
```

---

## Links

- HuggingFace Space (environment): TBD
- Training Notebook (Colab): TBD
- Demo Video (<2 min): TBD
- HuggingFace Blog Post: TBD
