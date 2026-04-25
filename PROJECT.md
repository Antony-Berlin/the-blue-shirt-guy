# Genesis Environment — Self-Improving Coding Agent

## Problem & Motivation

LLM agents are equipped with static tool sets. When a tool is poorly designed — a broken test runner, a weak code search utility, an unhelpful error explainer — the agent learns to work around its limitations rather than fix them. This permanently caps the agent's potential.

**Genesis Environment** breaks this ceiling. It is a coding agent that:
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
│  2. Environment scores: reward + per-tool-call grades +          │
│     natural language feedback + per-tool EMA weights            │
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

## System Architecture

```
[Local / Training Side]                     [HuggingFace Space]
  Research Agent                   HTTP       Genesis Env Server
  + 5 tool implementations  ──────POST──▶   POST /reset  → task + starter code
  + ToolExecutor (logs calls)                POST /step   → evaluate submission
  + Tool Architect                           GET  /state  → weights snapshot
  + GRPO trainer
```

The env server is a **stateless evaluator** — it holds tasks, runs hidden tests, and grades tool usage. The agent runs locally, calls tools as plain Python functions, logs every call, then POSTs the log + final code to the env.

The Tool Architect rewrites tool files **locally** (`agent/tools/`). No server restart required.

---

## Environment Design

### Task Domain: Programming Problems

The Research Agent is given programming tasks drawn from a benchmark of 50 episodes (HumanEval subset + original tasks):

- *"Implement a function `most_frequent(lst)` that returns the most frequently occurring element."*
- *"Write a function that detects if a linked list has a cycle."*

Each task has a problem description, starter code, a difficulty label (easy/medium/hard), and a hidden test suite.

### Reward Formula

```
reward = pass_score × 0.6 + tool_usage_score × 0.2 + reasoning_score × 0.2
```

| Component | Source | Weight |
|-----------|--------|--------|
| `pass_score` | Fraction of hidden tests passing | 0.6 |
| `tool_usage_score` | Mean per-tool-call grade from GraderRouter | 0.2 |
| `reasoning_score` | LLM-as-judge holistic quality score | 0.2 |

Each `step()` returns a `GenEnvObservation` with:

| Field | Type | Description |
|-------|------|-------------|
| `reward` | `float` (0–1) | Combined reward from formula above |
| `tool_grades` | `list[float]` | Per-tool-call score [0.0, 1.0] in submission order |
| `nl_feedback` | `str` | LLM-as-judge critique of tool usage and reasoning |
| `tool_weights` | `dict[str, float]` | EMA performance weight per tool name |

### Multi-Grader System (`server/tool_graders.py`)

Each tool call log entry is routed to graders based on **what the result contains**, not which tool produced it. This means new tools added during self-evolution are graded automatically without code changes.

#### Content-Type Detection

| Content Type | Detection Signal |
|---|---|
| `CODE` | Contains `def `, `class `, `import `, or multiline indented block |
| `TEST_RESULT` | Starts with `PASSED:` or `FAILED:` |
| `ERROR_TRACEBACK` | Contains `Traceback`, `TypeError:`, `Error:`, etc. |
| `SEARCH_RESULT` | Numbered result lines or similarity scores |
| `DOCUMENTATION` | Contains pydoc-style headers (`Help on`, `DESCRIPTION`, `def`/`class` signatures) |
| `TEXT` | Everything else (catch-all) |

#### Per-Entry Graders

| Grader | Input Type | What It Checks |
|--------|-----------|----------------|
| `CodeSyntaxGrader` | `CODE` | AST parse validity, placeholder detection (`pass`/`...`/`NotImplementedError`), nesting depth |
| `CodeStyleGrader` | `CODE` | McCabe cyclomatic complexity, single-char variable names |
| `TestResultGrader` | `TEST_RESULT` | Pass rate (N/M parsing), informative failure messages |
| `ErrorExplanationGrader` | `ERROR_TRACEBACK` | Error type referenced, line number cited, fix suggested, information density |
| `SearchRelevanceGrader` | `SEARCH_RESULT` | Result count, Jaccard overlap with query |
| `DocQualityGrader` | `DOCUMENTATION` | Length, signature presence, truncation detection |
| `ReasoningGrader` | `TEXT` | Information density (content tokens / total), context reference alignment |

#### Sequence-Level Graders (applied across full log)

| Grader | What It Checks |
|--------|----------------|
| `RedundancyGrader` | Same tool + same args within 3 steps → −0.3 penalty on duplicate entry |
| `ErrorPropagationGrader` | Tool errored → next call ignores it → −0.4 penalty; using `explain_error` after error → +0.2 bonus |

#### Already-Graded Skip Logic

Each log entry carries `graded: bool` and `grade: float` fields. When the same log is re-submitted (e.g., during iterative refinement), already-graded entries are skipped and their cached scores are reused.

### Tool Weight Attribution (ToolRegistry)

After each episode, the `ToolRegistry` updates EMA weights using **per-entry grades** (not uniform episode reward). This gives each tool a quality signal proportional to the actual usefulness of its outputs.

| EMA Weight | Flag | Action |
|------------|------|--------|
| > 0.7 | `KEEP` | No change |
| 0.4–0.7 | `REVIEW` | Candidate for improvement |
| < 0.4 | `REPLACE` | Primary target for Tool Architect |

### The 5 Evolvable Tools (client-side, `agent/tools/`)

Tools are standalone Python files on the agent side. The Tool Architect rewrites them locally — no server restart needed.

| Tool | Purpose | Initial Quality |
|------|---------|-----------------|
| `search_code_examples` | BM25 keyword search over local code corpus | Medium — naive lexical search, small corpus |
| `run_tests` | Executes test cases against agent code in subprocess | High — objective, direct signal |
| `lint_code` | Static analysis via pyflakes | Low — too verbose, no severity ranking |
| `fetch_docs` | Retrieves Python pydoc for libraries/symbols | Medium — unfiltered, no relevance trimming |
| `explain_error` | Diagnoses Python tracebacks | Low — generic templates, no line-specific diagnosis |

---

## Repository Structure

```
scaler-fin/
├── envs/gen_env/
│   ├── openenv.yaml                      ← spec_version 1 manifest (name: gen_env)
│   ├── models.py                         ← GenEnvAction / GenEnvObservation / GenEnvState
│   └── server/
│       ├── app.py                        ← FastAPI app via create_app()
│       ├── gen_env_environment.py        ← GenesisEnvironment(Environment)
│       ├── tool_registry.py              ← EMA weight tracker + KEEP/REVIEW/REPLACE
│       ├── rubric.py                     ← LLM-as-judge holistic scorer
│       └── tool_graders.py              ← Multi-grader: ContentTypeDetector + 9 graders
├── agent/
│   ├── tools/                            ← evolvable tool implementations (client-side)
│   │   ├── search_code_examples.py
│   │   ├── run_tests.py
│   │   ├── lint_code.py
│   │   ├── fetch_docs.py
│   │   └── explain_error.py
│   ├── tool_executor.py                  ← dynamic tool loader + call logger
│   ├── env_http_client.py               ← HTTP client: POST /reset, /step; GET /state
│   └── research_agent.py                ← LLM agent loop (tool use → submit)
├── tasks/
│   └── benchmark.json                    ← 50 programming tasks with hidden test suites
├── training/
│   ├── train_tool_architect.py           ← GRPOTrainer + tool_improvement_reward
│   ├── coding_agent.py                   ← OpenCode CLI wrapper
│   ├── tool_patcher.py                   ← sandboxed env with injected tool
│   ├── evaluation.py                     ← episode runner returning mean reward
│   ├── reward.py                         ← tool_improvement_reward function
│   └── dataset.py                        ← GRPO training dataset builder
├── tests/
│   └── test_tool_graders.py              ← 48 unit tests for all grader classes
├── inference.py                          ← standalone inference script (direct env import)
├── scripts/
│   └── pre_validation.sh                 ← HF Space ping + openenv validate checker
├── docker-compose.yml
├── Dockerfile
└── PROJECT.md
```

---

## Two Separate LLMs

**Research Agent (Primary LLM)** — frozen during Tool Architect training
- Role: solves coding tasks using the available tools
- Model: `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router (inference) or local vLLM (training)
- Frozen during Tool Architect training; re-evaluated before/after tool improvements

**Tool Architect (Secondary LLM)** — trained with GRPO
- Role: reads tool performance data and generates OpenCode instructions
- Model: Qwen2.5-7B-Instruct, fine-tuned with Unsloth + TRL GRPOTrainer
- Reward: improvement delta `r_new - r_old` clipped to `[-1, 1]`

---

## GRPO Training Loop

```
For each training step:
  prompt = {tool_weights, tool_grades, nl_feedback, tool_name, current_tool_source_code}
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

## Hackathon Requirements

| Requirement | Implementation |
|-------------|----------------|
| **Use OpenEnv** | `GenesisEnvironment(Environment)` from `openenv-core` |
| **Gym-style API** | `reset()`, `step()`, `state` property all implemented |
| **Valid `openenv.yaml`** | `spec_version: 1`, name `gen_env`, runtime, app, port, variables |
| **No reserved tool names** | Tools: `search_code_examples`, `run_tests`, `lint_code`, `fetch_docs`, `explain_error` |
| **Training script** | `train_tool_architect_colab.ipynb` using Unsloth + TRL `GRPOTrainer` |
| **Evidence of training** | Per-tool improvement delta curves, loss plots, before/after reward histograms |
| **Short writeup** | HuggingFace blog post with loop diagram, reward curves, example tool evolution |

---

## Evidence of Self-Improvement

**Primary metric: Tool Improvement Delta** — average reward gain when the Research Agent uses a Tool Architect-improved tool vs the original, measured across 50 benchmark tasks. Positive and increasing delta over GRPO training steps = self-programming loop working.

**Secondary evidence:**
- `tool_grades` distribution shifts right after tool improvements (graders score improved tool outputs higher)
- NL feedback shifts character: from *"explain_error returned a generic message"* to *"the improved diagnostic correctly identified the off-by-one in the loop boundary"*
- Tool EMA weight distribution shifts toward KEEP zone after training
- Before/after reward histogram: research agent scores higher on the full benchmark after tool improvements

---

## Quickstart

```bash
# Validate the environment
pip install openenv-core
openenv validate envs/gen_env/

# Run a single inference episode
cp .env.example .env   # set HF_TOKEN, API_BASE_URL, MODEL_NAME
python inference.py

# Run the env server locally
docker-compose up

# Run grader unit tests
pip install pytest
pytest tests/test_tool_graders.py -v
```

---

## Links

- HuggingFace Space (environment): TBD
- Training Notebook (Colab): TBD
- Demo Video (<2 min): TBD
- HuggingFace Blog Post: TBD
