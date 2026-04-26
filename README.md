---
title: Genesis Environment
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Genesis — The Coding Agent That Fixes Its Own Tools

> *What if the agent could fire its bad tools and hire better ones?*

---

## The Problem

Every LLM agent ships with a fixed toolbox. A broken search tool? The agent learns to work around it. A useless error explainer? The agent ignores it and guesses. The tools rot quietly, and nobody notices — because the agent adapts to the tools instead of the other way around.

This is the **static toolbox ceiling**. The agent can only ever be as good as its worst tool.

---

## The Idea

Genesis is a coding agent environment where the agent doesn't just solve problems — it **identifies its own underperforming tools and rewrites them**.

Every tool call is graded. The grades are tracked over time. When a tool consistently scores poorly, a second LLM (the **Tool Architect**) reads the grades, reads the broken tool's source code, and rewrites it. Then we measure if the new tool actually helped. If it didn't, we revert. If it did, we keep it — and use the improvement delta to train the Tool Architect to write better rewrites next time.

It's turtles all the way down. Except the turtles are getting smarter.

---

## How It Works

```
Agent solves task → tools get graded → worst tool gets rewritten
→ agent re-evaluated → delta measured → Tool Architect trained on that delta
→ repeat
```

The reward is brutally honest:

```
reward = tests_passed × 0.6 + tool_quality × 0.2 + reasoning_quality × 0.2
```

Each tool call gets its own grade based on *what it returned* — not just which tool produced it. This means newly evolved tools get graded automatically, no code changes needed.

---

## What Actually Evolves

Five tools live on the agent side as plain Python files. The Tool Architect rewrites them locally — no server restart, no deployment, just a file write:

| Tool | What it does | Starts as |
|------|-------------|-----------|
| `search_code_examples` | Finds similar code | Naive BM25 keyword search |
| `run_tests` | Runs test cases | Works fine, actually |
| `lint_code` | Static analysis | Extremely verbose, no severity ranking |
| `fetch_docs` | Fetches Python docs | Dumps everything, filters nothing |
| `explain_error` | Diagnoses tracebacks | Generic templates, no line-specific insight |

The weak ones get noticed. The Tool Architect gets to work. Things improve — or they don't, and we revert and try again.

---

## The Training Loop

The Tool Architect is fine-tuned with **GRPO** (Group Relative Policy Optimization). For each improvement attempt, we sample multiple rewrite strategies, run them through the agent, measure the reward delta, and update using group-normalized advantages. No separate critic — the environment *is* the critic.

On top of that, the agent's own LLM gets fine-tuned in parallel with GRPO on the benchmark tasks — so both the model weights *and* the tool files improve together each cycle.

Combined loop per cycle:
1. **GRPO fine-tune** the agent LLM on a batch of tasks
2. **Self-improve**: evaluate → Tool Architect rewrites worst tool → re-evaluate → revert if worse

---

## Architecture

```
[Your Machine / Colab]              [HuggingFace Space]
  Agent + 5 tool files   ──POST──▶  Genesis Env Server
  Tool Architect                     scores code, grades tools
  GRPO trainer                       returns reward + per-tool grades
```

The server is a stateless evaluator. It holds the benchmark tasks and hidden test suites. The agent and tools run wherever you run the training script.

---

## Dive Deeper

| Document | What's in it |
|----------|-------------|
| [Environment Design](PROJECT.md) | Full reward formula, grader system, tool weight tracking, ToolRegistry logic |
| [Reward Mechanism](reward_mechanism.md) | How per-tool grades are computed and aggregated |
| [Hackathon Brief](Hackathon.md) | The original problem statement |
| [Training Notebook](training/self_improve.ipynb) | Run everything in Colab — self-improvement loop + GRPO cells |

---

## Run It

```bash
# Clone and set up
git clone https://huggingface.co/spaces/berlin1906/genesis_env
cd genesis_env
pip install -r requirements.txt
cp .env.example .env  # add your HF_TOKEN

# Run one episode (see the agent + tools in action)
python inference.py

# Run the self-improvement loop
python training/self_improve.py --cycles 3 --n 3

# Run GRPO + self-improvement together
python training/combined_loop.py --cycles 3 --n 3 --batch-size 4

# Or just open the notebook
jupyter notebook training/self_improve.ipynb
```

---

## The Punchline

Most agents are handed a toolbox and told to get on with it.

Genesis asks: *what if the agent could look at its tools, decide they're terrible, and build better ones?*

That's the whole bet. And the loop that makes it work — grade every call, track what's weak, rewrite it, measure the delta, train on the delta — is surprisingly simple once you stop trying to make the agent smarter and start making it self-critical.
