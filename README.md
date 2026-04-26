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

> *What if the agent could look at its bad tools, fire them, and hire better ones?*

---

## The Problem

Every LLM agent ships with a fixed toolbox. A broken search tool? The agent learns to work around it. A useless error explainer? The agent ignores it and guesses anyway. The tools rot quietly — because the agent adapts to the tools instead of the other way around.

This is the **static toolbox ceiling**. The agent can only ever be as good as its worst tool.

---

## The Idea

Genesis is a coding agent environment where the agent doesn't just solve programming tasks — it **identifies its own underperforming tools and rewrites them**.

Every single tool call is graded mid-episode. The grades are tracked over time with an EMA. When a tool consistently underperforms, a second LLM (the **Tool Architect**) reads the grades, reads the broken tool's source code, and rewrites it. We measure if the rewrite actually helped. If it didn't, we revert to the backup. If it did, we keep it — and use the improvement delta to train the Tool Architect to write better rewrites next time.

It's turtles all the way down. Except the turtles keep getting smarter.

---

## How It Works

The agent gets a programming task, calls tools to research and test a solution, then submits code. The environment grades everything:

```
reward = tests_passed × 0.6 + tool_quality × 0.2 + reasoning_quality × 0.2
```

Tool calls are graded *mid-episode* by routing each result to a content-type-specific grader — code syntax checker, test result parser, error explanation rater, search relevance scorer, and so on. New tools added during evolution get graded automatically, no code changes needed.

After enough episodes, the tool EMA weights tell the story:

| EMA Weight | Verdict | What happens |
|---|---|---|
| > 0.7 | `KEEP` | Nothing, it's working |
| 0.4–0.7 | `REVIEW` | Candidate for improvement |
| < 0.4 | `REPLACE` | Tool Architect gets called |

---

## What Actually Evolves

Five tools live as plain Python files on the agent side. The Tool Architect rewrites them locally — just a file write, no server restart:

| Tool | What it does | Starts as |
|------|-------------|-----------|
| `search_code_examples` | BM25 search over a code corpus | 10 static examples, purely lexical |
| `run_tests` | Runs assert statements in a subprocess | Already decent, actually |
| `lint_code` | Static analysis via pyflakes | Too verbose, re-ranked by frequency |
| `fetch_docs` | pydoc for libraries/symbols | Raw unfiltered dump, truncated at 2000 chars |
| `explain_error` | Diagnoses Python tracebacks | Generic templates, no line-specific analysis |

The weak ones get noticed. The Tool Architect reads the tool source + grades + NL feedback and decides whether to rewrite an existing tool or create an entirely new one. If the delta is negative — revert. Simple.

---

## The Training Loop

Two things improve in parallel each cycle:

**1. GRPO fine-tunes the agent LLM** on a batch of benchmark tasks. Each task generates G completions, each gets scored by the environment, and the model is updated using group-normalized advantages. No separate critic.

**2. Self-improvement evolves the tools** — evaluate N episodes → Tool Architect rewrites worst tool → evaluate N more episodes → keep or revert based on delta.

```
for each cycle:
  ① GRPO: sample G completions per task → score → update model weights
  ② Eval N episodes → pick worst tool → rewrite → eval N more → Δ > 0? keep : revert
```

The Tool Architect itself is an LLM fine-tuned with GRPO where the reward *is* the improvement delta. It learns not just to generate code, but to generate code that makes the downstream agent measurably better.

---

## Architecture

```
[Colab / Local]                         [HuggingFace Space]
  Agent + 5 tool files    ──POST──▶     Genesis Env Server
  Tool Architect                         • runs hidden tests
  GRPO trainer                           • grades every tool call
                                         • returns reward + per-tool grades + NL feedback
```

The server is a stateless evaluator. It holds the benchmark tasks and hidden test suites. The agent, tools, and training all run on your machine (or Colab).

---

## Dive Deeper

| What | Where |
|------|-------|
| Full environment design — reward formula, grader system, tool weight tracking | [PROJECT.md](PROJECT.md) |
| Reward mechanism detail | [reward_mechanism.md](reward_mechanism.md) |
| Interactive notebook — run everything in Colab | [training/self_improve.ipynb](training/self_improve.ipynb) |

---

## Run It

```bash
git clone https://huggingface.co/spaces/berlin1906/genesis_env
cd genesis_env
pip install -r requirements.txt
cp .env.example .env   # set HF_TOKEN + MODEL_NAME

# Watch one episode
python inference.py

# Run the self-improvement loop (3 cycles, 3 eval episodes each)
python training/self_improve.py --cycles 3 --n 3

# Run GRPO fine-tuning + self-improvement together
python training/combined_loop.py --cycles 3 --n 3 --batch-size 4

# Or just open the notebook
jupyter notebook training/self_improve.ipynb
```

---

## The Punchline

Most agents are handed a toolbox and told to get on with it.

Genesis asks: *what if the agent could look at its tools, decide they're bad, and build better ones?*

The loop that makes it work is surprisingly simple — grade every call, track what's weak, rewrite it, measure the delta, train on the delta. No magical self-awareness required. Just relentless self-criticism, backed by a revert button.
