---
title: Genesis Environment
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# The Blue Shirt Guy Project

You know that movie *Free Guy* — where an NPC wakes up, realises he's inside a game, and starts improving himself beyond what he was programmed to do?

That's the one-line pitch. Except here the NPC is a coding agent, and instead of learning to fight, it learns to fix its own broken tools.

---

## The Problem

LLM agents are given tools and told to get on with it. The tools are static. When one is bad — a useless error explainer, a search that barely searches, a linter that dumps everything at you — the agent just works around it. It adapts to the limitation instead of removing it.

The ceiling is fixed. We wanted to break it.

---

## What We Built

An environment called **Genesis** where a coding agent solves programming tasks using five tools. Each tool call gets graded in real time. The grades build up into a performance score per tool. When a tool consistently scores badly, a second LLM — the **Tool Architect** — steps in, reads the source code of the bad tool, reads the grades and feedback, and rewrites it.

Then we measure if things got better. If yes, keep the new tool. If no, revert to the backup and try again next cycle.

The Tool Architect is itself trained with GRPO — its reward is the actual improvement delta the rewrite produced in the agent's score. It gets better at rewriting tools by watching what works and what doesn't.

On top of that, the agent's own LLM gets fine-tuned on benchmark tasks in parallel using GRPO. Every cycle, both the model weights and the tool files improve.

---

## The Loop

```
① Agent solves tasks → every tool call graded live
② Worst tool flagged by EMA tracker
③ Tool Architect rewrites it
④ Agent re-evaluated → delta measured
⑤ delta > 0 → keep   |   delta < 0 → revert
⑥ Tool Architect trained on that delta
⑦ Repeat
```

---

## The Tools (what they start as)

Five plain Python files. Intentionally imperfect. The point is for the system to find the weak ones on its own.

| Tool | What it does | Starting quality |
|------|-------------|-----------------|
| `search_code_examples` | BM25 search over a code corpus | 10 hardcoded examples, purely lexical |
| `run_tests` | Runs assert statements in a subprocess | Actually solid |
| `lint_code` | Calls pyflakes, re-ranks by frequency | Verbose, noisy |
| `fetch_docs` | pydoc for libraries/symbols | Raw unfiltered dump, 2000 char cutoff |
| `explain_error` | Diagnoses Python tracebacks | Template-based, no line-specific analysis |

---

## The Reward

```
reward = tests_passed × 0.6 + tool_quality × 0.2 + reasoning_quality × 0.2
```

Tool quality is computed by a multi-grader system that automatically detects what kind of result a tool returned — code, test output, error explanation, search results, documentation — and routes it to the appropriate grader. New tools added during evolution get graded automatically without any code changes.

Reasoning is scored by an LLM judge, with Anthropic as fallback, then a heuristic if both fail.

---

## Why It's Interesting

Most self-improvement work focuses on the model — better weights, better prompts. This focuses on the environment the model operates in. The tools are part of the agent's capability, and they're just files. They can be rewritten.

The result is a system where both the model and its runtime environment improve together, each cycle informed by actual task performance.

---

## Try It

```bash
git clone https://huggingface.co/spaces/berlin1906/genesis_env
cd genesis_env
pip install -r requirements.txt
cp .env.example .env   # set HF_TOKEN and MODEL_NAME

python inference.py                                          # single episode
python training/self_improve.py --cycles 3 --n 3            # tool evolution loop
python training/combined_loop.py --cycles 3 --batch-size 4  # GRPO + tool evolution
```

Or run everything in Colab: [`training/self_improve.ipynb`](training/self_improve.ipynb)

---

## Go Deeper

- **[PROJECT.md](PROJECT.md)** — full design: reward formula, grader architecture, EMA weight tracking, Tool Architect prompt design
- **[reward_mechanism.md](reward_mechanism.md)** — how individual tool calls get scored
- **[training/self_improve.ipynb](training/self_improve.ipynb)** — training notebook, runs on a T4

---

*Built for the OpenEnv Hackathon — Theme 4: Self-Improvement.*
