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

You know that movie — *Free Guy* — where an NPC in a video game suddenly becomes aware that he's an NPC? He starts making choices. Improving himself. Breaking out of the script he was written with.

That's the vibe here.

---

## We called it Genesis for a reason

Genesis is a coding agent that was born with five tools and told to go solve programming problems. The tools weren't great. One of them just dumps raw pydoc at you. Another explains errors using templates from 2004. The search tool has exactly 10 examples hardcoded into it.

A normal agent would just work around them. Ignore the bad ones. Develop habits. Get stuck at a ceiling.

Genesis doesn't do that.

Genesis looks at its own tools, figures out which ones are failing it, and **rewrites them**. Then it checks if the rewrite actually helped. If it didn't, it rolls back. If it did, it keeps the improvement — and learns to write better rewrites next time.

It gains consciousness the hard way. One tool at a time.

---

## The loop that makes it real

Every tool call gets graded. Not just "did the agent succeed" — each individual tool call, mid-episode, gets a score based on what it returned. Bad search results score low. Useful error explanations score high. The grades accumulate into a running average per tool.

When a tool keeps scoring badly, a second LLM — the **Tool Architect** — gets called in. It reads the grades. It reads the NL feedback from the judge. It reads the actual source code of the bad tool. Then it rewrites it.

```
run episodes → grade every tool call → worst tool flagged
→ Tool Architect rewrites it → run episodes again → better? keep it : revert
→ train Tool Architect on that delta → repeat
```

The Tool Architect gets trained with GRPO where the reward is literally "did the agent do better after your rewrite." It learns to be a good programmer by watching what it breaks and what it fixes.

Meanwhile the agent's own LLM gets fine-tuned on the benchmark tasks in parallel. Model weights improving. Tool files improving. Both at once, every cycle.

---

## What the tools start as

Five plain Python files. Rewritable. Evolvable.

| Tool | The honest description |
|------|----------------------|
| `search_code_examples` | BM25 over 10 hardcoded examples. Yes, 10. |
| `run_tests` | Runs assert statements in a subprocess. Actually fine. |
| `lint_code` | Calls pyflakes, re-ranks by frequency. Verbose. |
| `fetch_docs` | Raw pydoc dump, cuts off at 2000 chars. |
| `explain_error` | Pattern-matches the error type and returns a template. Rough. |

The bad ones are bad on purpose. The interesting question is whether the system finds them, understands why, and makes them better — without being told which ones to fix.

---

## The reward

```
reward = tests_passed × 0.6 + tool_quality × 0.2 + reasoning_quality × 0.2
```

Tool quality comes from a multi-grader system that routes each tool result to the right grader based on what it returned — code gets syntax-checked, test output gets parsed, error explanations get rated for usefulness. No manual labelling. The environment figures out what kind of result it's looking at.

Reasoning gets scored by an LLM judge. If that fails it falls back to Anthropic. If that fails it uses a heuristic. Robust by default.

---

## Why it matters

The standard approach to agents is: pick better tools, write better prompts, hope for the best.

This project asks a different question. What if the agent could be the one to improve its tools? What if the ceiling wasn't fixed?

It's a small step toward the Free Guy problem — an agent that doesn't just operate within its constraints but actively works to dissolve them.

---

## Try it

```bash
git clone https://huggingface.co/spaces/berlin1906/genesis_env
cd genesis_env
pip install -r requirements.txt
cp .env.example .env  # add HF_TOKEN and MODEL_NAME

python inference.py                                         # one episode
python training/self_improve.py --cycles 3 --n 3           # self-improvement loop
python training/combined_loop.py --cycles 3 --batch-size 4 # GRPO + self-improvement
```

Or open [`training/self_improve.ipynb`](training/self_improve.ipynb) in Colab and run it cell by cell.

---

## Go deeper

- **[PROJECT.md](PROJECT.md)** — full design: reward formula, grader architecture, tool weight tracking, EMA thresholds, the whole thing
- **[reward_mechanism.md](reward_mechanism.md)** — how each tool call gets its grade
- **[training/self_improve.ipynb](training/self_improve.ipynb)** — the full training notebook, works on a T4

---

*Built for the OpenEnv hackathon, Theme 4 — Self-Improvement.*
*The NPC learned to code. Then it learned to improve its own tools. We're watching to see what happens next.*
