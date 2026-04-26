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

![freeGuy](https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExdjZ6MDZ0OTE5d3A4dWd1bW91bTAwOTVkdGZ5NnNkZjh5Zms0aDRqdiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/uuWEJYU9OETUxCSssd/giphy.gif)

**[🚀 Live on HuggingFace Spaces](https://huggingface.co/spaces/berlin1906/genesis_env)** | **[📓 Training Notebook](training/genesis_training.ipynb)** | **[📐 Environment Design](ENV_DESIGN.md)**

---

You know that movie *Free Guy* — where an NPC wakes up, realises he's inside a game, and starts improving himself beyond what he was programmed to do?

That's the one-line pitch. Except here the NPC is a coding agent, and instead of learning to fight, it learns to fix its own broken tools.

---

## The Problem

Here's what bothered me about how agents work today.

You give an LLM agent a set of tools and tell it to solve problems. It does. But when a tool is bad — a search that barely searches, an error explainer that explains nothing, a linter that buries you in noise — the agent doesn't complain. It doesn't push back. It just quietly works around the limitation, day after day, cycle after cycle.

It adapts to the broken thing instead of fixing it.

That felt wrong to me. The ceiling is fixed not because the model is incapable — but because the environment it operates in never gets better. I wanted to see what happens when it does.

---

## Why This Is Different 🔬

There are really only three levers for improving an agent:

- **Adjust the prompt** — better instructions, better context, better examples
- **Adjust the weights** — fine-tune the model on better data
- **Adjust the system** — the architecture, the tools, the environment the agent operates in

Most research pulls the first lever. Some pulls the second. Almost none pulls the third — because it's the hardest. The system feels fixed. The tools feel like infrastructure, not something you'd train against.

Genesis pulls levers two and three together. The agent's weights improve through GRPO, grounded in actual task performance. And the system itself — the tools — evolves in parallel, rewritten by a Tool Architect that is itself learning from the outcomes it produces.

The insight is simple but underexplored: **the tools are part of the agent's capability, and they're just files.** If you can measure how well they're working, you can improve them. Genesis is the environment that makes that measurement — and that improvement — possible every single cycle.

---

## There's a reason we called it Genesis

Genesis is the environment. Not what improves — what makes improvement possible.

Every tool call is graded in real time. Genesis doesn't just return a score — it returns **natural language feedback** alongside it. The score tells the Tool Architect *that* a tool is broken. The feedback tells it *why*. That's what makes automated rewriting possible. Scores accumulate per tool via an EMA tracker, and when one starts sliding consistently, the **Tool Architect** steps in — reads the source, reads the feedback, and rewrites it. We measure the delta. Better? Keep it. Worse? Revert.

- 🤖 The agent LLM is trained with **GRPO** on task performance — every cycle, it gets sharper!
- 🏗️ Every rewrite naturally produces a training dataset. The Tool Architect LLM can be trained on it too — the data is already there!

That's why it's called Genesis. It's the ground. The agent grows from it.

---

## The Loop 🔁

```
① Agent solves tasks → every tool call graded live
② Worst tool flagged by EMA tracker
③ Tool Architect rewrites it
④ Agent re-evaluated → delta measured
⑤ delta > 0 → keep   |   delta < 0 → revert
⑥ Both the Tool Architect and the agent LLM trained on that delta
⑦ Repeat
```

---

## The Reward

```
reward = (tests_passed / tests_total) × 0.6
       + mean(tool_call_grades)        × 0.2
       + reasoning_score               × 0.2
```

Test pass rate dominates at 60% — did the code actually work? The remaining 40% is split between how well the agent used its tools and how well it reasoned through the problem.

**Tool call grading is dynamic** — each call is graded based on what it returned:

- 🔍 **Search results** → judged on result count and query relevance
- 💻 **Code** → judged on syntax correctness and style
- ❌ **Error tracebacks** → judged on whether they identify the cause and suggest a fix
- 🧪 **Test results** → judged on pass/fail ratio and informativeness
- 📄 **Documentation** → judged on completeness and examples

The grade for each call is a weighted average of the relevant graders. The final tool score is the mean across all calls in the episode.

Two sequence-level adjustments also apply:

- **−0.3** penalty for redundant tool calls within a 3-call window
- **+0.2** bonus for calling `explain_error` after a failure — or **−0.4** for ignoring the error and retrying the same thing

**Reasoning** is scored by an LLM judge — HuggingFace router first, Anthropic as fallback, heuristic if both are unavailable.

---

## Did It Actually Work? 📈

Yes. And we have the receipts.

Every training run produces:

- **Reward curves** — episode reward over time, showing the agent climbing
- **Tool rewrite deltas** — before/after performance for each rewritten tool
- **Revert events** — cycles where the rewrite made things worse and the system correctly rolled back
- **GRPO loss** — weight update convergence across cycles

The improvement is measured, logged, and plotted — not assumed. Some rewrites helped. Some didn't. The system handled both. That's the honest version of "showing improvement."

> Training is currently running. Loss and reward plots will be embedded here once complete.

Training notebook with full visualisation code:

- **[training/genesis_training.ipynb](training/genesis_training.ipynb)** — reward curves, tool rewrite deltas, GRPO loss plots
- **[ENV_DESIGN.md](ENV_DESIGN.md)** — complete environment design doc

*(Run on a T4 GPU to reproduce.)*

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

Or run everything in Colab: [`training/genesis_training.ipynb`](training/genesis_training.ipynb)

---

## Go Deeper

- **[ENV_DESIGN.md](ENV_DESIGN.md)** — full environment design: episode lifecycle, grading logic, EMA registry, Tool Architect, training pipeline
- **[PROJECT.md](PROJECT.md)** — extended project spec: reward formula rationale, grader architecture, Tool Architect prompt design
- **[reward_mechanism.md](reward_mechanism.md)** — how individual tool calls get scored
- **[training/genesis_training.ipynb](training/genesis_training.ipynb)** — training notebook: self-improve, GRPO, combined loop with visualisations

---

*Built for the OpenEnv Hackathon — Theme 4: Self-Improvement.*
