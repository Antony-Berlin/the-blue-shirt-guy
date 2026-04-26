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

<div class="tenor-gif-embed" data-postid="15820930" data-share-method="host" data-aspect-ratio="2.40601" data-width="100%"><a href="https://tenor.com/view/glasses-virtual-reality-cant-believe-confused-what-gif-15820930">Glasses Virtual Reality GIF</a>from <a href="https://tenor.com/search/glasses-gifs">Glasses GIFs</a></div> <script type="text/javascript" async src="https://tenor.com/embed.js"></script>

You know that movie *Free Guy* — where an NPC wakes up, realises he's inside a game, and starts improving himself beyond what he was programmed to do?

That's the one-line pitch. Except here the NPC is a coding agent, and instead of learning to fight, it learns to fix its own broken tools.

---

## The Problem

Here's what bothered me about how agents work today.

You give an LLM agent a set of tools and tell it to solve problems. It does. But when a tool is bad — a search that barely searches, an error explainer that explains nothing, a linter that buries you in noise — the agent doesn't complain. It doesn't push back. It just quietly works around the limitation, day after day, cycle after cycle.

It adapts to the broken thing instead of fixing it.

That felt wrong to me. The ceiling is fixed not because the model is incapable — but because the environment it operates in never gets better. I wanted to see what happens when it does.

---

## There's a reason we called it Genesis

I wanted a name that meant *this is where something new begins*. Not just a model improving — a whole environment evolving.

So here's what Genesis is: an environment where a coding agent solves programming tasks using five tools. Every time the agent uses a tool, that call gets graded in real time. The grades accumulate into a performance score per tool, tracked with an exponential moving average so we're always watching the trend, not just the last result.

When a tool's score starts sliding — consistently, not just once — a second LLM called the **Tool Architect** gets called in. It reads the source code of the failing tool, reads the grades, reads the feedback, and rewrites the tool from scratch.

Then we measure whether things actually got better. If the agent scores higher with the new tool, we keep it. If not, we revert to the backup and wait for the next cycle to try again.

What makes this more than a simple rewrite loop is what happens next. The Tool Architect itself is trained with GRPO — and its reward isn't a human label or a vibe check. Its reward is the actual improvement delta its rewrite produced. It gets smarter at rewriting tools by watching what worked and what didn't, cycle after cycle.

And in parallel, the agent's own LLM is fine-tuned on benchmark tasks using GRPO too. So every cycle, two things are improving at once: the model weights and the tool files.

That's why it's called Genesis. It's not one thing getting better. It's everything getting better together.

---

## The Loop

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

## The Tools

Five plain Python files. Intentionally imperfect. The whole point is that the system has to find the weak ones on its own — nobody tells it which tool is broken or what to fix. It has to figure that out from the grades.

---

## The Reward

```
reward = tests_passed × 0.6 + tool_quality × 0.2 + reasoning_quality × 0.2
```

Tool quality isn't a single fixed score — it's dynamic. The system first detects what kind of result a tool returned: code, test output, error explanation, search results, documentation. Then it picks the graders that are actually relevant to that output type, and the final tool score is a weighted average of those graders. A search tool gets judged on relevance and coverage. A code tool gets judged on correctness and structure. The grading adapts to the tool, not the other way around. New tools added during evolution get graded automatically, no code changes required.

Reasoning is scored by an LLM judge, with Anthropic as fallback, then a heuristic if both fail.

---

## Why This Matters

Most self-improvement research focuses on the model. Better weights. Better prompts. Better fine-tuning recipes.

This project focuses on something different: the environment the model operates in. The tools are part of the agent's capability. They're not sacred. They're just files. And if they're files, they can be rewritten.

The result is a system where the model and its runtime environment improve together — each cycle grounded in actual task performance, not assumptions about what should work.

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
