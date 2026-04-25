---
title: Genesis Environment
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Genesis Environment

A self-improving coding agent environment deployed as an OpenEnv-compatible server.

## What it does

The server holds a benchmark of Python programming tasks and hidden test suites.
Agents POST to `/reset` to receive a task, then POST to `/step` with their code
solution and tool usage log to receive a dual reward signal:

```
reward = pass_score × 0.6 + tool_usage_score × 0.2 + reasoning_score × 0.2
```

Per-tool-call grades are returned in the observation so the agent's self-improvement
loop can identify which tools to rewrite via the Tool Architect.

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new episode — returns task description + starter code |
| `/step`  | POST | Submit code + tool log — returns reward, test results, tool grades |
| `/state` | GET  | Current episode metadata and tool weight snapshot |
| `/health`| GET  | Health check |

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | No | HuggingFace token for the LLM-as-judge rubric scorer |
| `RUBRIC_MODEL` | No | Judge model (default: `Qwen/Qwen2.5-Coder-7B-Instruct`) |
| `API_BASE_URL` | No | OpenAI-compatible endpoint for the judge (default: HF router) |
| `ANTHROPIC_API_KEY` | No | Fallback judge via Claude if HF judge fails |
| `MAX_STEPS_PER_EPISODE` | No | Max tool-call steps per episode (default: 10) |

## Local development

```bash
git clone https://huggingface.co/spaces/berlin1906/genesis_env
cd genesis_env
pip install -r requirements.txt
cp .env.example .env  # fill in your keys
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Docker

```bash
docker build -t genesis-env .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e RUBRIC_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct \
  genesis-env
```
