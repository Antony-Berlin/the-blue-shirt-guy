"""grpo_trainer.py — GRPO fine-tuning for the Genesis coding agent.

Group Relative Policy Optimization (GRPO) fine-tunes the LLM that drives the
agent's tool-use loop.  For each task prompt we sample G completions, score
each with the Genesis environment reward, then update using the GRPO objective:

    L_GRPO = -E[ (r_i - mean(r)) / std(r) * log π(y_i|x) ]

Memory strategy for 14-16 GiB GPUs (Colab T4):
  - QLoRA: 4-bit quantised base + LoRA adapters (requires bitsandbytes)
    Reduces 7B from ~14 GiB to ~5 GiB.
  - NO separate reference model — frozen base weights under LoRA serve as
    the reference. disable_adapter_layers() / enable_adapter_layers() toggle
    between ref and policy in the same forward pass — zero extra VRAM.
  - Completions sampled ONE AT A TIME — no G× VRAM spike from batched generate.
  - Gradients accumulated one completion at a time — backward() inside loop,
    activations freed immediately.
  - Gradient checkpointing enabled — trades compute for activation memory.
  - torch.cuda.empty_cache() after every task.

Usage:
    python training/grpo_trainer.py
    python training/grpo_trainer.py --model Qwen/Qwen2.5-Coder-7B-Instruct \\
        --epochs 3 --group-size 4 --lr 1e-5

Requirements:
    pip install transformers>=4.40 peft>=0.10 accelerate bitsandbytes

Environment variables (.env):
    HF_TOKEN    — HuggingFace token
    MODEL_NAME  — base model to fine-tune (overridden by --model flag)
"""

import argparse
import os
import sys
import json
import random
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------

def _require(pkg: str):
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError as e:
        raise SystemExit(
            f"\nMissing dependency: {pkg}\n"
            f"Install: pip install transformers peft accelerate bitsandbytes\n"
            f"Error: {e}"
        )


def _qlora_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GRPOConfig:
    # Model
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-7B-Instruct"))
    output_dir: str = "checkpoints/grpo"
    push_to_hub: bool = False
    hub_repo: str = ""

    # Training
    epochs: int = 2
    group_size: int = 4       # G — completions sampled per prompt
    lr: float = 5e-6
    max_grad_norm: float = 1.0
    save_steps: int = 50

    # Sampling — kept small to fit 14 GiB GPU
    temperature: float = 0.9
    max_new_tokens: int = 256  # enough for a JSON submit action
    max_prompt_tokens: int = 768

    # QLoRA — auto-disabled if bitsandbytes not installed
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_4bit: bool = True  # QLoRA; falls back to bf16 if bitsandbytes missing

    # GRPO
    kl_coef: float = 0.04
    reward_clip: float = 5.0

    # Evaluation
    eval_episodes: int = 3
    eval_seeds_start: int = 9000


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""
    You are a Python coding assistant solving programming challenges.
    You have access to these tools:
      - search_code_examples(query)
      - run_tests(code, test_cases)
      - lint_code(code)
      - fetch_docs(library, symbol="")
      - explain_error(traceback_text, code="")

    Strategy:
    1. Search for related examples first
    2. Write your solution
    3. Run tests to verify
    4. Fix errors using explain_error and lint_code
    5. Submit your final code

    To call a tool respond ONLY with a JSON object where "action" is the tool name:
    {"action": "search_code_examples", "query": "..."}
    {"action": "run_tests", "code": "...", "test_cases": "assert foo(1) == 2"}
    {"action": "lint_code", "code": "..."}
    {"action": "fetch_docs", "library": "collections", "symbol": "Counter"}
    {"action": "explain_error", "traceback_text": "...", "code": "..."}

    To submit your final answer:
    {"action": "submit", "code": "...your complete Python solution..."}

    Respond ONLY with a single JSON object. No prose, no markdown fences.
""").strip()


def build_prompt(task_description: str, starter_code: str) -> str:
    return (
        f"TASK:\n{task_description}\n\n"
        f"STARTER CODE:\n```python\n{starter_code}\n```\n\n"
        "Use tools to research and test your solution, then submit."
    )


def messages_to_text(tokenizer, task_description: str, starter_code: str) -> str:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": build_prompt(task_description, starter_code)},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<|system|>{_SYSTEM_PROMPT}\n<|user|>{build_prompt(task_description, starter_code)}\n<|assistant|>"


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def score_completion(completion_text: str, task: dict) -> float:
    """Score a single model completion against the Genesis environment.

    Extraction priority:
      1. {"action": "submit", "code": "..."}  — explicit submit JSON
      2. ```python ... ```                     — fenced code block
      3. Raw def/class/import lines            — model wrote code directly
      4. starter_code fallback                 — 0 reward
    """
    import re, json as _json
    from envs.gen_env.server.gen_env_environment import GenesisEnvironment
    from envs.gen_env.models import GenEnvAction

    code = task.get("starter_code", "")
    text = completion_text.strip()

    # 1. JSON with "code" key (submit action or partial)
    try:
        cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            data = _json.loads(m.group())
            if "code" in data:
                code = data["code"]
    except Exception:
        pass

    # 2. Fenced python block
    if code == task.get("starter_code", ""):
        m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
        if m:
            code = m.group(1).strip()

    # 3. Raw Python heuristic
    if code == task.get("starter_code", ""):
        lines = [l for l in text.splitlines() if l.strip()]
        if lines and re.match(r"(def |class |import |from )", lines[0]):
            code = text.strip()

    env = GenesisEnvironment()
    env._current_task = task
    env._episode_id   = "grpo_train"
    env._step_count   = 0
    env._tool_log     = []
    obs = env.step(GenEnvAction(code=code, task_id=task["id"], tool_usage_log=[]))
    return float(obs.reward or 0.0)


# ---------------------------------------------------------------------------
# GRPO loss — memory-efficient gradient accumulation
# ---------------------------------------------------------------------------

def grpo_loss_and_backward(
    model,
    ref_model,          # unused — kept for API compat; pass None
    tokenizer,
    prompt_text: str,
    completions: List[str],
    rewards: List[float],
    cfg: GRPOConfig,
    use_lora: bool = True,
) -> float:
    """Compute per-completion GRPO loss and call backward() immediately.

    Reference log-probs are computed by temporarily disabling LoRA adapters
    so the frozen base weights act as the reference — zero extra VRAM.
    Processes one completion at a time for memory efficiency.
    """
    import torch

    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    if rewards_t.std() < 1e-8:
        return 0.0  # all identical rewards — no gradient signal

    advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
    advantages = advantages.clamp(-cfg.reward_clip, cfg.reward_clip)

    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    prompt_len = prompt_ids.shape[1] - 1

    scalar_losses = []

    for completion, adv in zip(completions, advantages):
        enc = tokenizer(
            prompt_text + completion,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_prompt_tokens + cfg.max_new_tokens,
        )
        input_ids = enc["input_ids"].to(model.device)
        labels    = input_ids[:, 1:]
        comp_ids  = labels[:, prompt_len:]

        # Reference log-probs: disable adapters so frozen base weights are used
        with torch.no_grad():
            if use_lora:
                model.disable_adapter_layers()
            ref_logits = model(input_ids).logits[:, :-1, :]
            ref_lp_all = torch.nn.functional.log_softmax(ref_logits, dim=-1)
            ref_lp = ref_lp_all[:, prompt_len:, :].gather(
                2, comp_ids.unsqueeze(-1)
            ).squeeze(-1).sum()
            del ref_logits, ref_lp_all
            if use_lora:
                model.enable_adapter_layers()

        # Policy log-probs (with grad, adapters enabled)
        lp_all = torch.nn.functional.log_softmax(
            model(input_ids).logits[:, :-1, :], dim=-1
        )
        lp = lp_all[:, prompt_len:, :].gather(
            2, comp_ids.unsqueeze(-1)
        ).squeeze(-1).sum()
        del lp_all

        kl   = lp - ref_lp.detach()
        loss = -(adv.to(model.device) * lp - cfg.kl_coef * kl) / len(completions)

        loss.backward()
        scalar_losses.append(loss.item())
        del input_ids, labels, comp_ids, lp, ref_lp, kl, loss

    return sum(scalar_losses)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class GRPOTrainer:
    def __init__(self, cfg: GRPOConfig):
        self.cfg = cfg
        self._setup()

    def _setup(self):
        _require("torch")
        _require("transformers")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        hf_token  = os.getenv("HF_TOKEN", "")
        qlora_ok  = _qlora_available()
        use_4bit  = self.cfg.load_in_4bit and self.cfg.use_lora

        if use_4bit and not qlora_ok:
            print(
                "[GRPO] WARNING: bitsandbytes not installed — falling back to bf16 LoRA.\n"
                "         pip install bitsandbytes  (Linux/CUDA only)",
                flush=True,
            )
            self.cfg.load_in_4bit = False
            use_4bit = False

        print(f"[GRPO] Loading tokenizer: {self.cfg.model_name}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name, token=hf_token, trust_remote_code=True,
        )
        # Add a real pad token so attention_mask is unambiguous
        if self.tokenizer.pad_token_id is None or \
                self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        bnb_cfg = None
        if use_4bit:
            from transformers import BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        mode = "QLoRA 4-bit" if use_4bit else ("LoRA bf16" if self.cfg.use_lora else "full bf16")
        print(f"[GRPO] Loading policy model ({mode}): {self.cfg.model_name}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            token=hf_token,
            trust_remote_code=True,
            quantization_config=bnb_cfg,
            torch_dtype=None if use_4bit else torch.bfloat16,
            device_map="auto",
        )
        model.resize_token_embeddings(len(self.tokenizer))

        if self.cfg.use_lora:
            _require("peft")
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            if use_4bit:
                model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=True
                )
            else:
                model.gradient_checkpointing_enable()
            model = get_peft_model(model, LoraConfig(
                r=self.cfg.lora_r,
                lora_alpha=self.cfg.lora_alpha,
                lora_dropout=self.cfg.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            ))
            model.print_trainable_parameters()
        else:
            model.gradient_checkpointing_enable()

        self.model    = model
        # No separate ref model — base weights under LoRA serve as the reference.
        # grpo_loss_and_backward toggles adapters off/on around the ref forward pass.
        self.ref_model = None
        self._use_lora = self.cfg.use_lora

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.lr,
        )
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"[GRPO] Ready. Output: {self.cfg.output_dir}", flush=True)

    # ── helpers ────────────────────────────────────────────────────────────

    def _load_tasks(self) -> List[dict]:
        bp = Path(__file__).parent.parent / "tasks" / "benchmark.json"
        if bp.exists():
            tasks = json.loads(bp.read_text())
            print(f"[GRPO] {len(tasks)} tasks loaded")
            return tasks
        return [{"id": "fallback_001",
                 "description": "Write most_frequent(lst).",
                 "starter_code": "def most_frequent(lst):\n    pass\n",
                 "difficulty": "easy",
                 "tests": ["assert most_frequent([1,2,2,3])==2"]}]

    def _sample_one(self, prompt_text: str) -> str:
        """Generate a single completion — called G times to keep VRAM flat."""
        import torch
        enc = self.tokenizer(
            prompt_text, return_tensors="pt",
            truncation=True, max_length=self.cfg.max_prompt_tokens, padding=False,
        )
        ids  = enc["input_ids"].to(self.model.device)
        mask = enc["attention_mask"].to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                ids, attention_mask=mask,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    def _sample_completions(self, prompt_text: str) -> List[str]:
        return [self._sample_one(prompt_text) for _ in range(self.cfg.group_size)]

    def _score_completions(self, completions: List[str], task: dict) -> List[float]:
        out = []
        for c in completions:
            try:
                out.append(score_completion(c, task))
            except Exception as e:
                print(f"[GRPO] scoring error: {e}", flush=True)
                out.append(0.0)
        return out

    def _flush(self):
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _step(self, task: dict) -> tuple:
        """One GRPO step on a single task. Returns (loss, mean_reward)."""
        prompt      = messages_to_text(self.tokenizer, task["description"], task.get("starter_code", ""))
        completions = self._sample_completions(prompt)
        rewards     = self._score_completions(completions, task)
        mean_r      = sum(rewards) / len(rewards)

        self.optimizer.zero_grad()
        loss_val = grpo_loss_and_backward(
            self.model, None, self.tokenizer,
            prompt, completions, rewards, self.cfg,
            use_lora=self._use_lora,
        )
        if loss_val != 0.0:
            import torch
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                self.cfg.max_grad_norm,
            )
            self.optimizer.step()
        self._flush()
        return loss_val, mean_r, rewards

    # ── public API ─────────────────────────────────────────────────────────

    def train_batch(self, tasks: List[dict]) -> dict:
        """Fine-tune on the given task list for one pass.

        Returns {'mean_loss', 'mean_reward', 'n_tasks'}.
        """
        if not tasks:
            return {"mean_loss": 0.0, "mean_reward": 0.0, "n_tasks": 0}

        losses, rewards = [], []
        for task in tasks:
            loss_val, mean_r, task_rewards = self._step(task)
            losses.append(loss_val)
            rewards.append(mean_r)
            print(
                f"[GRPO-BATCH] task={task['id']} loss={loss_val:.4f} "
                f"mean_reward={mean_r:.4f} rewards={[f'{r:.2f}' for r in task_rewards]}",
                flush=True,
            )

        ml = sum(losses) / len(losses)
        mr = sum(rewards) / len(rewards)
        print(f"[GRPO-BATCH] done — mean_loss={ml:.4f} mean_reward={mr:.4f}", flush=True)
        return {"mean_loss": ml, "mean_reward": mr, "n_tasks": len(tasks)}

    def train(self):
        tasks = self._load_tasks()
        step  = 0
        print(f"\n[GRPO] epochs={self.cfg.epochs} tasks={len(tasks)} G={self.cfg.group_size}", flush=True)

        for epoch in range(self.cfg.epochs):
            random.shuffle(tasks)
            ep_losses, ep_rewards = [], []
            for idx, task in enumerate(tasks):
                loss_val, mean_r, _ = self._step(task)
                ep_losses.append(loss_val)
                ep_rewards.append(mean_r)
                step += 1
                if step % 10 == 0 or idx == 0:
                    print(
                        f"[GRPO] e={epoch+1} step={step} task={task['id']} "
                        f"loss={loss_val:.4f} reward={mean_r:.4f}",
                        flush=True,
                    )
                if step % self.cfg.save_steps == 0:
                    self._save(f"step_{step}")

            ml = sum(ep_losses)  / len(ep_losses)  if ep_losses  else 0.0
            mr = sum(ep_rewards) / len(ep_rewards) if ep_rewards else 0.0
            print(f"\n[GRPO] Epoch {epoch+1} — loss={ml:.4f} reward={mr:.4f}\n", flush=True)
            self._save(f"epoch_{epoch+1}")

        print("[GRPO] Training complete.", flush=True)
        self._save("final")

    def _save(self, tag: str):
        out = Path(self.cfg.output_dir) / tag
        self.model.save_pretrained(str(out))
        self.tokenizer.save_pretrained(str(out))
        print(f"[GRPO] Saved: {out}", flush=True)
        if self.cfg.push_to_hub and self.cfg.hub_repo:
            try:
                self.model.push_to_hub(self.cfg.hub_repo, commit_message=f"grpo {tag}")
            except Exception as e:
                print(f"[GRPO] hub push failed: {e}", flush=True)

    def evaluate(self) -> float:
        from training.self_improve import evaluate as run_eval
        seeds  = list(range(self.cfg.eval_seeds_start, self.cfg.eval_seeds_start + self.cfg.eval_episodes))
        result = run_eval(self.cfg.eval_episodes, seeds=seeds)
        print(f"[GRPO] eval mean_reward={result['mean_reward']:.4f}", flush=True)
        return result["mean_reward"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GRPO fine-tuning — Genesis coding agent")
    parser.add_argument("--model",       default="")
    parser.add_argument("--epochs",      type=int,   default=2)
    parser.add_argument("--group-size",  type=int,   default=4,   dest="group_size")
    parser.add_argument("--lr",          type=float, default=5e-6)
    parser.add_argument("--output-dir",  default="checkpoints/grpo", dest="output_dir")
    parser.add_argument("--no-lora",     action="store_true", dest="no_lora")
    parser.add_argument("--no-4bit",     action="store_true", dest="no_4bit")
    parser.add_argument("--kl-coef",     type=float, default=0.04, dest="kl_coef")
    parser.add_argument("--eval-only",   action="store_true", dest="eval_only")
    parser.add_argument("--push-to-hub", default="", dest="push_to_hub")
    args = parser.parse_args()

    cfg = GRPOConfig(
        epochs=args.epochs,
        group_size=args.group_size,
        lr=args.lr,
        output_dir=args.output_dir,
        use_lora=not args.no_lora,
        load_in_4bit=not args.no_4bit,
        kl_coef=args.kl_coef,
        push_to_hub=bool(args.push_to_hub),
        hub_repo=args.push_to_hub,
    )
    if args.model:
        cfg.model_name = args.model

    trainer = GRPOTrainer(cfg)
    trainer.evaluate() if args.eval_only else trainer.train()


if __name__ == "__main__":
    main()
