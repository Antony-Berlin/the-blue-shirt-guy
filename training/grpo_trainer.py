"""grpo_trainer.py — GRPO fine-tuning for the Genesis coding agent.

Group Relative Policy Optimization (GRPO) fine-tunes the LLM that drives the
agent's tool-use loop.  For each task prompt we sample G completions, score
each with the Genesis environment reward, then update using the GRPO objective:

    L_GRPO = -E[ (r_i - mean(r)) / std(r) * log π(y_i|x) ]

This avoids a separate critic model: the relative rank within a group is the
advantage estimate.

Usage:
    # Basic run (uses .env for model + API key)
    python training/grpo_trainer.py

    # Custom settings
    python training/grpo_trainer.py \\
        --model  Qwen/Qwen2.5-Coder-7B-Instruct \\
        --epochs 3 \\
        --group-size 4 \\
        --lr 1e-5 \\
        --output-dir checkpoints/grpo_v1

Requirements (install separately):
    pip install transformers>=4.40 peft>=0.10 trl>=0.8 accelerate bitsandbytes

Environment variables (from .env):
    HF_TOKEN       — HuggingFace token (model download + optional push)
    MODEL_NAME     — base model to fine-tune (overridden by --model flag)
    API_BASE_URL   — used for reward evaluation episodes
"""

import argparse
import os
import sys
import json
import math
import random
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

# ---------------------------------------------------------------------------
# Lazy imports — only pulled in after argument parsing so --help is instant
# ---------------------------------------------------------------------------

def _require(pkg: str):
    """Import a package, raising a clear error if missing."""
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError as e:
        raise SystemExit(
            f"\nMissing dependency: {pkg}\n"
            f"Install with: pip install transformers peft trl accelerate bitsandbytes\n"
            f"Original error: {e}"
        )


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
    batch_size: int = 1          # tasks per gradient step (memory-limited)
    group_size: int = 4          # completions sampled per prompt (G in GRPO)
    lr: float = 5e-6
    max_grad_norm: float = 1.0
    warmup_steps: int = 10
    save_steps: int = 50

    # Sampling
    temperature: float = 0.9
    max_new_tokens: int = 512
    max_prompt_tokens: int = 1024

    # LoRA / QLoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_4bit: bool = True    # QLoRA; set False for full-precision LoRA

    # GRPO
    kl_coef: float = 0.04        # KL penalty weight against reference policy
    gamma: float = 0.99          # discount for multi-turn reward (unused in single-turn)
    reward_clip: float = 5.0     # clip |advantage| to this value

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
    """Build a single text prompt suitable for the model's chat template."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": build_prompt(task_description, starter_code)},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Fallback: simple concatenation
    return f"<|system|>{_SYSTEM_PROMPT}\n<|user|>{build_prompt(task_description, starter_code)}\n<|assistant|>"


# ---------------------------------------------------------------------------
# Reward function — runs one full episode in the Genesis env
# ---------------------------------------------------------------------------

def score_completion(completion_text: str, task: dict) -> float:
    """Score a single model completion against the Genesis environment.

    The completion is expected to end with a submit action JSON.
    We extract the code from the submit action (or the whole text as fallback)
    and run env.step() to get the reward.
    """
    import re, json as _json

    from envs.gen_env.server.gen_env_environment import GenesisEnvironment
    from envs.gen_env.models import GenEnvAction, GenEnvObservation
    from agent.tool_executor import ToolExecutor

    # Extract submitted code from the completion
    code = task.get("starter_code", "")
    try:
        text = re.sub(r"```(?:json)?\s*", "", completion_text).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = _json.loads(match.group())
            if data.get("action") == "submit" and "code" in data:
                code = data["code"]
    except Exception:
        pass

    env = GenesisEnvironment()
    # Prime the env with this specific task
    env._current_task = task
    env._episode_id   = "grpo_train"
    env._step_count   = 0
    env._tool_log     = []

    action   = GenEnvAction(code=code, task_id=task["id"], tool_usage_log=[])
    step_obs = env.step(action)
    return float(step_obs.reward or 0.0)


# ---------------------------------------------------------------------------
# GRPO training step
# ---------------------------------------------------------------------------

def grpo_loss(
    model,
    ref_model,
    tokenizer,
    prompt_text: str,
    completions: List[str],
    rewards: List[float],
    cfg: GRPOConfig,
) -> "torch.Tensor":
    """Compute GRPO loss for one prompt with G completions."""
    import torch

    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    if rewards_t.std() < 1e-8:
        # All rewards identical — no gradient signal
        return torch.tensor(0.0, requires_grad=True)

    advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
    advantages = advantages.clamp(-cfg.reward_clip, cfg.reward_clip)

    total_loss = torch.tensor(0.0, requires_grad=True)

    for i, (completion, adv) in enumerate(zip(completions, advantages)):
        full_text = prompt_text + completion
        enc = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_prompt_tokens + cfg.max_new_tokens,
        )
        input_ids = enc["input_ids"].to(model.device)

        with torch.no_grad():
            ref_logits = ref_model(input_ids).logits

        logits = model(input_ids).logits

        # Shift: labels = input_ids[1:], logits = logits[:-1]
        shift_logits     = logits[:, :-1, :]
        shift_ref_logits = ref_logits[:, :-1, :]
        shift_labels     = input_ids[:, 1:]

        log_probs     = torch.nn.functional.log_softmax(shift_logits,     dim=-1)
        ref_log_probs = torch.nn.functional.log_softmax(shift_ref_logits, dim=-1)

        # Only consider completion tokens (not prompt tokens)
        prompt_enc = tokenizer(prompt_text, return_tensors="pt")
        prompt_len = prompt_enc["input_ids"].shape[1] - 1  # -1 for shift

        completion_log_probs     = log_probs[:, prompt_len:, :]
        completion_ref_log_probs = ref_log_probs[:, prompt_len:, :]
        completion_labels        = shift_labels[:, prompt_len:]

        # Gather log probs for the actual tokens
        token_log_probs     = completion_log_probs.gather(2, completion_labels.unsqueeze(-1)).squeeze(-1)
        token_ref_log_probs = completion_ref_log_probs.gather(2, completion_labels.unsqueeze(-1)).squeeze(-1)

        # Sequence log prob + KL penalty
        seq_log_prob = token_log_probs.sum()
        kl           = (token_log_probs - token_ref_log_probs).sum()

        loss_i = -(adv * seq_log_prob - cfg.kl_coef * kl)
        total_loss = total_loss + loss_i

    return total_loss / len(completions)


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class GRPOTrainer:
    def __init__(self, cfg: GRPOConfig):
        self.cfg = cfg
        self._setup()

    def _setup(self):
        torch     = _require("torch")
        transformers = _require("transformers")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        hf_token = os.getenv("HF_TOKEN", "")
        print(f"[GRPO] Loading tokenizer: {self.cfg.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name,
            token=hf_token,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_cfg = None
        if self.cfg.load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        print(f"[GRPO] Loading model: {self.cfg.model_name} (4bit={self.cfg.load_in_4bit})")
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            token=hf_token,
            trust_remote_code=True,
            quantization_config=bnb_cfg,
            device_map="auto",
        )

        if self.cfg.use_lora:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            if self.cfg.load_in_4bit:
                model = prepare_model_for_kbit_training(model)
            lora_cfg = LoraConfig(
                r=self.cfg.lora_r,
                lora_alpha=self.cfg.lora_alpha,
                lora_dropout=self.cfg.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()

        self.model = model
        # Frozen reference model for KL penalty
        print("[GRPO] Loading reference model (frozen)...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            token=hf_token,
            trust_remote_code=True,
            quantization_config=bnb_cfg,
            device_map="auto",
        )
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.lr,
        )

        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"[GRPO] Output dir: {self.cfg.output_dir}")

    def _load_tasks(self) -> List[dict]:
        benchmark_path = Path(__file__).parent.parent / "tasks" / "benchmark.json"
        if benchmark_path.exists():
            tasks = json.loads(benchmark_path.read_text())
            print(f"[GRPO] Loaded {len(tasks)} tasks from benchmark.json")
            return tasks
        # Fallback single task
        return [{
            "id": "fallback_001",
            "description": "Write a function most_frequent(lst) that returns the most frequently occurring element.",
            "starter_code": "def most_frequent(lst):\n    pass\n",
            "difficulty": "easy",
            "tests": [
                "assert most_frequent([1, 2, 2, 3]) == 2",
                "assert most_frequent(['a', 'b', 'a']) == 'a'",
            ],
        }]

    def _sample_completions(self, prompt_text: str) -> List[str]:
        """Sample G completions for one prompt."""
        import torch

        enc = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_prompt_tokens,
        )
        input_ids = enc["input_ids"].to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids.repeat(self.cfg.group_size, 1),
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the newly generated tokens
        completions = []
        for out in outputs:
            new_tokens = out[input_ids.shape[1]:]
            completions.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return completions

    def _score_completions(self, completions: List[str], task: dict) -> List[float]:
        rewards = []
        for c in completions:
            try:
                r = score_completion(c, task)
            except Exception as e:
                print(f"[GRPO] Scoring error: {e}")
                r = 0.0
            rewards.append(r)
        return rewards

    def train(self):
        import torch

        tasks    = self._load_tasks()
        n_tasks  = len(tasks)
        step     = 0

        print(f"\n[GRPO] Starting training: epochs={self.cfg.epochs} tasks={n_tasks} G={self.cfg.group_size}")

        for epoch in range(self.cfg.epochs):
            random.shuffle(tasks)
            epoch_losses = []
            epoch_rewards = []

            for task_idx, task in enumerate(tasks):
                prompt_text = messages_to_text(self.tokenizer, task["description"], task.get("starter_code", ""))

                # Sample G completions
                completions = self._sample_completions(prompt_text)
                rewards     = self._score_completions(completions, task)
                mean_r      = sum(rewards) / len(rewards)
                epoch_rewards.append(mean_r)

                # GRPO loss
                self.optimizer.zero_grad()
                loss = grpo_loss(
                    self.model, self.ref_model, self.tokenizer,
                    prompt_text, completions, rewards, self.cfg,
                )

                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.model.parameters()),
                        self.cfg.max_grad_norm,
                    )
                    self.optimizer.step()

                epoch_losses.append(loss.item())
                step += 1

                if step % 10 == 0 or task_idx == 0:
                    print(
                        f"[GRPO] epoch={epoch+1}/{self.cfg.epochs} "
                        f"step={step} task={task['id']} "
                        f"loss={loss.item():.4f} mean_reward={mean_r:.4f} "
                        f"rewards={[f'{r:.2f}' for r in rewards]}"
                    )

                if step % self.cfg.save_steps == 0:
                    self._save(f"step_{step}")

            mean_epoch_loss   = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            mean_epoch_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
            print(
                f"\n[GRPO] Epoch {epoch+1} complete — "
                f"mean_loss={mean_epoch_loss:.4f}  mean_reward={mean_epoch_reward:.4f}\n"
            )
            self._save(f"epoch_{epoch+1}")

        print("[GRPO] Training complete.")
        self._save("final")

    def _save(self, tag: str):
        out = Path(self.cfg.output_dir) / tag
        self.model.save_pretrained(str(out))
        self.tokenizer.save_pretrained(str(out))
        print(f"[GRPO] Saved checkpoint: {out}")

        if self.cfg.push_to_hub and self.cfg.hub_repo:
            try:
                self.model.push_to_hub(self.cfg.hub_repo, commit_message=f"grpo {tag}")
                print(f"[GRPO] Pushed to hub: {self.cfg.hub_repo}")
            except Exception as e:
                print(f"[GRPO] Hub push failed: {e}")

    def evaluate(self) -> float:
        """Run eval episodes using the Genesis env to measure current policy reward."""
        from training.self_improve import evaluate as run_eval

        seeds = list(range(self.cfg.eval_seeds_start, self.cfg.eval_seeds_start + self.cfg.eval_episodes))
        result = run_eval(self.cfg.eval_episodes, seeds=seeds)
        mean_r = result["mean_reward"]
        print(f"[GRPO] Eval mean_reward={mean_r:.4f} (n={self.cfg.eval_episodes})")
        return mean_r


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GRPO fine-tuning for the Genesis coding agent")
    parser.add_argument("--model",       default="",    help="Model name/path (overrides MODEL_NAME env var)")
    parser.add_argument("--epochs",      type=int,   default=2)
    parser.add_argument("--group-size",  type=int,   default=4,   dest="group_size", help="Completions per prompt (G)")
    parser.add_argument("--lr",          type=float, default=5e-6)
    parser.add_argument("--batch-size",  type=int,   default=1,   dest="batch_size")
    parser.add_argument("--output-dir",  default="checkpoints/grpo", dest="output_dir")
    parser.add_argument("--no-lora",     action="store_true", dest="no_lora")
    parser.add_argument("--no-4bit",     action="store_true", dest="no_4bit")
    parser.add_argument("--kl-coef",     type=float, default=0.04, dest="kl_coef")
    parser.add_argument("--eval-only",   action="store_true", dest="eval_only")
    parser.add_argument("--push-to-hub", default="", dest="push_to_hub", help="HF repo id to push checkpoints")
    args = parser.parse_args()

    cfg = GRPOConfig(
        epochs=args.epochs,
        group_size=args.group_size,
        lr=args.lr,
        batch_size=args.batch_size,
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

    if args.eval_only:
        trainer.evaluate()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
