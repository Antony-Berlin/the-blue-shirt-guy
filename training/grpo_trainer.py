"""grpo_trainer.py — GRPO fine-tuning using TRL's GRPOTrainer + QLoRA.

Delegates all GRPO mechanics to TRL 1.2.0. This file wires up:
  - QLoRA model loading (bitsandbytes 4-bit + LoRA via peft)
  - Reward function scoring completions via GenesisEnvironment
  - Dataset construction for TRL
  - train_batch() / train() entry points used by combined_loop.py

Requirements:
    pip install "trl>=1.2.0" "transformers>=4.40" "peft>=0.10" accelerate bitsandbytes datasets

Environment variables:
    HF_TOKEN    — HuggingFace token
    MODEL_NAME  — base model (default: Qwen/Qwen2.5-Coder-7B-Instruct)
"""

import os
import sys
import json
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GRPOConfig:
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-7B-Instruct"))
    output_dir: str = "checkpoints/grpo"

    # Training
    epochs: int = 1
    group_size: int = 4        # num_generations in TRL
    lr: float = 5e-6

    # Generation
    max_new_tokens: int = 256  # max_completion_length in TRL
    temperature: float = 0.9

    # QLoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_4bit: bool = True

    # Eval
    eval_episodes: int = 3
    eval_seeds_start: int = 9000


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""
    You are a Python coding assistant. Solve the task below.
    Write your complete solution inside a ```python ... ``` block.
    Then submit with: {"action": "submit", "code": "...your solution..."}
""").strip()


def build_prompt_text(task: dict, tokenizer) -> str:
    """Apply chat template to get a plain string prompt TRL can tokenize."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": (
            f"TASK:\n{task['description']}\n\n"
            f"STARTER:\n```python\n{task.get('starter_code', '')}\n```"
        )},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"<|system|>{_SYSTEM_PROMPT}\n<|user|>{messages[1]['content']}\n<|assistant|>"


# ---------------------------------------------------------------------------
# Code extraction from model completions
# ---------------------------------------------------------------------------

def _extract_code(text: str, starter: str) -> str:
    # 1. JSON submit action
    try:
        m = re.search(r"\{.*\}", re.sub(r"```(?:json)?\s*", "", text).strip(), re.DOTALL)
        if m:
            data = json.loads(m.group())
            if "code" in data:
                return data["code"]
    except Exception:
        pass
    # 2. Fenced python block
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # 3. Raw def/class/import
    lines = [l for l in text.splitlines() if l.strip()]
    if lines and re.match(r"(def |class |import |from )", lines[0]):
        return text.strip()
    return starter


# ---------------------------------------------------------------------------
# Reward function — TRL 1.2.0 calls reward_fn(prompts, completions, **dataset_cols)
# ---------------------------------------------------------------------------

def make_reward_fn(tasks_by_id: dict):
    from envs.gen_env.server.gen_env_environment import GenesisEnvironment
    from envs.gen_env.models import GenEnvAction

    def reward_fn(
        prompts: List[str],
        completions: List[str],
        task_id: List[str] = None,
        **kwargs,
    ) -> List[float]:
        rewards = []
        ids = task_id if task_id is not None else [""] * len(completions)
        for completion, tid in zip(completions, ids):
            task = tasks_by_id.get(tid)
            if task is None:
                rewards.append(0.0)
                continue
            code = _extract_code(completion, task.get("starter_code", ""))
            try:
                env = GenesisEnvironment()
                env._current_task = task
                env._episode_id   = "grpo_train"
                env._step_count   = 0
                env._tool_log     = []
                obs = env.step(GenEnvAction(code=code, task_id=tid, tool_usage_log=[]))
                rewards.append(float(obs.reward or 0.0))
            except Exception as e:
                print(f"[GRPO] reward error task={tid}: {e}", flush=True)
                rewards.append(0.0)
        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class GRPOTrainer:
    def __init__(self, cfg: GRPOConfig):
        self.cfg = cfg
        self._setup()

    def _setup(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        hf_token = os.getenv("HF_TOKEN", "")

        use_4bit = self.cfg.load_in_4bit and self.cfg.use_lora
        try:
            import bitsandbytes  # noqa
        except ImportError:
            if use_4bit:
                print("[GRPO] bitsandbytes not found — falling back to fp16 LoRA", flush=True)
                use_4bit = False
                self.cfg.load_in_4bit = False

        print(f"[GRPO] Loading tokenizer: {self.cfg.model_name}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name, token=hf_token, trust_remote_code=True,
        )
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
                bnb_4bit_compute_dtype=torch.float16,  # T4 supports fp16, not bf16
            )

        mode = "QLoRA 4-bit" if use_4bit else ("LoRA fp16" if self.cfg.use_lora else "fp16")
        print(f"[GRPO] Loading model ({mode}): {self.cfg.model_name}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            token=hf_token,
            trust_remote_code=True,
            quantization_config=bnb_cfg,
            torch_dtype=None if use_4bit else torch.float16,
            device_map="auto",
        )
        model.resize_token_embeddings(len(self.tokenizer))

        if self.cfg.use_lora:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            if use_4bit:
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
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
            # Cast all trainable (LoRA) params to float16 — T4 doesn't support bf16
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data = param.data.to(torch.float16)

        self.model = model
        print("[GRPO] Model ready.", flush=True)

    def _make_trl_trainer(self, tasks: List[dict]):
        from trl import GRPOTrainer as TRLGRPOTrainer, GRPOConfig as TRLGRPOConfig
        from datasets import Dataset

        tasks_by_id = {t["id"]: t for t in tasks}

        # Build plain-string prompts — TRL requires "prompt" column as strings
        rows = [
            {
                "prompt":  build_prompt_text(t, self.tokenizer),
                "task_id": t["id"],
            }
            for t in tasks
        ]
        dataset = Dataset.from_list(rows)

        # per_device_train_batch_size=1 (one prompt), steps_per_generation=num_generations
        # so generation_batch_size = 1 * num_generations = num_generations ✓ (divisible)
        trl_cfg = TRLGRPOConfig(
            output_dir=self.cfg.output_dir,
            num_train_epochs=self.cfg.epochs,
            num_generations=self.cfg.group_size,
            steps_per_generation=self.cfg.group_size,
            learning_rate=self.cfg.lr,
            max_completion_length=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=self.cfg.group_size,
            fp16=True,                  # T4 GPU — fp16 not bf16
            bf16=False,                 # explicitly disable bf16
            logging_steps=1,
            save_strategy="no",
            report_to="none",
        )

        trainer = TRLGRPOTrainer(
            model=self.model,
            reward_funcs=make_reward_fn(tasks_by_id),
            args=trl_cfg,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )
        return trainer

    def train_batch(self, tasks: List[dict]) -> dict:
        """Fine-tune on tasks for one pass. Returns {mean_loss, mean_reward, n_tasks}."""
        if not tasks:
            return {"mean_loss": 0.0, "mean_reward": 0.0, "n_tasks": 0}

        print(f"[GRPO] train_batch: {[t['id'] for t in tasks]}", flush=True)
        trainer = self._make_trl_trainer(tasks)
        result  = trainer.train()

        mean_loss   = getattr(result, "training_loss", 0.0) or 0.0
        mean_reward = 0.0
        if trainer.state.log_history:
            vals = [e["reward"] for e in trainer.state.log_history if "reward" in e]
            if vals:
                mean_reward = sum(vals) / len(vals)

        print(f"[GRPO] batch done — loss={mean_loss:.4f} mean_reward={mean_reward:.4f}", flush=True)
        return {"mean_loss": mean_loss, "mean_reward": mean_reward, "n_tasks": len(tasks)}

    def train(self):
        bp = Path(__file__).parent.parent / "tasks" / "benchmark.json"
        tasks = json.loads(bp.read_text()) if bp.exists() else [{
            "id": "fallback_001",
            "description": "Write most_frequent(lst) that returns the most frequent element.",
            "starter_code": "def most_frequent(lst):\n    pass\n",
            "tests": ["assert most_frequent([1,2,2,3])==2"],
        }]
        print(f"[GRPO] Full train: {len(tasks)} tasks, {self.cfg.epochs} epoch(s)", flush=True)
        result = self.train_batch(tasks)
        self._save("final")
        return result

    def _save(self, tag: str):
        out = Path(self.cfg.output_dir) / tag
        self.model.save_pretrained(str(out))
        self.tokenizer.save_pretrained(str(out))
        print(f"[GRPO] Saved: {out}", flush=True)

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
    import argparse
    parser = argparse.ArgumentParser(description="GRPO fine-tuning — Genesis coding agent")
    parser.add_argument("--model",      default="")
    parser.add_argument("--epochs",     type=int,   default=1)
    parser.add_argument("--group-size", type=int,   default=4,    dest="group_size")
    parser.add_argument("--lr",         type=float, default=5e-6)
    parser.add_argument("--output-dir", default="checkpoints/grpo", dest="output_dir")
    parser.add_argument("--no-lora",    action="store_true", dest="no_lora")
    parser.add_argument("--no-4bit",    action="store_true", dest="no_4bit")
    parser.add_argument("--eval-only",  action="store_true", dest="eval_only")
    args = parser.parse_args()

    cfg = GRPOConfig(
        epochs=args.epochs,
        group_size=args.group_size,
        lr=args.lr,
        output_dir=args.output_dir,
        use_lora=not args.no_lora,
        load_in_4bit=not args.no_4bit,
    )
    if args.model:
        cfg.model_name = args.model

    trainer = GRPOTrainer(cfg)
    trainer.evaluate() if args.eval_only else trainer.train()


if __name__ == "__main__":
    main()
