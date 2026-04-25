"""combined_loop.py — GRPO fine-tuning + self-improvement in one unified loop.

Each cycle:
  1. GRPO  — fine-tune the LLM on batch_size tasks (improves model weights)
  2. Improve — evaluate → Tool Architect rewrites worst tool → evaluate → revert if worse

The two phases are complementary: GRPO improves *what the model knows*,
self-improvement improves *what tools are available to it*.

Usage:
    python training/combined_loop.py                          # 3 cycles, defaults
    python training/combined_loop.py --cycles 5 --n 3 --batch-size 4
    python training/combined_loop.py --dry-run                # no file/model writes
    python training/combined_loop.py --grpo-only --cycles 2   # skip tool improvement
    python training/combined_loop.py --improve-only --cycles 2 # skip GRPO
    python training/combined_loop.py --status                 # print history and exit

State is persisted to training/combined_state.json.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

_STATE_FILE = Path(__file__).parent / "combined_state.json"


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _load_state() -> Dict:
    if _STATE_FILE.exists():
        return json.loads(_STATE_FILE.read_text())
    return {"cycle": 0, "history": []}


def _save_state(state: Dict) -> None:
    _STATE_FILE.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _print_history(history: List[Dict]) -> None:
    print(
        "\n  Cycle | GRPO Loss | GRPO Rew | Before  | After   | Delta   | Rev | Action",
        flush=True,
    )
    print("  ------|-----------|----------|---------|---------|---------|-----|------------------", flush=True)
    for h in history:
        rev = "yes" if h.get("reverted") else "no"
        grpo_loss = f"{h['grpo_mean_loss']:.3f}" if h.get("grpo_mean_loss") is not None else "  --  "
        grpo_rew  = f"{h['grpo_mean_reward']:.3f}" if h.get("grpo_mean_reward") is not None else "  --  "
        before    = f"{h['before_reward']:.3f}" if h.get("before_reward") is not None else "  --  "
        after     = f"{h['after_reward']:.3f}" if h.get("after_reward") is not None else "  --  "
        delta     = f"{h['delta']:+.3f}" if h.get("delta") is not None else "  --  "
        action    = f"{h.get('architect_action','--')}:{h.get('target_tool','?')}"
        print(
            f"  {h['cycle']:5d} | {grpo_loss:>9} | {grpo_rew:>8} | "
            f"{before:>7} | {after:>7} | {delta:>7} | {rev:>3} | {action}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Task pool helper
# ---------------------------------------------------------------------------

def _load_tasks() -> List[dict]:
    benchmark_path = Path(__file__).parent.parent / "tasks" / "benchmark.json"
    if benchmark_path.exists():
        return json.loads(benchmark_path.read_text())
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


def _pick_batch(tasks: List[dict], cycle_num: int, batch_size: int) -> List[dict]:
    """Rotating window — each cycle gets a different slice of the task pool."""
    n = len(tasks)
    start = ((cycle_num - 1) * batch_size) % n
    indices = [(start + i) % n for i in range(batch_size)]
    return [tasks[i] for i in indices]


# ---------------------------------------------------------------------------
# Main combined loop
# ---------------------------------------------------------------------------

def run_combined_loop(
    n_episodes: int,
    n_cycles: int,
    batch_size: int,
    dry_run: bool,
    grpo_only: bool = False,
    improve_only: bool = False,
    grpo_cfg=None,
) -> None:
    from training.self_improve import _one_cycle

    state      = _load_state()
    start_cycle = state["cycle"]
    all_tasks  = _load_tasks()

    # Lazy-init GRPO trainer only if needed (requires torch + transformers)
    trainer = None
    if not improve_only:
        from training.grpo_trainer import GRPOTrainer, GRPOConfig
        cfg = grpo_cfg or GRPOConfig()
        print(f"[COMBINED] Initialising GRPOTrainer (model={cfg.model_name})...", flush=True)
        trainer = GRPOTrainer(cfg)

    print(f"\n{'='*70}", flush=True)
    print(f"[COMBINED] Starting combined loop", flush=True)
    print(
        f"[COMBINED] cycles={n_cycles}  n_episodes={n_episodes}  "
        f"batch_size={batch_size}  dry_run={dry_run}",
        flush=True,
    )
    print(f"[COMBINED] grpo_only={grpo_only}  improve_only={improve_only}", flush=True)
    print(f"[COMBINED] Resuming from cycle {start_cycle}", flush=True)
    print(f"{'='*70}\n", flush=True)

    for cycle_idx in range(n_cycles):
        cycle_num = start_cycle + cycle_idx + 1
        print(f"\n{'─'*60}", flush=True)
        print(f"[COMBINED] Cycle {cycle_num}", flush=True)

        grpo_metrics: Dict = {"grpo_mean_loss": None, "grpo_mean_reward": None}
        improve_metrics: Dict = {
            "before_reward": None, "after_reward": None, "delta": None,
            "architect_action": None, "target_tool": None,
            "file_written": None, "reverted": False,
        }

        # ── Phase 1: GRPO ─────────────────────────────────────────────────
        if not improve_only and trainer is not None:
            batch = _pick_batch(all_tasks, cycle_num, batch_size)
            print(f"[COMBINED] GRPO phase — batch: {[t['id'] for t in batch]}", flush=True)
            if not dry_run:
                result = trainer.train_batch(batch)
                grpo_metrics["grpo_mean_loss"]   = result["mean_loss"]
                grpo_metrics["grpo_mean_reward"]  = result["mean_reward"]
            else:
                print("[COMBINED] (dry_run) skipping GRPO weight update", flush=True)
                grpo_metrics["grpo_mean_loss"]   = 0.0
                grpo_metrics["grpo_mean_reward"]  = 0.0

        # ── Phase 2: Self-Improvement ──────────────────────────────────────
        if not grpo_only:
            print(f"[COMBINED] Self-improve phase — cycle {cycle_num}", flush=True)
            cycle_record = _one_cycle(cycle_num, n_episodes, dry_run)
            improve_metrics = {
                "before_reward":    cycle_record["before_reward"],
                "after_reward":     cycle_record["after_reward"],
                "delta":            cycle_record["delta"],
                "architect_action": cycle_record["architect_action"],
                "target_tool":      cycle_record.get("target_tool"),
                "file_written":     cycle_record.get("file_written"),
                "reverted":         cycle_record["reverted"],
            }

        # ── Record ────────────────────────────────────────────────────────
        record = {
            "cycle": cycle_num,
            "dry_run": dry_run,
            **grpo_metrics,
            **improve_metrics,
        }
        state["cycle"] = cycle_num
        state["history"].append(record)
        _save_state(state)

        _print_history(state["history"][-5:])

    print(f"\n[COMBINED] All {n_cycles} cycles complete.", flush=True)

    # Save final GRPO checkpoint
    if trainer is not None and not dry_run:
        trainer._save("combined_final")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Combined GRPO + self-improvement loop")
    parser.add_argument("--n",            type=int,   default=3,    help="Episodes per self-improve eval")
    parser.add_argument("--cycles",       type=int,   default=3,    help="Number of combined cycles")
    parser.add_argument("--batch-size",   type=int,   default=4,    dest="batch_size", help="Tasks per GRPO batch")
    parser.add_argument("--dry-run",      action="store_true",       help="No file/model writes")
    parser.add_argument("--grpo-only",    action="store_true",       dest="grpo_only",    help="Skip self-improve phase")
    parser.add_argument("--improve-only", action="store_true",       dest="improve_only", help="Skip GRPO phase")
    parser.add_argument("--reset",        action="store_true",       help="Reset combined state before starting")
    parser.add_argument("--status",       action="store_true",       help="Print history and exit")
    # GRPO model override
    parser.add_argument("--model",        default="",               help="Override MODEL_NAME for GRPO")
    parser.add_argument("--lr",           type=float, default=5e-6)
    parser.add_argument("--group-size",   type=int,   default=4,    dest="group_size")
    args = parser.parse_args()

    if args.reset and _STATE_FILE.exists():
        _STATE_FILE.unlink()
        print("[COMBINED] State reset.", flush=True)

    if args.status:
        state = _load_state()
        print(f"Current cycle: {state['cycle']}")
        if state["history"]:
            _print_history(state["history"])
        else:
            print("No history yet.")
        return

    grpo_cfg = None
    if not args.improve_only:
        from training.grpo_trainer import GRPOConfig
        grpo_cfg = GRPOConfig(
            lr=args.lr,
            group_size=args.group_size,
        )
        if args.model:
            grpo_cfg.model_name = args.model

    run_combined_loop(
        n_episodes=args.n,
        n_cycles=args.cycles,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        grpo_only=args.grpo_only,
        improve_only=args.improve_only,
        grpo_cfg=grpo_cfg,
    )


if __name__ == "__main__":
    main()
