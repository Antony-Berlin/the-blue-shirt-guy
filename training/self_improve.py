"""self_improve.py — End-to-end self-improvement loop.

Runs N evaluation episodes, feeds results to the ToolArchitect, which rewrites
or creates tools, then runs N more episodes to measure the reward delta.

Usage:
    python training/self_improve.py                        # 1 cycle, 3 eval episodes
    python training/self_improve.py --cycles 5 --n 5       # 5 cycles, 5 episodes each
    python training/self_improve.py --dry-run              # decide but don't write files
    python training/self_improve.py --once                 # one eval + one architect call

State is persisted to training/loop_state.json so cycles survive restarts.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from agent.tool_executor import ToolExecutor
from envs.gen_env.models import GenEnvAction
from envs.gen_env.client import GenesisEnvClient as GenesisEnvironment
from envs.gen_env.tool_registry import ToolRegistry, ToolFlag
from openai import OpenAI

_STATE_FILE = Path(__file__).parent / "loop_state.json"

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-7B-Instruct")


# ---------------------------------------------------------------------------
# Episode runner (re-uses inference logic without the print scaffolding)
# ---------------------------------------------------------------------------

def _run_episode(seed: int = None, client=None) -> Dict:
    """Run one full episode. Returns dict with reward, tool_grades, nl_feedback, tool_log."""
    from inference import run_tool_loop

    if client is None:
        client = OpenAI(base_url=API_BASE, api_key=API_KEY)
    executor = ToolExecutor()
    env = GenesisEnvironment()

    executor.reset_log()
    obs = env.reset(seed=seed)

    final_code, _ = run_tool_loop(
        client, executor,
        obs.task_description,
        obs.starter_code,
        env=env,
    )
    tool_log = executor.get_log()

    action = GenEnvAction(code=final_code, task_id=obs.task_id, tool_usage_log=tool_log)
    step_obs = env.step(action)

    return {
        "task_id": obs.task_id,
        "reward": step_obs.reward or 0.0,
        "tool_grades": step_obs.tool_grades,
        "nl_feedback": step_obs.nl_feedback,
        "tool_weights": step_obs.tool_weights,
        "tool_log": tool_log,
    }


def evaluate(n: int, seeds: List[int] = None, client=None) -> Dict:
    """Run n episodes and aggregate results.

    Args:
        n: number of episodes
        seeds: list of seeds
        client: OpenAI-compatible client (or LocalClient). If None, uses HF router.

    Returns:
        mean_reward, per_tool_weights (averaged), all_feedback, all_tool_logs
    """
    seeds = seeds or list(range(n))
    results = []
    print(f"[EVAL] Running {n} episodes...", flush=True)
    for i, seed in enumerate(seeds):
        try:
            r = _run_episode(seed=seed, client=client)
            results.append(r)
            print(f"[EVAL] episode {i+1}/{n} seed={seed} reward={r['reward']:.3f} task={r['task_id']}", flush=True)
        except Exception as exc:
            print(f"[EVAL] episode {i+1}/{n} FAILED: {exc}", flush=True)

    if not results:
        return {"mean_reward": 0.0, "tool_weights": {}, "nl_feedback": "", "tool_logs": []}

    mean_reward = sum(r["reward"] for r in results) / len(results)

    # Aggregate tool weights across episodes (last episode's weights are most recent EMA)
    tool_weights = results[-1]["tool_weights"]

    # Concatenate all feedback
    all_feedback = " | ".join(r["nl_feedback"] for r in results if r["nl_feedback"])

    # Collect per-tool call logs keyed by tool name
    tool_logs: Dict[str, List[Dict]] = {}
    for r in results:
        for entry in r["tool_log"]:
            tn = entry.get("tool", "")
            tool_logs.setdefault(tn, []).append(entry)

    return {
        "mean_reward": mean_reward,
        "tool_weights": tool_weights,
        "nl_feedback": all_feedback,
        "tool_logs": tool_logs,
        "n_episodes": len(results),
    }


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def _load_state() -> Dict:
    if _STATE_FILE.exists():
        return json.loads(_STATE_FILE.read_text())
    return {"cycle": 0, "history": []}


def _save_state(state: Dict) -> None:
    _STATE_FILE.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _one_cycle(cycle_num: int, n_episodes: int, dry_run: bool, client=None) -> Dict:
    """Run one self-improvement cycle and return the cycle record dict.

    Callable independently so the combined loop can interleave GRPO steps.
    Args:
        client: OpenAI-compatible client (or LocalClient). If None, uses HF router.
    """
    from training.tool_architect import apply_improvement

    print(f"\n--- Self-Improve Cycle {cycle_num} ---", flush=True)

    # 1. Baseline evaluation
    before = evaluate(n_episodes, seeds=list(range(cycle_num * 100, cycle_num * 100 + n_episodes)), client=client)
    print(f"[LOOP] Before: mean_reward={before['mean_reward']:.3f}", flush=True)

    # 2. Build tool flags from weights
    registry = ToolRegistry()
    registry.ema_weights.update(before["tool_weights"])
    tool_flags = {t: registry.flag(t).value for t in before["tool_weights"]}
    print(f"[LOOP] Tool flags: {tool_flags}", flush=True)

    # 3. Call the Architect
    improvement = apply_improvement(
        tool_weights=before["tool_weights"],
        tool_flags=tool_flags,
        nl_feedback=before["nl_feedback"],
        recent_tool_calls=before["tool_logs"],
        dry_run=dry_run,
    )
    print(f"[LOOP] Architect result: {improvement}", flush=True)

    # 4. Post-improvement evaluation
    after = evaluate(n_episodes, seeds=list(range(cycle_num * 100 + 50, cycle_num * 100 + 50 + n_episodes)), client=client)
    delta = after["mean_reward"] - before["mean_reward"]
    print(f"[LOOP] After:  mean_reward={after['mean_reward']:.3f}  delta={delta:+.3f}", flush=True)

    # 5. Revert if delta is negative
    reverted = False
    if delta < 0 and not dry_run:
        file_written = improvement.get("file_written")
        if file_written:
            reverted = _revert_tool(file_written)

    cycle_record = {
        "cycle": cycle_num,
        "before_reward": before["mean_reward"],
        "after_reward": after["mean_reward"],
        "delta": delta,
        "architect_action": improvement.get("action"),
        "target_tool": improvement.get("target_tool"),
        "file_written": improvement.get("file_written"),
        "dry_run": dry_run,
        "reverted": reverted,
    }

    status = "REVERTED" if reverted else ("improved" if delta >= 0 else "kept (no backup to revert)")
    print(f"[LOOP] Cycle {cycle_num} complete. delta={delta:+.3f}  status={status}", flush=True)
    return cycle_record


def run_loop(n_episodes: int, n_cycles: int, dry_run: bool, client=None) -> None:
    state = _load_state()
    start_cycle = state["cycle"]

    print(f"\n{'='*60}", flush=True)
    print(f"[LOOP] Starting self-improvement loop", flush=True)
    print(f"[LOOP] Cycles: {n_cycles}  Episodes per eval: {n_episodes}  Dry-run: {dry_run}", flush=True)
    print(f"[LOOP] Resuming from cycle {start_cycle}", flush=True)
    print(f"{'='*60}\n", flush=True)

    for cycle_idx in range(n_cycles):
        cycle_num = start_cycle + cycle_idx + 1
        cycle_record = _one_cycle(cycle_num, n_episodes, dry_run, client=client)
        state["cycle"] = cycle_num
        state["history"].append(cycle_record)
        _save_state(state)
        _print_history(state["history"][-5:])

    print(f"\n[LOOP] All {n_cycles} cycles complete.", flush=True)


def _revert_tool(file_written: str) -> bool:
    """Restore the .bak backup for the given tool path. Returns True if reverted."""
    from pathlib import Path as _Path
    dest = _Path(file_written)
    backup = dest.with_suffix(".py.bak")
    if backup.exists():
        dest.write_text(backup.read_text())
        backup.unlink()
        print(f"[LOOP] Reverted {dest.name} — negative delta, restored from backup", flush=True)
        return True
    print(f"[LOOP] No backup found for {dest.name}, cannot revert", flush=True)
    return False


def _print_history(history: List[Dict]) -> None:
    print("\n  Cycle | Before | After  | Delta  | Reverted | Action", flush=True)
    print("  ------|--------|--------|--------|----------|------------------", flush=True)
    for h in history:
        rev = "yes" if h.get("reverted") else "no"
        print(
            f"  {h['cycle']:5d} | {h['before_reward']:.3f}  | {h['after_reward']:.3f}  | "
            f"{h['delta']:+.3f}  | {rev:8s} | {h['architect_action']}:{h.get('target_tool','?')}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Genesis self-improvement loop")
    parser.add_argument("--n", type=int, default=3, help="Episodes per evaluation (default: 3)")
    parser.add_argument("--cycles", type=int, default=1, help="Number of improvement cycles (default: 1)")
    parser.add_argument("--dry-run", action="store_true", help="Decide but don't write tool files")
    parser.add_argument("--reset", action="store_true", help="Reset loop state before starting")
    parser.add_argument("--status", action="store_true", help="Print loop history and exit")
    args = parser.parse_args()

    if args.reset and _STATE_FILE.exists():
        _STATE_FILE.unlink()
        print("[LOOP] State reset.", flush=True)

    if args.status:
        state = _load_state()
        print(f"Current cycle: {state['cycle']}")
        if state["history"]:
            _print_history(state["history"])
        else:
            print("No history yet.")
        return

    run_loop(n_episodes=args.n, n_cycles=args.cycles, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
