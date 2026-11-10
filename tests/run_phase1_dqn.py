"""
Phase 1 — Multi-seed DQN runner.

Calls Vidali training_session() for seeds 1-5 (via reproducible episode seeds),
logs per-episode metrics, and saves CSVs + checkpoints to results/retrain_phase1/
(a new folder, so the shipped results/ and models/ are never overwritten).

OPTIONAL and SLOW: 5 seeds x 100 episodes. The trained weights are already in models/.

Usage (from project root, with the venv activated):
    python tests/run_phase1_dqn.py
"""

import csv
import os
import sys
import time
import statistics
from datetime import datetime
from pathlib import Path

# Path + SUMO setup (portable; all paths are relative to this file - see src/project_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from project_paths import setup_all, SETTINGS_FILE, VIDALI_DIR as REPO_DIR, ROOT_DIR, output_dir  # noqa: E402
setup_all()  # also changes CWD to the Vidali folder so intersection/ resolves

from tlcs.settings import load_training_settings
from tlcs.agent import Agent
from tlcs.env import Environment
from tlcs.episode import run_episode
from tlcs.memory import Memory, Sample

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
# Retraining writes to a NEW folder so the shipped results/models are never overwritten.
# (Override with the RESULTS_OUT_DIR environment variable.)
OUT_DIR = output_dir(ROOT_DIR / "results" / "retrain_phase1")

SEEDS = [1, 2, 3, 4, 5]


def run_seed(seed: int) -> dict:
    """Run a full training session for a given global seed.

    We use the seed to offset episode seeds so that each seed produces a
    distinct, reproducible trajectory. Returns per-episode metrics as a list
    of dicts.

    Args:
        seed: Outer seed index (1-5).

    Returns:
        dict with key 'episodes': list of per-episode metric dicts.
    """
    print(f"\n{'='*60}")
    print(f"  DQN Seed {seed} — started at {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")

    settings = load_training_settings(SETTINGS_FILE)

    memory = Memory(size_max=settings.memory_size_max, size_min=settings.memory_size_min)
    agent = Agent(settings=settings)

    tot_episodes = settings.total_episodes
    episode_rows = []

    for episode in range(tot_episodes):
        # episode seed is deterministic: outer_seed * 10000 + episode_index
        ep_seed = seed * 10000 + episode
        new_epsilon = round(1.0 - (episode / tot_episodes), 2)
        agent.set_epsilon(new_epsilon)

        env = Environment(
            n_cars_generated=settings.n_cars_generated,
            max_steps=settings.max_steps,
            yellow_duration=settings.yellow_duration,
            green_duration=settings.green_duration,
            turn_chance=settings.turn_chance,
            gui=False,
            sumocfg_file=settings.sumocfg_file,
        )

        t0 = time.time()
        episode_history, env_stats = run_episode(env=env, agent=agent, seed=ep_seed)
        elapsed = time.time() - t0

        # Add experience to memory
        for i in range(len(episode_history) - 1):
            sample = Sample(
                state=episode_history[i].state,
                action=episode_history[i].action,
                reward=episode_history[i].reward,
                next_state=episode_history[i + 1].state,
            )
            memory.add_sample(sample)

        # Train
        for _ in range(settings.training_epochs):
            agent.replay(memory=memory, gamma=settings.gamma, batch_size=settings.batch_size)

        # Collect metrics
        sum_neg_reward = sum(r.reward for r in episode_history if r.reward < 0)
        cumulative_wait = sum(s.queue_length for s in env_stats)
        avg_queue = round(cumulative_wait / max(settings.max_steps, 1), 4)
        throughput = len(episode_history)
        avg_waiting_time = round(cumulative_wait / max(len(episode_history), 1), 4)
        avg_delay = abs(sum_neg_reward) / max(len(episode_history), 1)

        row = {
            "seed": seed,
            "episode": episode + 1,
            "epsilon": new_epsilon,
            "sum_neg_reward": round(sum_neg_reward, 4),
            "cumulative_wait": cumulative_wait,
            "avg_waiting_time": avg_waiting_time,
            "avg_delay": round(avg_delay, 4),
            "avg_queue_length": avg_queue,
            "throughput_steps": throughput,
            "elapsed_s": round(elapsed, 2),
        }
        episode_rows.append(row)

        if (episode + 1) % 10 == 0 or episode == 0:
            print(
                f"  Ep {episode+1:>3}/{tot_episodes} | eps={new_epsilon:.2f} | "
                f"neg_rew={sum_neg_reward:>8.1f} | wait={cumulative_wait:>6} | "
                f"queue={avg_queue:.2f} | {elapsed:.1f}s"
            )

    # Save model checkpoint at the end of the 100 episodes
    ckpt_dir = OUT_DIR / "checkpoints" / f"seed_{seed}"
    agent.save_model(ckpt_dir)
    print(f"  -> Model checkpoint saved: {ckpt_dir / 'trained_model.pt'}")

    return {"seed": seed, "episodes": episode_rows}


def save_seed_csv(seed: int, rows: list) -> Path:
    """Save per-episode metrics for one seed to CSV."""
    out_file = OUT_DIR / f"dqn_seed_{seed}.csv"
    if not rows:
        return out_file
    fieldnames = list(rows[0].keys())
    with out_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> Saved: {out_file.name}")
    return out_file


def compute_summary(all_results: list) -> None:
    """Compute mean +/- std across seeds for last-10-episode metrics."""
    metrics = [
        "avg_waiting_time", "avg_delay", "avg_queue_length",
        "sum_neg_reward", "throughput_steps",
    ]

    summary_rows = []
    for metric in metrics:
        vals = []
        for result in all_results:
            last_10 = result["episodes"][-10:]
            vals.extend([ep[metric] for ep in last_10])

        mean_val = statistics.mean(vals)
        std_val = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summary_rows.append({
            "controller": "DQN",
            "metric": metric,
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "n_seeds": len(all_results),
        })

    summary_file = OUT_DIR / "summary_dqn.csv"
    with summary_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["controller", "metric", "mean", "std", "n_seeds"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n{'='*60}")
    print("  DQN Summary (last 10 episodes per seed, mean +/- std):")
    print(f"{'='*60}")
    for row in summary_rows:
        print(f"  {row['metric']:30s}: {row['mean']:>10.4f} +/- {row['std']:.4f}")
    print(f"  -> Saved: {summary_file.name}")


def main():
    print("\n" + "="*60)
    print("  PHASE 1 -- DQN Baseline (5 seeds)")
    print("="*60)

    all_results = []
    for seed in SEEDS:
        result = run_seed(seed)
        save_seed_csv(seed, result["episodes"])
        all_results.append(result)

    compute_summary(all_results)
    print("\nPhase 1 DQN baseline complete.")


if __name__ == "__main__":
    main()
