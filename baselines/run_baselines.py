"""
Fixed-Time and Max Pressure baseline controllers for Phase 1.

Both controllers use the same SUMO environment as the DQN but replace the
neural network with deterministic/rule-based policies.

Usage (from project root):
    python baselines/run_baselines.py
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
from project_paths import setup_all, SETTINGS_FILE, VIDALI_DIR as REPO_DIR, MODELS_DIR, PHASE2_DIR, ROOT_DIR, output_dir  # noqa: E402
setup_all()

import sumolib
import traci
import numpy as np
from tlcs.settings import load_training_settings
from tlcs.env import Environment
from tlcs.generator import generate_routefile
from tlcs.constants import (
    ACTION_TO_TL_PHASE,
    TRAFFIC_LIGHT_ID,
    TL_GREEN_TO_YELLOW,
    INCOMING_EDGES,
)

out_env = os.environ.get("BASELINES_OUT_DIR", "")
if out_env:
    OUT_DIR = Path(out_env)
else:
    OUT_DIR = output_dir(ROOT_DIR / "results" / "phase1_baseline_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Fixed-Time Controller
# ---------------------------------------------------------------------------

class FixedTimeController:
    """Cycles through all 4 green phases with a fixed duration each.

    This is the classical pre-timed signal controller baseline.
    """

    def __init__(self, green_duration: int, yellow_duration: int):
        """Initialize fixed-time controller.

        Args:
            green_duration: Steps each green phase is held.
            yellow_duration: Steps each yellow (transition) phase is held.
        """
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.phase_index = 0  # cycles through 0,1,2,3
        self.steps_in_phase = 0

    def choose_action(self, _state) -> int:
        """Return the current scheduled phase action (ignores state).

        Args:
            _state: Unused state vector (interface compatibility).

        Returns:
            Phase action index (0-3).
        """
        if self.steps_in_phase >= self.green_duration:
            self.phase_index = (self.phase_index + 1) % 4
            self.steps_in_phase = 0
        self.steps_in_phase += 1
        return self.phase_index

    def reset(self):
        """Reset the phase counter for a new episode."""
        self.phase_index = 0
        self.steps_in_phase = 0


# ---------------------------------------------------------------------------
# Max Pressure Controller
# ---------------------------------------------------------------------------

class MaxPressureController:
    """Selects the phase that maximises pressure across competing lane groups.

    Pressure for a phase = sum of halted vehicles on that phase's incoming
    lanes minus sum of halted on corresponding outgoing lanes (approximated
    here as incoming lane queues).

    For this intersection the 4 actions correspond to:
        0: NS straight  (N2TL_0, N2TL_1, N2TL_2, S2TL_0, S2TL_1, S2TL_2)
        1: NS left-turn (N2TL_3, S2TL_3)
        2: EW straight  (E2TL_0, E2TL_1, E2TL_2, W2TL_0, W2TL_1, W2TL_2)
        3: EW left-turn (E2TL_3, W2TL_3)
    """

    # Incoming lanes serving each action independently
    ACTION_TO_LANES = {
        0: ("N2TL_0", "N2TL_1", "N2TL_2", "S2TL_0", "S2TL_1", "S2TL_2"),
        1: ("N2TL_3", "S2TL_3"),
        2: ("E2TL_0", "E2TL_1", "E2TL_2", "W2TL_0", "W2TL_1", "W2TL_2"),
        3: ("E2TL_3", "W2TL_3"),
    }

    def __init__(self, min_green: int = 5):
        """Initialize Max Pressure controller.

        Args:
            min_green: Minimum steps a phase must be held before switching.
        """
        self.min_green = min_green
        self.current_action = 0
        self.steps_in_action = 0
        self.action_history = []

    def _get_pressure(self, lanes: tuple) -> int:
        """Compute total halted vehicles on the given lanes.

        Args:
            lanes: Tuple of lane IDs to query.

        Returns:
            Total halted vehicle count.
        """
        return sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)

    def choose_action(self, _state) -> int:
        """Return the action with maximum incoming pressure.

        Args:
            _state: Unused (pressure is computed directly from traci).

        Returns:
            Phase action index (0-3).
        """
        if self.steps_in_action < self.min_green:
            self.steps_in_action += 1
            self.action_history.append(self.current_action)
            return self.current_action

        pressures = {
            action: self._get_pressure(lanes)
            for action, lanes in self.ACTION_TO_LANES.items()
        }
        max_pressure = max(pressures.values())

        # If current action achieves max pressure and queue > 0, hold phase
        if pressures[self.current_action] == max_pressure and max_pressure > 0:
            best_action = self.current_action
        else:
            best_candidates = [a for a, p in pressures.items() if p == max_pressure]
            if len(best_candidates) == 1:
                best_action = best_candidates[0]
            elif self.current_action in best_candidates:
                best_action = self.current_action
            else:
                # Cyclic tie-breaking from current action to prevent bias to action 0
                best_action = min(best_candidates, key=lambda a: (a - self.current_action) % 4)

        if best_action != self.current_action:
            self.current_action = best_action
            self.steps_in_action = 0
        self.steps_in_action += 1
        self.action_history.append(self.current_action)
        return self.current_action

    def reset(self):
        """Reset state for a new episode."""
        self.current_action = 0
        self.steps_in_action = 0
        self.action_history = []


# ---------------------------------------------------------------------------
# Episode runner for non-DQN controllers
# ---------------------------------------------------------------------------

def run_baseline_episode(controller, settings, seed: int) -> tuple:
    """Run one simulation episode with a baseline controller.

    Args:
        controller: Controller instance with choose_action(state) -> int.
        settings: Training settings object.
        seed: Episode seed.

    Returns:
        Tuple of (list[reward], list[queue_length]).
    """
    generate_routefile(
        seed=seed,
        n_cars_generated=settings.n_cars_generated,
        max_steps=settings.max_steps,
        turn_chance=settings.turn_chance,
    )

    controller.reset()

    sumo_binary = sumolib.checkBinary("sumo")
    sumo_cmd = [
        sumo_binary,
        "-c", str(settings.sumocfg_file),
        "--no-step-log", "true",
        "--waiting-time-memory", str(settings.max_steps),
    ]

    traci.start(sumo_cmd)

    step = 0
    rewards = []
    queue_lengths = []
    previous_total_wait = 0.0

    # state placeholder (baseline controllers may ignore it)
    state = np.zeros(80, dtype=float)

    current_green_phase = 0
    green_duration = settings.green_duration
    yellow_duration = settings.yellow_duration
    steps_in_phase = 0

    # We drive the simulation manually using the action from choose_action
    while step < settings.max_steps:
        action = controller.choose_action(state)
        next_green_phase = ACTION_TO_TL_PHASE[action]

        # Yellow transition if phase changes
        if next_green_phase != current_green_phase:
            yellow_phase = TL_GREEN_TO_YELLOW[current_green_phase]
            traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, yellow_phase)
            for _ in range(min(yellow_duration, settings.max_steps - step)):
                traci.simulationStep()
                step += 1
                queue_lengths.append(_get_queue_length())

        if step >= settings.max_steps:
            break

        # Green phase
        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, next_green_phase)
        current_green_phase = next_green_phase

        for _ in range(min(green_duration, settings.max_steps - step)):
            traci.simulationStep()
            step += 1
            queue_lengths.append(_get_queue_length())

        # Reward
        current_total_wait = _get_cumulated_waiting_time()
        reward = previous_total_wait - current_total_wait
        previous_total_wait = current_total_wait
        rewards.append(reward)

    traci.close()
    return rewards, queue_lengths


def _get_queue_length() -> int:
    """Return total halted vehicles on all incoming edges."""
    return sum(
        traci.edge.getLastStepHaltingNumber(e)
        for e in ("N2TL", "S2TL", "E2TL", "W2TL")
    )


def _get_cumulated_waiting_time() -> float:
    """Return total accumulated waiting time on incoming edges."""
    wait = 0.0
    for car_id in traci.vehicle.getIDList():
        road_id = traci.vehicle.getRoadID(car_id)
        if road_id in INCOMING_EDGES:
            wait += float(traci.vehicle.getAccumulatedWaitingTime(car_id))
    return wait


# ---------------------------------------------------------------------------
# Multi-seed runner for a given controller
# ---------------------------------------------------------------------------

def run_controller(name: str, controller, settings, seeds: list) -> list:
    """Run a controller over multiple seeds and save CSVs.

    Args:
        name: Controller name (for file naming).
        controller: Controller instance.
        settings: Training settings.
        seeds: List of seed integers.

    Returns:
        List of per-seed summary dicts.
    """
    all_episode_rows = []
    seed_summaries = []

    for seed in seeds:
        ep_seed = seed * 10000  # single episode per seed for baselines
        print(f"\n  [{name}] Seed {seed} | ep_seed={ep_seed}")

        t0 = time.time()
        rewards, queue_lengths = run_baseline_episode(controller, settings, ep_seed)
        elapsed = time.time() - t0

        sum_neg_reward = sum(r for r in rewards if r < 0)
        cumulative_wait = sum(queue_lengths)
        avg_queue = round(cumulative_wait / max(settings.max_steps, 1), 4)
        avg_waiting_time = round(cumulative_wait / max(len(rewards), 1), 4)
        avg_delay = abs(sum_neg_reward) / max(len(rewards), 1)

        row = {
            "controller": name,
            "seed": seed,
            "sum_neg_reward": round(sum_neg_reward, 4),
            "cumulative_wait": cumulative_wait,
            "avg_waiting_time": avg_waiting_time,
            "avg_delay": round(avg_delay, 4),
            "avg_queue_length": avg_queue,
            "throughput_steps": len(rewards),
            "elapsed_s": round(elapsed, 2),
        }
        all_episode_rows.append(row)
        seed_summaries.append(row)

        print(
            f"    neg_rew={sum_neg_reward:.1f} | wait={cumulative_wait} | "
            f"queue={avg_queue:.2f} | {elapsed:.1f}s"
        )

    # Save per-controller CSV
    out_file = OUT_DIR / f"{name.lower().replace(' ', '_')}_seeds.csv"
    with out_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_episode_rows)
    print(f"  -> Saved: {out_file.name}")

    return seed_summaries


def print_and_save_summary(all_controllers: dict) -> None:
    """Print and save cross-controller summary CSV.

    Args:
        all_controllers: Dict mapping controller name -> list of seed rows.
    """
    metrics = ["avg_waiting_time", "avg_delay", "avg_queue_length", "sum_neg_reward"]
    summary_rows = []

    print(f"\n{'='*65}")
    print("  BASELINE SUMMARY (mean +/- std across seeds)")
    print(f"{'='*65}")

    for ctrl_name, rows in all_controllers.items():
        for metric in metrics:
            vals = [r[metric] for r in rows]
            mean_val = statistics.mean(vals)
            std_val = statistics.stdev(vals) if len(vals) > 1 else 0.0
            summary_rows.append({
                "controller": ctrl_name,
                "metric": metric,
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "n_seeds": len(rows),
            })
            print(f"  {ctrl_name:20s} | {metric:25s}: {mean_val:>8.4f} +/- {std_val:.4f}")

    summary_file = OUT_DIR / "summary_baselines.csv"
    with summary_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["controller", "metric", "mean", "std", "n_seeds"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n  -> Saved: {summary_file.name}")


def main():
    print("\n" + "="*65)
    print("  PHASE 1 -- Reference Baselines (Fixed-Time + Max Pressure)")
    print("="*65)

    settings = load_training_settings(SETTINGS_FILE)

    ft_controller = FixedTimeController(
        green_duration=settings.green_duration,
        yellow_duration=settings.yellow_duration,
    )
    mp_controller = MaxPressureController(min_green=settings.green_duration)

    all_controllers = {}
    all_controllers["FixedTime"] = run_controller("FixedTime", ft_controller, settings, SEEDS)
    all_controllers["MaxPressure"] = run_controller("MaxPressure", mp_controller, settings, SEEDS)

    print_and_save_summary(all_controllers)
    print("\nBaseline runs complete.")


if __name__ == "__main__":
    main()
