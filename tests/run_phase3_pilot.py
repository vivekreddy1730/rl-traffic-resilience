"""
Phase 3 Pilot Evaluation Script.

Executes a scoped evaluation on Seeds 7, 8, 9 (fresh, unseen seeds):
- Seed 7: Observation shift (p=0.20 bit flips, lane 0 dropout) injected mid-episode
- Seed 8: Transition shift (50% max speed & deceleration) injected mid-episode
- Seed 9: Action shift (1-step actuator latency) injected mid-episode

Compares 4 controllers on each seed:
1. DQN_Unmitigated: Nominal DQN agent running under shift without intervention
2. DQN_Adaptive: Autonomous AdaptiveController (Monitor -> Diagnosis -> Mitigation -> Fallback)
3. Fixed_Time: Classical 4-phase pre-timed controller
4. Max_Pressure: Corrected lane-level 4-phase Max-Pressure controller

Outputs results to results/phase3_pilot/ with summary table CSV and comparison plot PNG.
"""

import os
import sys
import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Path + SUMO setup (portable; all paths are relative to this file - see src/project_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from project_paths import setup_all, SETTINGS_FILE, VIDALI_DIR as REPO_DIR, MODELS_DIR, PHASE2_DIR, ROOT_DIR, output_dir  # noqa: E402
setup_all()

import traci
from tlcs.settings import load_training_settings
from tlcs.agent import Agent
from shifts import ShiftedEnvironment, ShiftConfig, ShiftType
from adaptive_controller import AdaptiveController

# Import baseline controllers from baselines
sys.path.insert(0, str(ROOT_DIR / "baselines"))
from run_baselines import FixedTimeController, MaxPressureController

RESULTS_DIR = output_dir(ROOT_DIR / "results" / "phase3_pilot")
CHECKPOINT_DIR = MODELS_DIR  # frozen DQN weights: models/seed_N/trained_model.pt
CALIB_PROFILE = ROOT_DIR / "results" / "phase2_shifts" / "calibration_profile.json"
DIAG_CLASSIFIER = ROOT_DIR / "results" / "phase2_shifts" / "diagnosis_classifier.joblib"

INJECTION_STEP = 120  # Mid-episode injection step (~1200s into 3600s episode)


def run_episode(
    controller_type: str,
    seed: int,
    target_shift_cfg: ShiftConfig,
    settings: Any,
) -> Dict[str, Any]:
    """Execute a single episode under the given controller and mid-episode shift."""
    ep_seed = seed * 10000

    # Start with nominal shift configuration
    env = ShiftedEnvironment(
        n_cars_generated=settings.n_cars_generated,
        max_steps=settings.max_steps,
        yellow_duration=settings.yellow_duration,
        green_duration=settings.green_duration,
        turn_chance=settings.turn_chance,
        sumocfg_file=settings.sumocfg_file,
        gui=False,
        shift_config=ShiftConfig(shift_type=ShiftType.NOMINAL),
    )
    env.set_seed(ep_seed)
    env.generate_routefile(seed=ep_seed)

    # Initialize controller
    agent = Agent(settings=settings, epsilon=0.0, model_path=CHECKPOINT_DIR / "seed_1")
    adaptive_ctrl = None
    fixed_time_ctrl = None
    max_pressure_ctrl = None

    if controller_type == "DQN_Adaptive":
        adaptive_ctrl = AdaptiveController(
            agent=agent,
            profile_path=CALIB_PROFILE,
            classifier_path=DIAG_CLASSIFIER,
        )
    elif controller_type == "Fixed_Time":
        fixed_time_ctrl = FixedTimeController(
            green_duration=settings.green_duration,
            yellow_duration=settings.yellow_duration,
        )
    elif controller_type == "Max_Pressure":
        max_pressure_ctrl = MaxPressureController(min_green=5)

    waiting_times = []
    queues = []
    previous_total_wait = 0.0
    previous_action = 0
    total_steps = 0
    shift_injected = False
    diagnoses_seen = []
    fallback_steps = 0

    env.activate()
    state = env.get_state()

    while not env.is_over():
        total_steps += 1
        current_queue = float(env.get_queue_length())
        mean_speed = float(env.get_incoming_mean_speed())
        free_flow_speed = float(env.get_incoming_free_flow_speed())

        # Mid-episode shift injection
        if total_steps >= INJECTION_STEP and not shift_injected:
            env.shift_config = target_shift_cfg
            if target_shift_cfg.shift_type == ShiftType.TRANSITION:
                env._apply_transition_shift()
            shift_injected = True
            print(f"      [t={total_steps}] INJECTED SHIFT: {target_shift_cfg.shift_type.value}")

        extra_green_steps = 0

        # Controller action decision
        if controller_type == "DQN_Unmitigated":
            q_values = agent.model.predict_one(state)
            effective_action = int(np.argmax(q_values))

        elif controller_type == "DQN_Adaptive":
            prev_wait_delta = previous_total_wait - env.get_cumulated_waiting_time()
            effective_action, extra_green_steps, info = adaptive_ctrl.select_action(
                raw_state=state,
                queue_len=current_queue,
                mean_speed=mean_speed,
                free_flow_speed=free_flow_speed,
                prev_reward=0.0,
                prev_wait_delta=prev_wait_delta,
            )
            diagnoses_seen.append(info["active_diagnosis"])
            if info["is_fallback_active"]:
                fallback_steps += 1

        elif controller_type == "Fixed_Time":
            effective_action = fixed_time_ctrl.choose_action(state)

        elif controller_type == "Max_Pressure":
            effective_action = max_pressure_ctrl.choose_action(state)

        else:
            raise ValueError(f"Unknown controller: {controller_type}")

        # Execute action in environment
        if extra_green_steps > 0:
            env.green_duration += extra_green_steps
            env.execute(effective_action)
            env.green_duration -= extra_green_steps
        else:
            env.execute(effective_action)

        current_total_wait = env.get_cumulated_waiting_time()
        previous_total_wait = current_total_wait
        previous_action = effective_action

        waiting_times.append(current_total_wait)
        queues.append(current_queue)
        state = env.get_state()

    env.deactivate()

    avg_wait = float(np.mean(waiting_times)) if waiting_times else 0.0
    avg_queue = float(np.mean(queues)) if queues else 0.0
    fallback_rate = (fallback_steps / total_steps) if total_steps > 0 else 0.0

    return {
        "avg_waiting_time": avg_wait,
        "avg_queue_length": avg_queue,
        "fallback_rate": fallback_rate,
        "total_steps": total_steps,
        "diagnoses_seen": diagnoses_seen,
    }


def main():
    print("=" * 70)
    print("  PHASE 3 PILOT EVALUATION: ADAPTIVE CONTROLLER VS BASELINES")
    print("=" * 70)

    settings = load_training_settings(SETTINGS_FILE)

    pilot_scenarios = [
        {
            "seed": 7,
            "shift_name": "Observation (dropout p=0.20, lane 0)",
            "config": ShiftConfig(
                shift_type=ShiftType.OBSERVATION,
                obs_bit_flip_prob=0.20,
                obs_dropout_lanes=[0],
            ),
        },
        {
            "seed": 8,
            "shift_name": "Transition (50% max speed & decel)",
            "config": ShiftConfig(
                shift_type=ShiftType.TRANSITION,
                trans_speed_factor=0.50,
                trans_decel_factor=0.50,
            ),
        },
        {
            "seed": 9,
            "shift_name": "Action (1-step actuator latency)",
            "config": ShiftConfig(
                shift_type=ShiftType.ACTION,
                action_latency_steps=1,
            ),
        },
    ]

    controllers = [
        "DQN_Unmitigated",
        "DQN_Adaptive",
        "Fixed_Time",
        "Max_Pressure",
    ]

    results = []

    for sc in pilot_scenarios:
        seed = sc["seed"]
        s_name = sc["shift_name"]
        cfg = sc["config"]

        print(f"\n>>> Running Scenario: Seed {seed} | Shift: {s_name} (Injected at step {INJECTION_STEP})")

        seed_results = {"seed": seed, "shift": s_name}

        for ctrl in controllers:
            print(f"    Evaluating Controller: {ctrl} ...")
            t0 = time.time()
            res = run_episode(ctrl, seed, cfg, settings)
            elapsed = time.time() - t0
            print(f"      -> Avg Waiting Time: {res['avg_waiting_time']:8.2f} s | Queue: {res['avg_queue_length']:5.2f} | Time: {elapsed:.1f}s")

            seed_results[f"{ctrl}_wait"] = res["avg_waiting_time"]
            seed_results[f"{ctrl}_queue"] = res["avg_queue_length"]
            if ctrl == "DQN_Adaptive":
                seed_results["adaptive_fallback_rate"] = res["fallback_rate"]

        results.append(seed_results)

    # Save summary table CSV
    csv_path = RESULTS_DIR / "phase3_pilot_summary.csv"
    fieldnames = [
        "seed", "shift",
        "DQN_Unmitigated_wait", "DQN_Adaptive_wait", "Fixed_Time_wait", "Max_Pressure_wait",
        "DQN_Unmitigated_queue", "DQN_Adaptive_queue", "Fixed_Time_queue", "Max_Pressure_queue",
        "adaptive_fallback_rate",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\n[Saved] Summary table saved to: {csv_path}")

    # Generate Comparison Plot
    plot_path = RESULTS_DIR / "phase3_pilot_comparison.png"
    plt.figure(figsize=(10, 6))

    labels = [f"Seed {r['seed']}\n({r['shift'].split(' ')[0]})" for r in results]
    x = np.arange(len(labels))
    width = 0.20

    plt.bar(x - 1.5 * width, [r["DQN_Unmitigated_wait"] for r in results], width, label="DQN (Unmitigated)", color="#e74c3c")
    plt.bar(x - 0.5 * width, [r["DQN_Adaptive_wait"] for r in results], width, label="DQN (Adaptive Closed-Loop)", color="#2ecc71")
    plt.bar(x + 0.5 * width, [r["Fixed_Time_wait"] for r in results], width, label="Fixed-Time Baseline", color="#3498db")
    plt.bar(x + 1.5 * width, [r["Max_Pressure_wait"] for r in results], width, label="Max-Pressure Baseline", color="#9b59b6")

    plt.xlabel("Evaluation Scenario (Seed & Mid-Episode Shift Type)", fontsize=12, fontweight="bold")
    plt.ylabel("Average Vehicle Waiting Time (s)", fontsize=12, fontweight="bold")
    plt.title("Phase 3 Pilot: Adaptive Controller vs Baselines under Mid-Episode Shifts", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(x, labels, fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()

    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[Saved] Comparison plot saved to: {plot_path}")

    print("\n" + "=" * 70)
    print("  PHASE 3 PILOT EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
