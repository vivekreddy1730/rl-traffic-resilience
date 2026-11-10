"""
Empirical Calibration & Verification script for Shift Monitor (R2).

1. Collects nominal trajectories across all 5 seeds with frozen DQN agents.
2. Calibrates baseline profile (window W=50).
3. Evaluates False Positive Rate (FPR) across all 5 nominal seeds under mu_0 + 3sigma_0 and reports if adjustment is needed.
4. Tests True Positive Rate (TPR) and Detection Latency against all 4 shift types:
   - Observation shift (p=0.20 bit flips + lane dropout)
   - Transition shift (50% max speed & deceleration)
   - Action shift (1-step actuator latency queue)
   - Reward shift (phase switch penalty + queue disparity penalty)
5. Checkpoints calibration profile to results/phase2_shifts/calibration_profile.json.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np

# Path + SUMO setup (portable; all paths are relative to this file - see src/project_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from project_paths import setup_all, SETTINGS_FILE, VIDALI_DIR as REPO_DIR, MODELS_DIR, PHASE2_DIR, ROOT_DIR, output_dir  # noqa: E402
setup_all()

from tlcs.settings import load_training_settings
from tlcs.agent import Agent
from shifts import ShiftedEnvironment, ShiftConfig, ShiftType
from monitor import ShiftMonitor, BaselineProfile, calibrate_nominal_profile

RESULTS_DIR = PHASE2_DIR  # official inputs (read-only)
OUT_DIR = output_dir(PHASE2_DIR)  # where this script writes; set RESULTS_OUT_DIR to protect official files
CHECKPOINT_DIR = MODELS_DIR  # frozen DQN weights: models/seed_N/trained_model.pt
SEEDS = [1, 2, 3, 4, 5]


def collect_trajectory(env: ShiftedEnvironment, agent: Agent, seed: int) -> list:
    """Run an episode and record per-step (s, a, r, s_next, queue_len, Q(s), Q(s_next))."""
    env.set_seed(seed)
    env.generate_routefile(seed=seed)

    previous_total_wait = 0.0
    previous_action = 0
    trajectory = []

    env.activate()
    state = env.get_state()

    while not env.is_over():
        q_values = agent.model.predict_one(state)
        action = int(np.argmax(q_values))

        action_stats = env.execute(action)
        queue_len = float(env.get_queue_length())
        mean_speed = float(env.get_incoming_mean_speed())

        next_state = env.get_state()
        next_q_values = agent.model.predict_one(next_state)

        current_total_wait = env.get_cumulated_waiting_time()
        reward = env.compute_reward(
            previous_total_wait=previous_total_wait,
            current_total_wait=current_total_wait,
            previous_action=previous_action,
            current_action=action,
        )
        previous_total_wait = current_total_wait
        previous_action = action

        step_record = {
            "state": state.copy(),
            "action": action,
            "reward": float(reward),
            "next_state": next_state.copy(),
            "queue_len": queue_len,
            "mean_speed": mean_speed,
            "q_values": q_values.copy(),
            "next_q_values": next_q_values.copy(),
        }
        trajectory.append(step_record)
        state = next_state

    env.deactivate()
    return trajectory


def run_monitor_verification():
    print("=" * 65)
    print("  PHASE 2 (R2): SHIFT MONITOR EMPIRICAL CALIBRATION & TESTING")
    print("=" * 65)

    settings = load_training_settings(SETTINGS_FILE)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Collect nominal trajectories across all 5 seeds
    print("\n[Step 1] Collecting nominal trajectories across Seeds 1-5...")
    nominal_trajectories = []
    for seed in SEEDS:
        ckpt_path = CHECKPOINT_DIR / f"seed_{seed}"
        agent = Agent(settings=settings, epsilon=0.0, model_path=ckpt_path)
        agent.set_epsilon(0.0)

        ep_seed = seed * 10000
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
        traj = collect_trajectory(env, agent, seed=ep_seed)
        nominal_trajectories.append(traj)
        print(f"  Seed {seed} (ep_seed={ep_seed}): {len(traj)} decision steps recorded.")

    # 2. Calibrate baseline profile on Seeds 1-3
    print("\n[Step 2] Calibrating nominal baseline profile (Window W=50, Seeds 1-3)...")
    profile_3sigma = calibrate_nominal_profile(
        nominal_trajectories=nominal_trajectories[:3],
        window_size=50,
        threshold_multiplier=3.0,
        consecutive_steps_required=3,
    )
    print(f"  Baseline Feature Means: { {k: round(v, 4) for k, v in profile_3sigma.feature_means.items()} }")
    print(f"  Baseline Feature Stds:  { {k: round(v, 4) for k, v in profile_3sigma.feature_stds.items()} }")
    print(f"  Anomaly Score Mean: {profile_3sigma.anomaly_score_mean:.4f}, Std: {profile_3sigma.anomaly_score_std:.4f}")
    print(f"  Initial Threshold (mu + 3.0 sigma): {profile_3sigma.detection_threshold:.4f}")

    # 3. Evaluate False Positive Rate (FPR) across all 5 nominal seeds
    print("\n[Step 3] Evaluating False Positive Rate (FPR) on all 5 nominal seeds...")
    
    def evaluate_nominal_fpr(profile: BaselineProfile) -> tuple[float, list[float]]:
        seed_fprs = []
        for s_idx, traj in enumerate(nominal_trajectories):
            monitor = ShiftMonitor(profile=profile)
            for step in traj:
                monitor.update(
                    state=step["state"],
                    action=step["action"],
                    reward=step["reward"],
                    next_state=step["next_state"],
                    queue_len=step["queue_len"],
                    mean_speed=step["mean_speed"],
                    q_values=step["q_values"],
                    next_q_values=step["next_q_values"],
                )
            alarms = sum(monitor.history_alarms)
            total = len(monitor.history_alarms)
            fpr = (alarms / total) if total > 0 else 0.0
            seed_fprs.append(fpr)
        overall_fpr = float(np.mean(seed_fprs))
        return overall_fpr, seed_fprs

    fpr_3sigma, per_seed_3sigma = evaluate_nominal_fpr(profile_3sigma)
    print(f"  At mu + 3.0 sigma: Overall FPR = {fpr_3sigma:.2%} (Per seed: {[f'{x:.2%}' for x in per_seed_3sigma]})")

    # Evaluate multiple multipliers to show calibration sweep
    final_profile = profile_3sigma
    final_multiplier = 3.0
    
    if fpr_3sigma > 0.05:
        print("\n  [Threshold Tuning] FPR at 3.0 sigma > 5.0%. Performing calibration sweep:")
        for mult in [3.5, 4.0, 4.5, 5.0]:
            test_prof = calibrate_nominal_profile(
                nominal_trajectories=nominal_trajectories[:3],
                window_size=50,
                threshold_multiplier=mult,
                consecutive_steps_required=3,
            )
            test_fpr, _ = evaluate_nominal_fpr(test_prof)
            print(f"    Threshold mult = {mult:.1f} sigma (thresh={test_prof.detection_threshold:.4f}) -> Overall FPR = {test_fpr:.2%}")
            if test_fpr <= 0.05 and final_profile == profile_3sigma:
                final_profile = test_prof
                final_multiplier = mult
    else:
        print(f"  [Threshold Verdict] FPR at 3.0 sigma is {fpr_3sigma:.2%} <= 5.0%. Standard 3.0 sigma is VALID.")

    # 4. Checkpoint calibration profile to results/phase2_shifts/
    calib_path = OUT_DIR / "calibration_profile.json"
    final_profile.save(calib_path)
    print(f"\n[Step 4] Checkpointed calibrated profile to: {calib_path}")

    # 5. Test True Positive Rate (TPR) and Detection Latency across all 4 shift types on held-out seeds
    print("\n[Step 5] Evaluating True Positive Rate (TPR) and Detection Latency on held-out seeds (Seeds 4 & 5)...")
    
    shift_configs = {
        "Observation Shift (p=0.20, lane 0 dropout)": ShiftConfig(
            shift_type=ShiftType.OBSERVATION, obs_bit_flip_prob=0.20, obs_dropout_lanes=[0]
        ),
        "Transition Shift (50% max speed & decel)": ShiftConfig(
            shift_type=ShiftType.TRANSITION, trans_speed_factor=0.50, trans_decel_factor=0.50
        ),
        "Action Shift (1-step actuator latency)": ShiftConfig(
            shift_type=ShiftType.ACTION, action_latency_steps=1
        ),
        "Reward Shift (Switch penalty=10, queue imbalance=2)": ShiftConfig(
            shift_type=ShiftType.REWARD, reward_switch_penalty=10.0, reward_queue_penalty_weight=2.0
        ),
    }

    test_seeds = [4, 5]
    results_summary = []

    for name, s_cfg in shift_configs.items():
        print(f"\n  Testing {name}:")
        detected_seeds = 0
        latencies = []

        for seed in test_seeds:
            ckpt_path = CHECKPOINT_DIR / f"seed_{seed}"
            agent = Agent(settings=settings, epsilon=0.0, model_path=ckpt_path)
            agent.set_epsilon(0.0)

            ep_seed = seed * 10000
            env = ShiftedEnvironment(
                n_cars_generated=settings.n_cars_generated,
                max_steps=settings.max_steps,
                yellow_duration=settings.yellow_duration,
                green_duration=settings.green_duration,
                turn_chance=settings.turn_chance,
                sumocfg_file=settings.sumocfg_file,
                gui=False,
                shift_config=s_cfg,
            )
            traj = collect_trajectory(env, agent, seed=ep_seed)
            
            monitor = ShiftMonitor(profile=final_profile)
            first_detect_step = None
            for step_idx, step in enumerate(traj):
                _, is_detected, _ = monitor.update(
                    state=step["state"],
                    action=step["action"],
                    reward=step["reward"],
                    next_state=step["next_state"],
                    queue_len=step["queue_len"],
                    mean_speed=step["mean_speed"],
                    q_values=step["q_values"],
                    next_q_values=step["next_q_values"],
                )
                if is_detected and first_detect_step is None:
                    first_detect_step = step_idx

            if first_detect_step is not None:
                detected_seeds += 1
                latencies.append(first_detect_step)
                print(f"    Seed {seed}: DETECTED at step {first_detect_step} (out of {len(traj)} steps)")
            else:
                print(f"    Seed {seed}: NOT DETECTED")

        tpr = detected_seeds / len(test_seeds)
        avg_lat = float(np.mean(latencies)) if latencies else float("nan")
        results_summary.append({
            "shift_name": name,
            "tpr": tpr,
            "avg_latency_steps": avg_lat,
        })
        print(f"    -> TPR: {tpr:.0%}, Mean Detection Latency: {avg_lat:.1f} decision steps")

    # Save summary report to JSON
    summary_path = OUT_DIR / "monitor_verification_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "fpr_3sigma": fpr_3sigma,
            "final_multiplier": final_multiplier,
            "final_detection_threshold": final_profile.detection_threshold,
            "shifts_evaluated": results_summary,
        }, f, indent=2)
    print(f"\n[Step 6] Saved verification summary to: {summary_path}")

    print("\n" + "=" * 65)
    print("  SHIFT MONITOR (R2) VERIFICATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    run_monitor_verification()
