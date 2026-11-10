"""
Task 4: Monitor Grid Evaluation (Report Only)
Evaluate FPR on nominal runs and detection on the 10 swept seeds for:
- Threshold multipliers: 2.0, 2.5, 3.0 sigma
- Persistence: 1, 2, 3 steps

DO NOT edit monitor.py or calibration_profile.json.
Labelled: "exploratory only, these seeds are exposed"
The final threshold must not be chosen from this table.
"""
import os
import sys
import json
from pathlib import Path
import numpy as np

# Path + SUMO setup (portable; all paths are relative to this file - see src/project_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from project_paths import setup_all, SETTINGS_FILE, VIDALI_DIR as REPO_DIR, MODELS_DIR, PHASE2_DIR, ROOT_DIR, output_dir  # noqa: E402
setup_all()
ROOT, REPO = ROOT_DIR, REPO_DIR

from tlcs.settings import load_training_settings
from tlcs.agent import Agent
from shifts import ShiftedEnvironment, ShiftConfig, ShiftType
from monitor import BaselineProfile, ShiftMonitor

SETTINGS_FILE = REPO / "settings" / "training_settings.yaml"
CHECKPOINT_DIR = MODELS_DIR  # frozen DQN weights: models/seed_N/trained_model.pt
CALIB_PROFILE_PATH = ROOT / "results" / "phase2_shifts" / "calibration_profile.json"
INJECTION_STEP = 120

settings = load_training_settings(SETTINGS_FILE)
calib_profile = BaselineProfile.load(CALIB_PROFILE_PATH)

mu = calib_profile.anomaly_score_mean
sigma = calib_profile.anomaly_score_std

multipliers = [2.0, 2.5, 3.0]
persistences = [1, 2, 3]

print(f"Calibration baseline: mu={mu:.4f}, sigma={sigma:.4f}")
print("Threshold values:")
for m in multipliers:
    print(f"  {m:.1f} sigma: {mu + m * sigma:.4f}")

CACHE_FILE = ROOT / "results" / "final_checks" / "eval_trajectories_cache.pkl"
import pickle

if CACHE_FILE.exists():
    print(f"\n[Cache] Loading pre-collected trajectories from {CACHE_FILE}...")
    with open(CACHE_FILE, "rb") as f:
        cache_data = pickle.load(f)
    nominal_trajectories = cache_data["nominal"]
    swept_trajectories = cache_data["swept"]
    print(f"  Loaded {len(nominal_trajectories)} nominal and {len(swept_trajectories)} swept trajectories.")
else:
    # ---------------------------------------------------------------------------
    # 1. Collect / Run Nominal Trajectories (Seeds 1-5)
    # ---------------------------------------------------------------------------
    print("\n[Step 1] Running nominal trajectories across Seeds 1-5...")
    nominal_trajectories = {}

    for seed in [1, 2, 3, 4, 5]:
        ckpt_path = CHECKPOINT_DIR / f"seed_{seed}"
        agent = Agent(settings=settings, epsilon=0.0, model_path=ckpt_path)
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
        env.set_seed(ep_seed)
        env.generate_routefile(seed=ep_seed)
        env.activate()
        state = env.get_state()
        traj = []
        prev_wait = 0.0
        prev_act = 0

        while not env.is_over():
            q_values = agent.model.predict_one(state)
            action = int(np.argmax(q_values))
            env.execute(action)
            q_len = float(env.get_queue_length())
            m_speed = float(env.get_incoming_mean_speed())
            next_state = env.get_state()
            next_q = agent.model.predict_one(next_state)
            cur_wait = env.get_cumulated_waiting_time()
            reward = env.compute_reward(prev_wait, cur_wait, prev_act, action)
            prev_wait = cur_wait
            prev_act = action

            traj.append({
                "state": state.copy(),
                "action": action,
                "reward": float(reward),
                "next_state": next_state.copy(),
                "queue_len": q_len,
                "mean_speed": m_speed,
                "q_values": q_values.copy(),
                "next_q_values": next_q.copy(),
            })
            state = next_state
        env.deactivate()
        nominal_trajectories[seed] = traj
        print(f"  Nominal Seed {seed}: {len(traj)} steps collected.")

    # ---------------------------------------------------------------------------
    # 2. Collect / Run Shifted Trajectories on the 10 Swept Seeds
    # (Observation Shift injected at step 120, using Checkpoint seed_1)
    # ---------------------------------------------------------------------------
    print("\n[Step 2] Running 10 swept seeds under mid-episode Observation shift (Checkpoint seed_1)...")
    swept_seeds = [4, 5, 6, 7, 10, 11, 12, 13, 14, 15]
    obs_shift = ShiftConfig(shift_type=ShiftType.OBSERVATION, obs_bit_flip_prob=0.20, obs_dropout_lanes=[0])
    agent_seed1 = Agent(settings=settings, epsilon=0.0, model_path=CHECKPOINT_DIR / "seed_1")
    swept_trajectories = {}

    for s in swept_seeds:
        ep_seed = s * 10000
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
        env.activate()
        state = env.get_state()
        traj = []
        steps = 0
        shift_on = False
        prev_wait = 0.0
        prev_act = 0

        while not env.is_over():
            steps += 1
            if steps >= INJECTION_STEP and not shift_on:
                env.shift_config = obs_shift
                shift_on = True

            q_values = agent_seed1.model.predict_one(state)
            action = int(np.argmax(q_values))
            env.execute(action)
            q_len = float(env.get_queue_length())
            m_speed = float(env.get_incoming_mean_speed())
            next_state = env.get_state()
            next_q = agent_seed1.model.predict_one(next_state)
            cur_wait = env.get_cumulated_waiting_time()
            reward = env.compute_reward(prev_wait, cur_wait, prev_act, action)
            prev_wait = cur_wait
            prev_act = action

            traj.append({
                "step_idx": steps,
                "state": state.copy(),
                "action": action,
                "reward": float(reward),
                "next_state": next_state.copy(),
                "queue_len": q_len,
                "mean_speed": m_speed,
                "q_values": q_values.copy(),
                "next_q_values": next_q.copy(),
            })
            state = next_state
        env.deactivate()
        swept_trajectories[s] = traj
        print(f"  Swept Seed {s:2d}: {len(traj)} steps collected.")

    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"nominal": nominal_trajectories, "swept": swept_trajectories}, f)
    print(f"[Cache] Saved collected trajectories to {CACHE_FILE}")


# ---------------------------------------------------------------------------
# 3. Evaluate Grid of (Multiplier, Persistence)
# ---------------------------------------------------------------------------
def run_monitor_on_trajectory(traj, profile):
    monitor = ShiftMonitor(profile=profile)
    first_detect_step = None
    for idx, step in enumerate(traj):
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
            first_detect_step = step.get("step_idx", idx + 1)
    
    total_windows = len(monitor.history_alarms)
    alarm_count = sum(monitor.history_alarms)
    fpr = alarm_count / total_windows if total_windows > 0 else 0.0
    return {
        "total_windows": total_windows,
        "alarm_count": alarm_count,
        "fpr": fpr,
        "has_alarm": (alarm_count > 0),
        "first_detect_step": first_detect_step,
    }

grid_results = []

for m in multipliers:
    for p in persistences:
        thresh = mu + m * sigma
        prof = BaselineProfile(
            window_size=calib_profile.window_size,
            feature_means=calib_profile.feature_means,
            feature_stds=calib_profile.feature_stds,
            anomaly_score_mean=calib_profile.anomaly_score_mean,
            anomaly_score_std=calib_profile.anomaly_score_std,
            threshold_multiplier=m,
            detection_threshold=thresh,
            consecutive_steps_required=p,
        )

        # A. Nominal evaluation
        nom_windows_all = 0
        nom_alarms_all = 0
        nom_windows_heldout = 0
        nom_alarms_heldout = 0
        nom_episodes_with_alarm = 0

        for seed, traj in nominal_trajectories.items():
            res = run_monitor_on_trajectory(traj, prof)
            nom_windows_all += res["total_windows"]
            nom_alarms_all += res["alarm_count"]
            if res["has_alarm"]:
                nom_episodes_with_alarm += 1
            if seed in [4, 5]:
                nom_windows_heldout += res["total_windows"]
                nom_alarms_heldout += res["alarm_count"]

        overall_nominal_fpr = nom_alarms_all / nom_windows_all if nom_windows_all > 0 else 0.0
        heldout_nominal_fpr = nom_alarms_heldout / nom_windows_heldout if nom_windows_heldout > 0 else 0.0

        # B. Swept seeds detection
        detected_count = 0
        latencies = []
        detected_seed_list = []

        for s, traj in swept_trajectories.items():
            res = run_monitor_on_trajectory(traj, prof)
            if res["first_detect_step"] is not None and res["first_detect_step"] >= INJECTION_STEP:
                detected_count += 1
                lat = res["first_detect_step"] - INJECTION_STEP
                latencies.append(lat)
                detected_seed_list.append(s)

        mean_lat = float(np.mean(latencies)) if latencies else None

        grid_results.append({
            "multiplier_sigma": m,
            "persistence_steps": p,
            "threshold_val": round(thresh, 4),
            "nominal_fpr_all_seeds": round(overall_nominal_fpr * 100, 3),
            "nominal_alarms_all": nom_alarms_all,
            "nominal_windows_all": nom_windows_all,
            "nominal_fpr_heldout_4_5": round(heldout_nominal_fpr * 100, 3),
            "nominal_alarms_heldout": nom_alarms_heldout,
            "nominal_windows_heldout": nom_windows_heldout,
            "nominal_episodes_alarmed": f"{nom_episodes_with_alarm}/5",
            "swept_detection_count": f"{detected_count}/10",
            "swept_detection_pct": round((detected_count / 10) * 100, 1),
            "swept_detected_seeds": detected_seed_list,
            "swept_mean_latency_steps": round(mean_lat, 1) if mean_lat is not None else "N/A",
        })

print("\n" + "=" * 90)
print("MONITOR SENSITIVITY GRID EVALUATION (REPORT ONLY)")
print("Label: exploratory only, these seeds are exposed")
print("Notice: The final threshold must not be chosen from this table.")
print("=" * 90)
print(f"{'Multiplier':>10} | {'Persist':>7} | {'Threshold':>9} | {'Nom FPR (All)':>13} | {'Nom FPR (4-5)':>13} | {'Nom Ep Alarm':>12} | {'Swept Detect':>12} | {'Swept Latency':>13}")
print("-" * 90)
for r in grid_results:
    print(f"{r['multiplier_sigma']:>6.1f} sigma | {r['persistence_steps']:>7d} | {r['threshold_val']:>9.4f} | {r['nominal_fpr_all_seeds']:>12.2f}% | {r['nominal_fpr_heldout_4_5']:>12.2f}% | {r['nominal_episodes_alarmed']:>12s} | {r['swept_detection_count']:>12s} | {str(r['swept_mean_latency_steps']):>13s}")

out_json = ROOT / "results" / "final_checks" / "monitor_grid_evaluation.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({
        "label": "exploratory only, these seeds are exposed",
        "advisory": "The final threshold must not be chosen from this table.",
        "grid": grid_results,
    }, f, indent=2)

print("\nSaved grid evaluation artifact to:", out_json)
