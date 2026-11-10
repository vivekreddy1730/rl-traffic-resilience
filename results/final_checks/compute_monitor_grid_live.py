"""
Compute exact grid of (multiplier: 2.0, 2.5, 3.0 sigma) x (persistence: 1, 2, 3)
for:
1. Nominal runs (FPR) across Seeds 1-5 (both offline verify_phase2 mode and live mode)
2. The 10 swept seeds under mid-episode Observation shift with AdaptiveController

Output is saved to results/final_checks/monitor_grid_report.json and printed as table.
"""
import os
import sys
import json
import pickle
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
from adaptive_controller import AdaptiveController
from monitor import BaselineProfile, ShiftMonitor

SETTINGS_FILE = REPO / "settings" / "training_settings.yaml"
CHECKPOINT_DIR = MODELS_DIR  # frozen DQN weights: models/seed_N/trained_model.pt
CALIB_PROFILE = ROOT / "results" / "phase2_shifts" / "calibration_profile.json"
DIAG_CLASSIFIER = ROOT / "results" / "phase2_shifts" / "diagnosis_classifier.joblib"
INJECTION_STEP = 120

settings = load_training_settings(SETTINGS_FILE)
prof = BaselineProfile.load(CALIB_PROFILE)
mu = prof.anomaly_score_mean
sigma = prof.anomaly_score_std

multipliers = [2.0, 2.5, 3.0]
persistences = [1, 2, 3]

CACHE_FILE = ROOT / "results" / "final_checks" / "adaptive_scores_cache.pkl"

if CACHE_FILE.exists():
    print(f"Loading scores cache from {CACHE_FILE}...")
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)
    nominal_runs = cache["nominal"]
    swept_runs = cache["swept"]
else:
    print("Collecting live scores for nominal and swept seeds...")
    # 1. Nominal runs on seeds 1-5
    nominal_runs = {}
    for s in [1, 2, 3, 4, 5]:
        ep_seed = s * 10000
        agent = Agent(settings=settings, epsilon=0.0, model_path=CHECKPOINT_DIR / f"seed_{s}")
        ctrl = AdaptiveController(agent=agent, profile_path=CALIB_PROFILE, classifier_path=DIAG_CLASSIFIER)
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
        scores = []
        prev_wait = 0.0

        while not env.is_over():
            q_len = float(env.get_queue_length())
            m_speed = float(env.get_incoming_mean_speed())
            ff_speed = float(env.get_incoming_free_flow_speed())
            prev_delta = prev_wait - env.get_cumulated_waiting_time()
            act, extra, info = ctrl.select_action(
                raw_state=state,
                queue_len=q_len,
                mean_speed=m_speed,
                free_flow_speed=ff_speed,
                prev_reward=0.0,
                prev_wait_delta=prev_delta,
            )
            scores.append(info["anomaly_score"])
            env.execute(act)
            prev_wait = env.get_cumulated_waiting_time()
            state = env.get_state()
        env.deactivate()
        nominal_runs[s] = scores
        print(f"Nominal seed {s}: {len(scores)} steps, max score: {max(scores):.4f}")

    # 2. Swept runs on 10 seeds under Observation shift
    swept_runs = {}
    swept_seeds = [4, 5, 6, 7, 10, 11, 12, 13, 14, 15]
    obs_shift = ShiftConfig(shift_type=ShiftType.OBSERVATION, obs_bit_flip_prob=0.20, obs_dropout_lanes=[0])
    agent1 = Agent(settings=settings, epsilon=0.0, model_path=CHECKPOINT_DIR / "seed_1")

    for s in swept_seeds:
        ep_seed = s * 10000
        ctrl = AdaptiveController(agent=agent1, profile_path=CALIB_PROFILE, classifier_path=DIAG_CLASSIFIER)
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
        scores = []
        steps = 0
        shift_on = False
        prev_wait = 0.0

        while not env.is_over():
            steps += 1
            if steps >= INJECTION_STEP and not shift_on:
                env.shift_config = obs_shift
                shift_on = True
            q_len = float(env.get_queue_length())
            m_speed = float(env.get_incoming_mean_speed())
            ff_speed = float(env.get_incoming_free_flow_speed())
            prev_delta = prev_wait - env.get_cumulated_waiting_time()
            act, extra, info = ctrl.select_action(
                raw_state=state,
                queue_len=q_len,
                mean_speed=m_speed,
                free_flow_speed=ff_speed,
                prev_reward=0.0,
                prev_wait_delta=prev_delta,
            )
            scores.append(info["anomaly_score"])
            env.execute(act)
            prev_wait = env.get_cumulated_waiting_time()
            state = env.get_state()
        env.deactivate()
        swept_runs[s] = scores
        print(f"Swept seed {s:2d}: {len(scores)} steps, max score: {max(scores):.4f}")

    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"nominal": nominal_runs, "swept": swept_runs}, f)
    print(f"Saved scores cache to {CACHE_FILE}")

# Evaluate detection and FPR for any (threshold, persistence)
def evaluate_sequence(scores, threshold, persistence, min_step=0):
    consecutive = 0
    alarms = 0
    total_evaluated = 0
    first_detect_idx = None

    for idx, s in enumerate(scores):
        if idx < 10:  # warm-up
            continue
        total_evaluated += 1
        if s >= threshold:
            consecutive += 1
        else:
            consecutive = 0

        is_alarm = (consecutive >= persistence)
        if is_alarm:
            alarms += 1
            if first_detect_idx is None and idx >= min_step:
                first_detect_idx = idx + 1

    fpr = (alarms / total_evaluated) if total_evaluated > 0 else 0.0
    return {
        "total_windows": total_evaluated,
        "alarms": alarms,
        "fpr": fpr,
        "has_alarm": (alarms > 0),
        "first_detect_step": first_detect_idx,
    }

grid_output = []

for m in multipliers:
    for p in persistences:
        thresh = mu + m * sigma

        # Nominal evaluation
        nom_alarms_all = 0
        nom_windows_all = 0
        nom_alarms_heldout = 0
        nom_windows_heldout = 0
        nom_episodes_alarmed = 0

        for seed, scores in nominal_runs.items():
            res = evaluate_sequence(scores, thresh, p)
            nom_alarms_all += res["alarms"]
            nom_windows_all += res["total_windows"]
            if res["has_alarm"]:
                nom_episodes_alarmed += 1
            if seed in [4, 5]:
                nom_alarms_heldout += res["alarms"]
                nom_windows_heldout += res["total_windows"]

        nom_fpr_all = (nom_alarms_all / nom_windows_all * 100) if nom_windows_all > 0 else 0.0
        nom_fpr_heldout = (nom_alarms_heldout / nom_windows_heldout * 100) if nom_windows_heldout > 0 else 0.0

        # Swept seeds detection (injected at step 120)
        detected_count = 0
        detected_seeds = []
        latencies = []

        for s, scores in swept_runs.items():
            res = evaluate_sequence(scores, thresh, p, min_step=INJECTION_STEP)
            if res["first_detect_step"] is not None:
                detected_count += 1
                detected_seeds.append(s)
                lat = res["first_detect_step"] - INJECTION_STEP
                latencies.append(lat)

        mean_lat = float(np.mean(latencies)) if latencies else None

        row = {
            "multiplier_sigma": m,
            "persistence_steps": p,
            "threshold_val": round(thresh, 4),
            "nominal_fpr_all_seeds": round(nom_fpr_all, 3),
            "nominal_alarms_all": nom_alarms_all,
            "nominal_windows_all": nom_windows_all,
            "nominal_fpr_heldout_4_5": round(nom_fpr_heldout, 3),
            "nominal_alarms_heldout": nom_alarms_heldout,
            "nominal_windows_heldout": nom_windows_heldout,
            "nominal_episodes_alarmed": f"{nom_episodes_alarmed}/5",
            "swept_detection_count": f"{detected_count}/10",
            "swept_detection_pct": round(detected_count / 10 * 100, 1),
            "swept_detected_seeds": detected_seeds,
            "swept_mean_latency_steps": round(mean_lat, 1) if mean_lat is not None else "N/A",
        }
        grid_output.append(row)

print("\n" + "=" * 95)
print("TASK 4: MONITOR SENSITIVITY GRID EVALUATION (REPORT ONLY)")
print("Label: exploratory only, these seeds are exposed")
print("Notice: The final threshold must not be chosen from this table.")
print("=" * 95)
print(f"| Multiplier | Persist | Threshold | Nom FPR (Seeds 1-5) | Nom FPR (Seeds 4-5) | Nom Ep Alarms | Swept Detection | Swept Detected Seeds | Mean Latency |")
print(f"| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
for r in grid_output:
    print(f"| {r['multiplier_sigma']:.1f} sigma | {r['persistence_steps']} | {r['threshold_val']:.4f} | {r['nominal_fpr_all_seeds']:.2f}% ({r['nominal_alarms_all']}/{r['nominal_windows_all']}) | {r['nominal_fpr_heldout_4_5']:.2f}% ({r['nominal_alarms_heldout']}/{r['nominal_windows_heldout']}) | {r['nominal_episodes_alarmed']} | **{r['swept_detection_count']} ({r['swept_detection_pct']}%)** | {r['swept_detected_seeds']} | {str(r['swept_mean_latency_steps'])} steps |")

out_json = ROOT / "results" / "final_checks" / "monitor_grid_report.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({
        "label": "exploratory only, these seeds are exposed",
        "advisory": "The final threshold must not be chosen from this table.",
        "grid": grid_output,
    }, f, indent=2)
print(f"\nSaved report to: {out_json}")
