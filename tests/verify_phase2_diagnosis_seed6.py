"""
Single-Shot Confirmatory Evaluation of Frozen Diagnosis Classifier (R3) on Seed 6.

Zero tuning. Evaluates pure out-of-sample generalization on:
- Seed 6 (ep_seed = 60000)
- Novel Shift Severities:
  1. Nominal: Unperturbed
  2. Observation: p=0.30 bit-flips + Lane 0 dropout
  3. Transition: 65% max speed (9.0 m/s) + 65% decel (1.95 m/s^2)
  4. Action: 3-step actuator latency queue (k=3)
  5. Reward: Switch penalty 12.0 + Queue imbalance weight 2.5
"""

import os
import sys
import json
from typing import Any, List, Dict, Tuple
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Path + SUMO setup (portable; all paths are relative to this file - see src/project_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from project_paths import setup_all, SETTINGS_FILE, VIDALI_DIR as REPO_DIR, MODELS_DIR, PHASE2_DIR, ROOT_DIR, output_dir  # noqa: E402
setup_all()

from tlcs.settings import load_training_settings
from tlcs.agent import Agent
from shifts import ShiftedEnvironment, ShiftConfig, ShiftType
from diagnosis import (
    DiagnosisClassifier,
    extract_diagnostic_feature_vector,
    CLASS_NAMES,
    CLASS_TO_ID,
)

RESULTS_DIR = PHASE2_DIR  # official inputs (read-only)
OUT_DIR = output_dir(PHASE2_DIR)  # where this script writes; set RESULTS_OUT_DIR to protect official files
CHECKPOINT_DIR = MODELS_DIR  # frozen DQN weights: models/seed_N/trained_model.pt
MODEL_PATH = RESULTS_DIR / "diagnosis_classifier.joblib"


def collect_raw_trajectory(env: ShiftedEnvironment, agent: Agent, seed: int) -> list:
    """Run an episode and return list of step records."""
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
        free_flow_speed = float(env.get_incoming_free_flow_speed())

        next_state = env.get_state()
        next_q_values = agent.model.predict_one(next_state)

        current_total_wait = env.get_cumulated_waiting_time()
        waiting_delta = previous_total_wait - current_total_wait
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
            "waiting_delta": float(waiting_delta),
            "next_state": next_state.copy(),
            "queue_len": queue_len,
            "mean_speed": mean_speed,
            "free_flow_speed": free_flow_speed,
            "q_values": q_values.copy(),
            "next_q_values": next_q_values.copy(),
        }
        trajectory.append(step_record)
        state = next_state

    env.deactivate()
    return trajectory


def run_seed6_confirmatory_evaluation():
    print("=" * 65)
    print("  PHASE 2 (R3): SINGLE-SHOT CONFIRMATORY EVALUATION (SEED 6)")
    print("=" * 65)

    settings = load_training_settings(SETTINGS_FILE)
    
    # Load frozen trained classifier
    if not MODEL_PATH.exists():
        print(f"[Error] Model checkpoint not found at: {MODEL_PATH}")
        sys.exit(1)
        
    clf = DiagnosisClassifier.load(MODEL_PATH)
    print(f"  Loaded frozen classifier from: {MODEL_PATH}")

    # Use trained Agent policy
    agent = Agent(settings=settings, epsilon=0.0, model_path=CHECKPOINT_DIR / "seed_1")
    agent.set_epsilon(0.0)

    # 5 Novel Evaluation Scenarios on Seed 6
    seed6_scenarios = [
        ("Nominal", ShiftConfig(shift_type=ShiftType.NOMINAL)),
        ("Observation", ShiftConfig(shift_type=ShiftType.OBSERVATION, obs_bit_flip_prob=0.30, obs_dropout_lanes=[0])),
        ("Transition", ShiftConfig(shift_type=ShiftType.TRANSITION, trans_speed_factor=0.65, trans_decel_factor=0.65)),
        ("Action", ShiftConfig(shift_type=ShiftType.ACTION, action_latency_steps=3)),
        ("Reward", ShiftConfig(shift_type=ShiftType.REWARD, reward_switch_penalty=12.0, reward_queue_penalty_weight=2.5)),
    ]

    ep_seed = 60000
    window_size = 30
    stride = 10

    X_test_list = []
    y_test_list = []

    print(f"\n[Step 1] Running Seed 6 (ep_seed={ep_seed}) across 5 shift scenarios...")
    for label_name, cfg in seed6_scenarios:
        class_id = CLASS_TO_ID[label_name]
        env = ShiftedEnvironment(
            n_cars_generated=settings.n_cars_generated,
            max_steps=settings.max_steps,
            yellow_duration=settings.yellow_duration,
            green_duration=settings.green_duration,
            turn_chance=settings.turn_chance,
            sumocfg_file=settings.sumocfg_file,
            gui=False,
            shift_config=cfg,
        )
        traj = collect_raw_trajectory(env, agent, seed=ep_seed)
        
        # Slicing into rolling windows
        n_windows = 0
        for start_idx in range(0, len(traj) - window_size + 1, stride):
            window = traj[start_idx : start_idx + window_size]
            feat = extract_diagnostic_feature_vector(window)
            X_test_list.append(feat)
            y_test_list.append(class_id)
            n_windows += 1
        print(f"  Scenario: {label_name:12s} -> {len(traj)} steps, {n_windows} test windows.")

    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.int64)

    # Predict with frozen model
    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    conf_mat = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASS_NAMES))))
    report_dict = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)

    print("\n" + "=" * 65)
    print("  CONFIRMATORY GENERALIZATION RESULTS (SEED 6)")
    print("=" * 65)
    print(f"\n  Overall Seed 6 Accuracy: {accuracy:.2%}")

    print("\n  Full Confusion Matrix (Rows: True Class, Columns: Predicted Class):")
    header = f"  {'':15s}" + "".join([f"{name:>12s}" for name in CLASS_NAMES])
    print(header)
    print("  " + "-" * (15 + 12 * len(CLASS_NAMES)))
    for i, name in enumerate(CLASS_NAMES):
        row_str = f"  {name:15s}" + "".join([f"{conf_mat[i, j]:12d}" for j in range(len(CLASS_NAMES))])
        print(row_str)

    print("\n  Classification Report (Precision / Recall / F1-Score):")
    print(report_str)

    # Save summary artifact
    confirmatory_path = OUT_DIR / "diagnosis_seed6_confirmatory.json"
    with open(confirmatory_path, "w") as f:
        json.dump({
            "seed": 6,
            "ep_seed": ep_seed,
            "overall_accuracy": accuracy,
            "confusion_matrix": conf_mat.tolist(),
            "class_names": CLASS_NAMES,
            "classification_report": report_dict,
        }, f, indent=2)
    print(f"\n[Step 2] Checkpointed Seed 6 confirmatory summary to: {confirmatory_path}")
    print("\n" + "=" * 65)
    print("  SEED 6 CONFIRMATORY EVALUATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    run_seed6_confirmatory_evaluation()
