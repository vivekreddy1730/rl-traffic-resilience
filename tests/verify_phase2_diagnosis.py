"""
Verification and Evaluation script for Diagnosis Module (R3).

1. Generates training dataset solely from Seeds 1, 2, 3 across baseline parameter ranges.
2. Trains DiagnosisClassifier and checkpoints model to results/phase2_shifts/diagnosis_classifier.joblib.
3. Generates strictly held-out test dataset on Seeds 4 & 5 under unseen shift severities & configurations:
   - Nominal: Seeds 4 & 5
   - Observation: Unseen p=0.25 bit-flips, unseen Lane 2 & 3 dropouts
   - Transition: Unseen speed factor 0.40 (severe) and 0.75 (mild)
   - Action: Unseen 2-step latency queue, unseen 0.35 stuck prob
   - Reward: Unseen switch penalty 15.0, unseen queue weight 3.0
4. Computes and reports:
   - Overall accuracy (target >= 85%)
   - Full 5x5 confusion matrix
   - Per-class precision, recall, and F1-score
5. Checkpoints summary metrics to results/phase2_shifts/diagnosis_evaluation_summary.json.
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


def extract_dataset_from_scenarios(
    scenarios: list,
    seeds: list,
    settings: Any,
    window_size: int = 30,
    stride: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Run episodes for scenarios across specified seeds and extract windowed feature vectors."""
    X_list = []
    y_list = []

    for label_name, cfg in scenarios:
        class_id = CLASS_TO_ID[label_name]
        for seed in seeds:
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
                shift_config=cfg,
            )
            traj = collect_raw_trajectory(env, agent, seed=ep_seed)

            # Slice into rolling windows
            for start_idx in range(0, len(traj) - window_size + 1, stride):
                window = traj[start_idx : start_idx + window_size]
                feat = extract_diagnostic_feature_vector(window)
                X_list.append(feat)
                y_list.append(class_id)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


def run_diagnosis_verification():
    print("=" * 65)
    print("  PHASE 2 (R3): DIAGNOSIS MODULE TRAINING & HELD-OUT EVALUATION")
    print("=" * 65)

    settings = load_training_settings(SETTINGS_FILE)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Define Training Scenarios (Seeds 1, 2, 3)
    train_seeds = [1, 2, 3]
    train_scenarios = [
        ("Nominal", ShiftConfig(shift_type=ShiftType.NOMINAL)),
        ("Observation", ShiftConfig(shift_type=ShiftType.OBSERVATION, obs_bit_flip_prob=0.15, obs_dropout_lanes=[0])),
        ("Observation", ShiftConfig(shift_type=ShiftType.OBSERVATION, obs_bit_flip_prob=0.20, obs_dropout_lanes=[1])),
        ("Transition", ShiftConfig(shift_type=ShiftType.TRANSITION, trans_speed_factor=0.50, trans_decel_factor=0.50)),
        ("Transition", ShiftConfig(shift_type=ShiftType.TRANSITION, trans_speed_factor=0.60, trans_decel_factor=0.60)),
        ("Action", ShiftConfig(shift_type=ShiftType.ACTION, action_latency_steps=1)),
        ("Action", ShiftConfig(shift_type=ShiftType.ACTION, action_stuck_prob=0.20)),
        ("Reward", ShiftConfig(shift_type=ShiftType.REWARD, reward_switch_penalty=10.0, reward_queue_penalty_weight=2.0)),
        ("Reward", ShiftConfig(shift_type=ShiftType.REWARD, reward_switch_penalty=8.0, reward_queue_penalty_weight=1.5)),
    ]

    print(f"\n[Step 1] Collecting Training Data across Seeds {train_seeds}...")
    X_train, y_train = extract_dataset_from_scenarios(train_scenarios, train_seeds, settings)
    print(f"  Training Samples: {X_train.shape[0]} windows across {len(train_scenarios)} scenario configurations.")
    for c_id, c_name in enumerate(CLASS_NAMES):
        count = int(np.sum(y_train == c_id))
        print(f"    Class {c_id} ({c_name}): {count} samples")

    # 2. Train Diagnosis Classifier
    print("\n[Step 2] Training Diagnosis Classifier (RandomForest)...")
    clf = DiagnosisClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 3. Checkpoint Model Artifact
    model_save_path = OUT_DIR / "diagnosis_classifier.joblib"
    clf.save(model_save_path)
    print(f"  [Checkpoint] Saved trained classifier artifact to: {model_save_path}")

    # 4. Define Strictly Held-Out Test Scenarios (Seeds 4 & 5 + Unseen Severities)
    test_seeds = [4, 5]
    test_scenarios = [
        # Unseen Nominal
        ("Nominal", ShiftConfig(shift_type=ShiftType.NOMINAL)),
        # Unseen Observation severities & masks
        ("Observation", ShiftConfig(shift_type=ShiftType.OBSERVATION, obs_bit_flip_prob=0.25, obs_dropout_lanes=[2])),
        ("Observation", ShiftConfig(shift_type=ShiftType.OBSERVATION, obs_bit_flip_prob=0.10, obs_dropout_lanes=[3])),
        # Unseen Transition speed severities
        ("Transition", ShiftConfig(shift_type=ShiftType.TRANSITION, trans_speed_factor=0.40, trans_decel_factor=0.40)),
        ("Transition", ShiftConfig(shift_type=ShiftType.TRANSITION, trans_speed_factor=0.75, trans_decel_factor=0.75)),
        # Unseen Action latency & stuck severities
        ("Action", ShiftConfig(shift_type=ShiftType.ACTION, action_latency_steps=2)),
        ("Action", ShiftConfig(shift_type=ShiftType.ACTION, action_stuck_prob=0.35)),
        # Unseen Reward penalties
        ("Reward", ShiftConfig(shift_type=ShiftType.REWARD, reward_switch_penalty=15.0, reward_queue_penalty_weight=3.0)),
        ("Reward", ShiftConfig(shift_type=ShiftType.REWARD, reward_switch_penalty=5.0, reward_queue_penalty_weight=1.0)),
    ]

    print(f"\n[Step 3] Collecting Held-Out Evaluation Data across Seeds {test_seeds} (Unseen Severities)...")
    X_test, y_test = extract_dataset_from_scenarios(test_scenarios, test_seeds, settings)
    print(f"  Held-Out Test Samples: {X_test.shape[0]} windows.")

    # 5. Evaluate on Held-Out Test Data
    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    conf_mat = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASS_NAMES))))
    report_dict = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)

    print("\n" + "=" * 65)
    print("  HELD-OUT EVALUATION RESULTS (SEEDS 4 & 5 + UNSEEN SEVERITIES)")
    print("=" * 65)
    print(f"\n  Overall Held-Out Accuracy: {accuracy:.2%} (Target >= 85.0%)")
    print(f"  Verdict: {'PASSED (>= 85%)' if accuracy >= 0.85 else 'FAILED (< 85%)'}\n")

    print("  Full Confusion Matrix (Rows: True Class, Columns: Predicted Class):")
    header = f"  {'':15s}" + "".join([f"{name:>12s}" for name in CLASS_NAMES])
    print(header)
    print("  " + "-" * (15 + 12 * len(CLASS_NAMES)))
    for i, name in enumerate(CLASS_NAMES):
        row_str = f"  {name:15s}" + "".join([f"{conf_mat[i, j]:12d}" for j in range(len(CLASS_NAMES))])
        print(row_str)

    print("\n  Classification Report (Precision / Recall / F1-Score):")
    print(report_str)

    # Checkpoint summary to JSON
    summary_path = OUT_DIR / "diagnosis_evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "overall_accuracy": accuracy,
            "target_threshold": 0.85,
            "is_passed": accuracy >= 0.85,
            "confusion_matrix": conf_mat.tolist(),
            "class_names": CLASS_NAMES,
            "classification_report": report_dict,
            "train_seeds": train_seeds,
            "test_seeds": test_seeds,
            "model_path": str(model_save_path),
        }, f, indent=2)
    print(f"\n[Step 4] Checkpointed evaluation summary to: {summary_path}")

    assert accuracy >= 0.85, f"Held-out accuracy {accuracy:.2%} < 85.0%"
    print("\n" + "=" * 65)
    print("  DIAGNOSIS MODULE (R3) VERIFICATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    run_diagnosis_verification()
