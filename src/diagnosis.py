"""
Diagnosis Module (R3) for root-cause classification of distribution shifts.

Classifies anomalous behavior into 5 distinct categories:
0: Nominal (No shift)
1: Observation Shift (Sensor noise / bit flips / lane dropout)
2: Transition Shift (Environmental dynamics / speed & deceleration changes)
3: Action Shift (Actuator latency queue / stuck actions)
4: Reward Shift (Modified objective functions / switch penalties)

Extracts domain-informed diagnostic signatures from transition streams:
- Spatial clustering / isolated-cell entropy (Observation)
- Free-flow velocity & deceleration ratio (Transition)
- Action-lag autocorrelation & mismatch (Action)
- Reward-to-waiting discrepancy (Reward)
- Calibrated baseline reference z-score distance (Nominal anchor)
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


CLASS_NAMES = ["Nominal", "Observation", "Transition", "Action", "Reward"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

# Nominal reference profile (calibrated strictly on Seeds 1-3)
NOMINAL_BASE_MEANS = {
    "mean_occupancy": 0.1059,
    "occupancy_delta": 0.1029,
    "mean_speed": 12.2889,
    "td_error": 30.4597,
    "reward": -0.0377,
    "queue_len": 2.3984,
}
NOMINAL_BASE_STDS = {
    "mean_occupancy": 0.0757,
    "occupancy_delta": 0.0628,
    "mean_speed": 0.8993,
    "td_error": 22.2585,
    "reward": 1.4112,
    "queue_len": 2.1768,
}


def compute_isolated_cells_ratio(state: np.ndarray) -> float:
    """
    Compute fraction of occupied cells that are completely isolated from adjacent cells.
    Bit-flip noise creates uniformly scattered isolated 1s across the 80 cells,
    whereas real vehicular traffic forms contiguous spatial clusters on incoming lanes.
    Each of the 4 incoming lane groups consists of 20 cells [0..19].
    """
    s = state.reshape(4, 20)
    total_active = float(np.sum(s))
    if total_active < 1e-5:
        return 0.0

    isolated_count = 0.0
    for g in range(4):
        lane = s[g]
        for c in range(20):
            if lane[c] > 0.5:
                left_neighbor = lane[c - 1] if c > 0 else 0.0
                right_neighbor = lane[c + 1] if c < 19 else 0.0
                if left_neighbor < 0.5 and right_neighbor < 0.5:
                    isolated_count += 1.0

    return isolated_count / total_active


def extract_diagnostic_feature_vector(
    window_records: List[Dict[str, Any]],
    gamma: float = 0.75,
) -> np.ndarray:
    """
    Extract multi-dimensional diagnostic feature signature from a rolling window.
    """
    occupancies = []
    occ_deltas = []
    isolated_ratios = []
    speeds = []
    free_flow_speeds = []
    td_errors = []
    q_spreads = []
    rewards = []
    queues = []
    reward_discrepancies = []
    action_mismatches = []
    action_lag1_matches = []
    action_switches = []

    for i in range(len(window_records)):
        rec = window_records[i]
        s = rec["state"]
        next_s = rec["next_state"]
        a = rec["action"]
        r = rec["reward"]
        qlen = rec["queue_len"]
        sp = rec.get("mean_speed", 13.89)
        ff_sp = rec.get("free_flow_speed", sp)
        base_wait_delta = rec.get("waiting_delta", r)
        q_v = np.asarray(rec["q_values"]).squeeze()
        next_q_v = np.asarray(rec["next_q_values"]).squeeze()

        # 1. State occupancy & spatial clustering
        occupancies.append(float(np.mean(s)))
        occ_deltas.append(float(np.mean(np.abs(next_s - s))))
        isolated_ratios.append(compute_isolated_cells_ratio(s))

        # 2. Dynamics & Speed
        speeds.append(float(sp))
        free_flow_speeds.append(float(ff_sp))
        queues.append(float(qlen))
        rewards.append(float(r))

        # 3. Reward discrepancy: |r_t - delta_W_t|
        reward_discrepancies.append(float(abs(r - base_wait_delta)))

        # 4. Value & TD features
        target_q = r + gamma * float(np.max(next_q_v))
        td_err = float(abs(target_q - q_v[a]))
        td_errors.append(td_err)
        q_spreads.append(float(np.max(q_v) - np.min(q_v)))

        # 5. Action lag & mismatch features
        greedy_action = int(np.argmax(q_v))
        action_mismatches.append(1.0 if a != greedy_action else 0.0)

        if i > 0:
            prev_greedy = int(np.argmax(np.asarray(window_records[i - 1]["q_values"]).squeeze()))
            action_lag1_matches.append(1.0 if a == prev_greedy else 0.0)
            prev_a = window_records[i - 1]["action"]
            action_switches.append(1.0 if a != prev_a else 0.0)
        else:
            action_lag1_matches.append(0.0)

    # Summary statistics
    mean_occ = float(np.mean(occupancies))
    mean_occ_delta = float(np.mean(occ_deltas))
    mean_sp = float(np.mean(speeds))
    mean_ff_sp = float(np.mean(free_flow_speeds))
    mean_td = float(np.mean(td_errors))
    mean_rew = float(np.mean(rewards))
    mean_qlen = float(np.mean(queues))

    # Compute baseline reference z-scores (anchored on Seeds 1-3 baseline profile)
    z_occ = (mean_occ - NOMINAL_BASE_MEANS["mean_occupancy"]) / NOMINAL_BASE_STDS["mean_occupancy"]
    z_sp = (mean_sp - NOMINAL_BASE_MEANS["mean_speed"]) / NOMINAL_BASE_STDS["mean_speed"]
    z_td = (mean_td - NOMINAL_BASE_MEANS["td_error"]) / NOMINAL_BASE_STDS["td_error"]
    z_rew = (mean_rew - NOMINAL_BASE_MEANS["reward"]) / NOMINAL_BASE_STDS["reward"]
    z_qlen = (mean_qlen - NOMINAL_BASE_MEANS["queue_len"]) / NOMINAL_BASE_STDS["queue_len"]
    
    # Composite anomaly distance from nominal
    z_dist = float(np.sqrt((z_occ**2 + z_sp**2 + z_td**2 + z_rew**2 + z_qlen**2) / 5.0))

    feat_vector = np.array([
        mean_occ,
        mean_occ_delta,
        float(np.mean(isolated_ratios)),           # Observation signature
        mean_sp,
        mean_ff_sp,                                # Transition free-flow speed
        float(mean_ff_sp / 13.89),                 # Transition speed ratio
        float(np.std(speeds)),
        mean_td,
        float(np.std(td_errors)),
        float(np.mean(q_spreads)),
        mean_rew,
        float(np.mean(reward_discrepancies)),      # Reward signature
        mean_qlen,
        float(np.mean(action_switches)) if action_switches else 0.0,
        float(np.mean(action_mismatches)),         # Action mismatch
        float(np.mean(action_lag1_matches)),       # Action lag correlation
        z_dist,                                    # Nominal anomaly anchor
        z_occ,
        z_sp,
        z_rew,
    ], dtype=np.float32)

    return feat_vector


class DiagnosisClassifier:
    """
    Classifier for diagnosing the root-cause shift class from live telemetry.
    """

    def __init__(self, n_estimators: int = 150, random_state: int = 42) -> None:
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=12,
            min_samples_leaf=2,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit scaler and classifier on training dataset (Seeds 1-3 only)."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict shift class IDs on evaluation dataset."""
        if not self.is_fitted:
            raise RuntimeError("Classifier is not fitted.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Classifier is not fitted.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self, file_path: Path) -> None:
        """Checkpoint classifier and scaler to disk."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "model": self.model,
            "is_fitted": self.is_fitted,
            "class_names": CLASS_NAMES,
        }, file_path)

    @classmethod
    def load(cls, file_path: Path) -> "DiagnosisClassifier":
        """Load checkpointed classifier from disk."""
        data = joblib.load(file_path)
        clf = cls()
        clf.scaler = data["scaler"]
        clf.model = data["model"]
        clf.is_fitted = data["is_fitted"]
        return clf
