"""
Shift Monitor (R2) for detecting distribution shifts in online RL control.

Maintains rolling statistics of:
1. State features: Mean occupancy, occupancy velocity flux (occupancy delta).
2. Transition / dynamics features: Mean approach speed, queue clearance dynamics.
3. Value features: Absolute TD-error / Bellman residual, Q-spread.
4. Performance features: Rolling reward, queue length.

Computes standardized anomaly scores against a calibrated nominal baseline profile.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path
import numpy as np


@dataclass
class BaselineProfile:
    """Statistical profile of nominal behavior under unperturbed conditions."""
    window_size: int = 50
    feature_means: Dict[str, float] = field(default_factory=dict)
    feature_stds: Dict[str, float] = field(default_factory=dict)
    anomaly_score_mean: float = 0.0
    anomaly_score_std: float = 1.0
    threshold_multiplier: float = 3.0
    detection_threshold: float = 3.0
    consecutive_steps_required: int = 3

    def save(self, file_path: Path) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "window_size": self.window_size,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "anomaly_score_mean": self.anomaly_score_mean,
            "anomaly_score_std": self.anomaly_score_std,
            "threshold_multiplier": self.threshold_multiplier,
            "detection_threshold": self.detection_threshold,
            "consecutive_steps_required": self.consecutive_steps_required,
        }
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, file_path: Path) -> "BaselineProfile":
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls(**data)


class ShiftMonitor:
    """
    Online statistical shift monitor.
    Processes stream of (s, a, r, s_next, queue_len, speed, q_values, q_next_values)
    and computes live anomaly metrics.
    """

    def __init__(
        self,
        profile: Optional[BaselineProfile] = None,
        window_size: int = 50,
        gamma: float = 0.75,
    ) -> None:
        self.profile = profile
        self.window_size = profile.window_size if profile else window_size
        self.gamma = gamma

        # Rolling feature buffers
        self._occupancy_buf: List[float] = []
        self._occ_delta_buf: List[float] = []
        self._speed_buf: List[float] = []
        self._td_error_buf: List[float] = []
        self._q_spread_buf: List[float] = []
        self._reward_buf: List[float] = []
        self._queue_buf: List[float] = []

        # Detection state
        self._consecutive_alarms = 0
        self.history_scores: List[float] = []
        self.history_alarms: List[bool] = []

    def reset(self) -> None:
        """Reset internal buffers for a new episode."""
        self._occupancy_buf.clear()
        self._occ_delta_buf.clear()
        self._speed_buf.clear()
        self._td_error_buf.clear()
        self._q_spread_buf.clear()
        self._reward_buf.clear()
        self._queue_buf.clear()
        self._consecutive_alarms = 0
        self.history_scores.clear()
        self.history_alarms.clear()

    def extract_step_features(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        queue_len: float,
        mean_speed: float,
        q_values: np.ndarray,
        next_q_values: np.ndarray,
    ) -> Dict[str, float]:
        """Compute instantaneous per-step features."""
        q_v = np.asarray(q_values).squeeze()
        next_q_v = np.asarray(next_q_values).squeeze()

        # 1. State occupancy & occupancy velocity flux
        mean_occ = float(np.mean(state))
        occ_delta = float(np.mean(np.abs(next_state - state)))
        
        # 2. Dynamics & Approach velocity
        sp = float(mean_speed)

        # 3. TD Error / Bellman residual
        target_q = reward + self.gamma * float(np.max(next_q_v))
        td_err = float(abs(target_q - q_v[action]))
        
        # 4. Q-spread (confidence/spread across actions)
        q_spread = float(np.max(q_v) - np.min(q_v))
        
        return {
            "mean_occupancy": mean_occ,
            "occupancy_delta": occ_delta,
            "mean_speed": sp,
            "td_error": td_err,
            "q_spread": q_spread,
            "reward": float(reward),
            "queue_len": float(queue_len),
        }

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        queue_len: float,
        mean_speed: float,
        q_values: np.ndarray,
        next_q_values: np.ndarray,
    ) -> Tuple[float, bool, Dict[str, float]]:
        """
        Ingest a step transition.
        Returns: (anomaly_score, is_shift_detected, rolling_feature_dict)
        """
        step_feat = self.extract_step_features(
            state, action, reward, next_state, queue_len, mean_speed, q_values, next_q_values
        )

        # Update rolling windows
        self._occupancy_buf.append(step_feat["mean_occupancy"])
        self._occ_delta_buf.append(step_feat["occupancy_delta"])
        self._speed_buf.append(step_feat["mean_speed"])
        self._td_error_buf.append(step_feat["td_error"])
        self._q_spread_buf.append(step_feat["q_spread"])
        self._reward_buf.append(step_feat["reward"])
        self._queue_buf.append(step_feat["queue_len"])

        if len(self._occupancy_buf) > self.window_size:
            self._occupancy_buf.pop(0)
            self._occ_delta_buf.pop(0)
            self._speed_buf.pop(0)
            self._td_error_buf.pop(0)
            self._q_spread_buf.pop(0)
            self._reward_buf.pop(0)
            self._queue_buf.pop(0)

        # Rolling feature averages
        rolling_features = {
            "mean_occupancy": float(np.mean(self._occupancy_buf)),
            "occupancy_delta": float(np.mean(self._occ_delta_buf)),
            "mean_speed": float(np.mean(self._speed_buf)),
            "td_error": float(np.mean(self._td_error_buf)),
            "q_spread": float(np.mean(self._q_spread_buf)),
            "reward": float(np.mean(self._reward_buf)),
            "queue_len": float(np.mean(self._queue_buf)),
        }

        # If no profile or not enough warm-up, return 0.0
        if self.profile is None or len(self._occupancy_buf) < 10:
            return 0.0, False, rolling_features

        # Compute standardized distance from nominal baseline profile
        z_sq_sum = 0.0
        n_features = 0
        for feat_name, val in rolling_features.items():
            if feat_name in self.profile.feature_means and feat_name in self.profile.feature_stds:
                mu = self.profile.feature_means[feat_name]
                sigma = self.profile.feature_stds[feat_name]
                if sigma > 1e-6:
                    z = (val - mu) / sigma
                    z_sq_sum += z ** 2
                    n_features += 1

        anomaly_score = float(np.sqrt(z_sq_sum / max(1, n_features)))
        
        # Check alarm against detection threshold
        raw_alarm = anomaly_score >= self.profile.detection_threshold
        if raw_alarm:
            self._consecutive_alarms += 1
        else:
            self._consecutive_alarms = 0

        is_detected = self._consecutive_alarms >= self.profile.consecutive_steps_required
        
        self.history_scores.append(anomaly_score)
        self.history_alarms.append(is_detected)

        return anomaly_score, is_detected, rolling_features


def calibrate_nominal_profile(
    nominal_trajectories: List[List[Dict[str, Any]]],
    window_size: int = 50,
    threshold_multiplier: float = 3.0,
    consecutive_steps_required: int = 3,
    gamma: float = 0.75,
) -> BaselineProfile:
    """
    Calibrate nominal baseline profile using multi-seed unperturbed trajectories.
    """
    rolling_records: Dict[str, List[float]] = {
        "mean_occupancy": [],
        "occupancy_delta": [],
        "mean_speed": [],
        "td_error": [],
        "q_spread": [],
        "reward": [],
        "queue_len": [],
    }

    for traj in nominal_trajectories:
        occ_buf, occ_delta_buf, sp_buf, td_buf, q_buf, rew_buf, qlen_buf = [], [], [], [], [], [], []
        for step in traj:
            state = step["state"]
            action = step["action"]
            reward = step["reward"]
            next_state = step["next_state"]
            queue_len = step["queue_len"]
            mean_speed = step["mean_speed"]
            q_values = step["q_values"]
            next_q_values = step["next_q_values"]

            q_v = np.asarray(q_values).squeeze()
            next_q_v = np.asarray(next_q_values).squeeze()

            target_q = reward + gamma * float(np.max(next_q_v))
            td_err = float(abs(target_q - q_v[action]))

            occ_buf.append(float(np.mean(state)))
            occ_delta_buf.append(float(np.mean(np.abs(next_state - state))))
            sp_buf.append(float(mean_speed))
            td_buf.append(td_err)
            q_buf.append(float(np.max(q_v) - np.min(q_v)))
            rew_buf.append(float(reward))
            qlen_buf.append(float(queue_len))

            if len(occ_buf) > window_size:
                occ_buf.pop(0)
                occ_delta_buf.pop(0)
                sp_buf.pop(0)
                td_buf.pop(0)
                q_buf.pop(0)
                rew_buf.pop(0)
                qlen_buf.pop(0)

            if len(occ_buf) >= 10:
                rolling_records["mean_occupancy"].append(float(np.mean(occ_buf)))
                rolling_records["occupancy_delta"].append(float(np.mean(occ_delta_buf)))
                rolling_records["mean_speed"].append(float(np.mean(sp_buf)))
                rolling_records["td_error"].append(float(np.mean(td_buf)))
                rolling_records["q_spread"].append(float(np.mean(q_buf)))
                rolling_records["reward"].append(float(np.mean(rew_buf)))
                rolling_records["queue_len"].append(float(np.mean(qlen_buf)))

    feature_means = {k: float(np.mean(v)) for k, v in rolling_records.items()}
    feature_stds = {k: float(np.std(v)) for k, v in rolling_records.items()}

    # 2. Compute anomaly score distribution under nominal data
    anomaly_scores: List[float] = []
    n_samples = len(rolling_records["mean_occupancy"])
    for i in range(n_samples):
        z_sq_sum = 0.0
        n_feats = 0
        for k in feature_means:
            mu = feature_means[k]
            sigma = feature_stds[k]
            if sigma > 1e-6:
                val = rolling_records[k][i]
                z = (val - mu) / sigma
                z_sq_sum += z ** 2
                n_feats += 1
        anomaly_scores.append(float(np.sqrt(z_sq_sum / max(1, n_feats))))

    score_mean = float(np.mean(anomaly_scores))
    score_std = float(np.std(anomaly_scores))
    detection_thresh = score_mean + threshold_multiplier * score_std

    return BaselineProfile(
        window_size=window_size,
        feature_means=feature_means,
        feature_stds=feature_stds,
        anomaly_score_mean=score_mean,
        anomaly_score_std=score_std,
        threshold_multiplier=threshold_multiplier,
        detection_threshold=detection_thresh,
        consecutive_steps_required=consecutive_steps_required,
    )
