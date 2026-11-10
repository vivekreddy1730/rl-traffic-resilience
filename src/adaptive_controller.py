"""
Adaptive Controller Module for Closed-Loop Traffic Signal Control (Phase 3 Pilot).

Chains:
  1. Online Shift Detection Monitor (R2)
  2. Root-Cause Diagnosis Classifier (R3)
  3. Targeted Mitigation Modules (R4)
  4. Fallback Governor (Fail-Safe Max-Pressure)
automatically in real time during live simulation episodes.
"""

import sys
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np

# Repo & local src imports
REPO_DIR = Path(__file__).parent.parent / "Deep-QLearning-Agent-for-Traffic-Signal-Control"
SRC_DIR = REPO_DIR / "src"
OUR_SRC = Path(__file__).parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(OUR_SRC) not in sys.path:
    sys.path.insert(0, str(OUR_SRC))

from tlcs.agent import Agent
from monitor import ShiftMonitor, BaselineProfile
from diagnosis import DiagnosisClassifier, extract_diagnostic_feature_vector, CLASS_NAMES
from mitigation import (
    SpatialTemporalConsensusFilter,
    VelocityAdaptivePhaseStretcher,
    PredictiveLookaheadMitigator,
    FallbackGovernor,
)


class AdaptiveController:
    """
    Autonomous closed-loop controller that monitors operational telemetry,
    diagnoses active distribution shifts, applies targeted mitigation,
    and falls back to fail-safe lane-level Max-Pressure if anomalies escalate.
    """

    def __init__(
        self,
        agent: Agent,
        profile_path: Path,
        classifier_path: Path,
        window_size: int = 50,
        min_diagnosis_samples: int = 15,
        severity_threshold: float = 2.5,
        queue_increase_window: int = 4,
        enable_action_mitigation: bool = False,
    ) -> None:
        self.agent = agent
        self.profile = BaselineProfile.load(profile_path)
        self.monitor = ShiftMonitor(self.profile)
        self.diagnoser = DiagnosisClassifier.load(classifier_path)
        self.fallback_governor = FallbackGovernor(
            severity_threshold=severity_threshold,
            queue_increase_window=queue_increase_window,
        )

        # Mitigation actuators
        self.obs_filter = SpatialTemporalConsensusFilter()
        self.vaps = VelocityAdaptivePhaseStretcher()
        self.action_mit = PredictiveLookaheadMitigator(agent=agent)
        # Empirical finding: predictive lookahead achieved -46.62% recovery ratio (net harmful).
        # Disabled by default to prevent performance regression in live deployment.
        self.enable_action_mitigation = enable_action_mitigation

        self.window_size = window_size
        self.min_diagnosis_samples = min_diagnosis_samples

        # State tracking
        self.window_records: List[Dict[str, Any]] = []
        self.prev_state: Optional[np.ndarray] = None
        self.prev_action: int = 0
        self.prev_q_values: Optional[np.ndarray] = None
        self.total_steps: int = 0

        # Current regime
        self.active_diagnosis: str = "Nominal"
        self.is_fallback_active: bool = False
        self.last_anomaly_score: float = 0.0

        # Diagnostics telemetry log
        self.history: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Reset internal telemetry and controller state for a new episode."""
        self.monitor.reset()
        self.fallback_governor.reset()
        self.obs_filter.reset()
        self.vaps.reset()
        self.action_mit.reset()
        self.window_records.clear()
        self.prev_state = None
        self.prev_action = 0
        self.prev_q_values = None
        self.total_steps = 0
        self.active_diagnosis = "Nominal"
        self.is_fallback_active = False
        self.last_anomaly_score = 0.0
        self.history.clear()

    def select_action(
        self,
        raw_state: np.ndarray,
        queue_len: float,
        mean_speed: float,
        free_flow_speed: float,
        prev_reward: float = 0.0,
        prev_wait_delta: float = 0.0,
    ) -> Tuple[int, int, Dict[str, Any]]:
        """
        Executes one full adaptive control cycle.

        Args:
            raw_state: Current raw environment observation vector (80 cells).
            queue_len: Current total halted queue length.
            mean_speed: Fleet average velocity (m/s).
            free_flow_speed: Approaching free-flow speed (m/s).
            prev_reward: Reward from previous step transition.
            prev_wait_delta: Change in cumulated waiting time from previous step.

        Returns:
            effective_action: The traffic light phase action index (0-3) to execute.
            extra_green_steps: Additional simulation steps to extend green phase (Transition mit).
            info: Diagnostics metadata dictionary.
        """
        self.total_steps += 1

        # -------------------------------------------------------------
        # 1. State Preprocessing (Observation Mitigation if active)
        # -------------------------------------------------------------
        if self.active_diagnosis == "Observation" and not self.is_fallback_active:
            effective_state = self.obs_filter.filter(raw_state)
        else:
            effective_state = raw_state.copy()

        # -------------------------------------------------------------
        # 2. DQN Value Inference & Raw Policy Action
        # -------------------------------------------------------------
        q_values = self.agent.model.predict_one(effective_state)
        raw_action = int(np.argmax(q_values))

        # -------------------------------------------------------------
        # 3. Monitor Telemetry Update & Anomaly Detection
        # -------------------------------------------------------------
        dummy_next_q = q_values  # Online approximation before transition executes
        score, anomaly_detected, metrics = self.monitor.update(
            state=raw_state,
            action=raw_action,
            reward=prev_reward,
            next_state=raw_state,
            queue_len=queue_len,
            mean_speed=mean_speed,
            q_values=q_values,
            next_q_values=dummy_next_q,
        )
        self.last_anomaly_score = score

        # Update telemetry window for diagnosis
        rec = {
            "state": raw_state,
            "next_state": raw_state if self.prev_state is None else self.prev_state,
            "action": raw_action,
            "reward": prev_reward,
            "queue_len": queue_len,
            "mean_speed": mean_speed,
            "free_flow_speed": free_flow_speed,
            "waiting_delta": prev_wait_delta,
            "q_values": q_values,
            "next_q_values": q_values,
            "executed_action": raw_action,
        }
        self.window_records.append(rec)
        if len(self.window_records) > self.window_size:
            self.window_records.pop(0)

        # -------------------------------------------------------------
        # 4. Root-Cause Diagnosis (if anomaly detected & enough samples)
        # -------------------------------------------------------------
        if anomaly_detected and len(self.window_records) >= self.min_diagnosis_samples:
            feat_vec = extract_diagnostic_feature_vector(self.window_records)
            diag_idx = int(self.diagnoser.predict(feat_vec.reshape(1, -1))[0])
            self.active_diagnosis = CLASS_NAMES[diag_idx]
        elif not anomaly_detected:
            self.active_diagnosis = "Nominal"

        # -------------------------------------------------------------
        # 5. Fallback Governor Check (Fail-Safe Override)
        # -------------------------------------------------------------
        self.is_fallback_active = self.fallback_governor.check_and_update(queue_len, score)

        # -------------------------------------------------------------
        # 6. Action Execution Selection & Mitigation
        # -------------------------------------------------------------
        extra_green_steps = 0

        if self.is_fallback_active:
            # Revert to fail-safe lane-level Max-Pressure
            effective_action = FallbackGovernor.get_maxpressure_action()
            mitigation_applied = "Fallback_MaxPressure"
        elif self.active_diagnosis == "Action":
            if self.enable_action_mitigation:
                # Apply predictive lookahead mitigator (EXPERIMENTAL / WARNING: measured rho = -46.62%)
                effective_action = self.action_mit.filter_action(
                    state=effective_state,
                    proposed_action=raw_action,
                    agent=self.agent,
                    step=self.total_steps,
                )
                mitigation_applied = "Predictive_Lookahead"
            else:
                # Gated: Predictive lookahead is known to degrade performance under stochastic arrivals (-46.62% recovery).
                # Retain raw policy action and rely on FallbackGovernor if queue accumulates.
                effective_action = raw_action
                mitigation_applied = "Action_Bypassed (empirical deficit -46.62%)"
        elif self.active_diagnosis == "Transition":
            # Apply velocity-adaptive green phase stretch
            effective_action = raw_action
            extra_green_steps = self.vaps.get_extra_green_duration(free_flow_speed)
            mitigation_applied = f"Phase_Stretch_{extra_green_steps}"
        elif self.active_diagnosis == "Observation":
            effective_action = raw_action
            mitigation_applied = "SpatialTemporal_Filter"
        else:
            effective_action = raw_action
            mitigation_applied = "None"

        # -------------------------------------------------------------
        # 7. Record History & Update Step State
        # -------------------------------------------------------------
        info = {
            "step": self.total_steps,
            "raw_action": raw_action,
            "effective_action": effective_action,
            "extra_green_steps": extra_green_steps,
            "anomaly_score": score,
            "anomaly_detected": anomaly_detected,
            "active_diagnosis": self.active_diagnosis,
            "is_fallback_active": self.is_fallback_active,
            "mitigation_applied": mitigation_applied,
            "queue_len": queue_len,
            "mean_speed": mean_speed,
        }
        self.history.append(info)

        self.prev_state = raw_state.copy()
        self.prev_action = effective_action
        self.prev_q_values = q_values.copy()

        return effective_action, extra_green_steps, info
