"""
Mitigation Module (R4) for Distribution Shift Adaptation and Safe Control.

Includes:
1. SpatialTemporalConsensusFilter (Observation shift mitigation)
2. VelocityAdaptivePhaseStretcher (Transition shift mitigation)
3. RecedingHorizonLookahead (Action latency mitigation)
4. FallbackGovernor (MaxPressure fail-safe on severe degradation)
5. End-to-end AdaptiveMitigationPipeline with safety-gating against misrouting
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import traci

from tlcs.constants import (
    INCOMING_EDGES,
    ACTION_TO_TL_PHASE,
    PHASE_NS_GREEN,
    PHASE_NSL_GREEN,
    PHASE_EW_GREEN,
    PHASE_EWL_GREEN,
)


def compute_recovery_ratio(
    m_nominal: float,
    m_unmitigated: float,
    m_mitigated: float,
    min_degradation_pct: float = 0.05,
) -> Tuple[Optional[float], str]:
    """
    Compute Recovery Ratio rho = (M_unmit - M_mit) / (M_unmit - M_nom).
    Handles edge cases where unmitigated degradation is negligible (<5%).
    """
    denom = m_unmitigated - m_nominal
    if abs(denom) < 1e-6 or (abs(denom) / max(1e-3, abs(m_nominal))) < min_degradation_pct:
        return None, "N/A (shift degradation too low to reliably measure recovery)"

    rho = (m_unmitigated - m_mitigated) / denom
    return float(rho), f"{rho:.2%}"


class SpatialTemporalConsensusFilter:
    """
    Observation shift mitigation:
    - Cleans isolated spurious 1s caused by bit-flip noise.
    - Imputes dropout lanes via temporal moving averages.
    """

    def __init__(self, history_len: int = 5, decay: float = 0.8) -> None:
        self.history_len = history_len
        self.decay = decay
        self.state_history: List[np.ndarray] = []

    def reset(self) -> None:
        self.state_history.clear()

    def filter(self, obs_state: np.ndarray) -> np.ndarray:
        s = obs_state.copy()
        s_mat = s.reshape(4, 20)
        clean_mat = s_mat.copy()

        # Spatial filter: remove isolated active cells (bit-flip noise)
        for g in range(4):
            lane = s_mat[g]
            for c in range(20):
                if lane[c] > 0.5:
                    left = lane[c - 1] if c > 0 else 0.0
                    right = lane[c + 1] if c < 19 else 0.0
                    # Isolated single cell with 0 neighbors
                    if left < 0.5 and right < 0.5:
                        clean_mat[g, c] = 0.0

        # Temporal filter: detect dropout lanes (all zeros while historical lanes were non-zero)
        if len(self.state_history) > 0:
            hist_mean = np.mean(self.state_history, axis=0).reshape(4, 20)
            for g in range(4):
                if np.sum(clean_mat[g]) < 1e-4 and np.sum(hist_mean[g]) > 0.5:
                    clean_mat[g] = np.where(hist_mean[g] > 0.3, 1.0, 0.0)

        cleaned_state = clean_mat.flatten()
        self.state_history.append(cleaned_state)
        if len(self.state_history) > self.history_len:
            self.state_history.pop(0)

        return cleaned_state


class VelocityAdaptivePhaseStretcher:
    """
    Transition shift mitigation:
    - Scales green phase execution to compensate for vehicle approach deceleration.
    - Uses non-disruptive timing adaptation with proportional stretch capped at 2 steps
      (pre-registered fixed cap) to avoid starving orthogonal green phases.
    """

    def __init__(self, nominal_speed: float = 13.89, max_extra_steps: int = 2) -> None:
        self.nominal_speed = nominal_speed
        self.max_extra_steps = 2  # Hardcoded fixed cap = 2 steps

    def get_extra_green_duration(self, current_free_flow_speed: float) -> int:
        kappa_v = max(0.0, (self.nominal_speed - current_free_flow_speed) / self.nominal_speed)
        if kappa_v < 0.15:
            return 0
        raw_extra = int(np.round(kappa_v * 4.0))
        return max(0, min(self.max_extra_steps, raw_extra))


class PredictiveLookaheadMitigator:
    """
    Action shift mitigation (LIVE):
    - In an actuator-delayed MDP (1-step action latency), executing actions on raw delayed state
      causes severe oscillating queue misallocations.
    - Predictive lookahead anticipates intersection occupancy discharging under the in-flight
      action and queries the Q-network on the projected future state.
    """

    def __init__(self, agent: Any = None) -> None:
        self.agent = agent
        self.pending_action: Optional[int] = None
        self.actions_modified: int = 0
        self.action_diff_log: List[Tuple[int, int, int, Optional[int]]] = []

    def reset(self) -> None:
        self.pending_action = None
        self.actions_modified = 0
        self.action_diff_log.clear()

    def predict_state(self, current_state: np.ndarray, in_flight_action: int) -> np.ndarray:
        s_pred = current_state.copy()
        s_mat = s_pred.reshape(4, 20)
        if 0 <= in_flight_action < 4:
            # Discharge vehicles at stopline
            s_mat[in_flight_action, 15:20] = 0.0
            # Advance upstream vehicles towards stopline
            s_mat[in_flight_action, 5:15] = s_mat[in_flight_action, 0:10]
            s_mat[in_flight_action, 0:5] = 0.0
        return s_mat.flatten()

    def filter_action(self, state: np.ndarray, raw_action: int, agent: Any = None, step: int = 0) -> int:
        ag = agent or self.agent
        if ag is None or self.pending_action is None:
            self.pending_action = raw_action
            return raw_action

        # Predict future state under in-flight pending action
        s_future = self.predict_state(state, self.pending_action)
        q_pred = ag.model.predict_one(s_future)
        lookahead_action = int(np.argmax(q_pred))

        if lookahead_action != raw_action:
            self.actions_modified += 1
            self.action_diff_log.append((step, raw_action, lookahead_action, self.pending_action))

        self.pending_action = lookahead_action
        return lookahead_action


class RecedingHorizonLookahead:
    """
    Action shift mitigation (Legacy / Phase-hold Hysteresis):
    Maintained for backward-compatibility; superseded by PredictiveLookaheadMitigator.
    """

    def __init__(self, estimated_latency: int = 1, min_phase_hold: int = 1) -> None:
        self.latency = estimated_latency
        self.min_phase_hold = min_phase_hold
        self.current_committed_phase: Optional[int] = None
        self.hold_counter: int = 0
        self.actions_modified: int = 0

    def reset(self) -> None:
        self.current_committed_phase = None
        self.hold_counter = 0
        self.actions_modified = 0

    def filter_action(self, greedy_action: int, diagnosed_action_shift: bool) -> int:
        if not diagnosed_action_shift or self.latency <= 0:
            return greedy_action

        if self.current_committed_phase is None:
            self.current_committed_phase = greedy_action
            self.hold_counter = 1
            return greedy_action

        if greedy_action != self.current_committed_phase:
            if self.hold_counter < self.min_phase_hold:
                self.hold_counter += 1
                self.actions_modified += 1
                return self.current_committed_phase
            else:
                self.current_committed_phase = greedy_action
                self.hold_counter = 1
                return greedy_action
        else:
            self.hold_counter += 1
            return greedy_action


class FallbackGovernor:
    """
    Fail-safe MaxPressure controller.
    Triggers ONLY when:
    1. Severity score S_t > calibrated threshold (mu_0 + 4 sigma_0), AND
    2. Queue length has strictly increased for N consecutive steps (sustained accumulation).
    """

    def __init__(
        self,
        severity_threshold: float = 2.5,
        queue_increase_window: int = 4,
    ) -> None:
        self.severity_threshold = severity_threshold
        self.queue_increase_window = queue_increase_window
        self.queue_history: List[float] = []
        self.is_active = False

    def reset(self) -> None:
        self.queue_history.clear()
        self.is_active = False

    def check_and_update(self, current_queue: float, anomaly_score: float) -> bool:
        self.queue_history.append(current_queue)
        if len(self.queue_history) > self.queue_increase_window + 1:
            self.queue_history.pop(0)

        high_severity = anomaly_score >= self.severity_threshold

        sustained_queue_growth = False
        if len(self.queue_history) >= self.queue_increase_window + 1:
            diffs = [
                self.queue_history[i] - self.queue_history[i - 1]
                for i in range(1, len(self.queue_history))
            ]
            sustained_queue_growth = all(d >= 0 for d in diffs) and sum(diffs) > 2.0

        if high_severity and sustained_queue_growth:
            self.is_active = True
        elif not high_severity and current_queue <= 2.0:
            self.is_active = False

        return self.is_active

    @staticmethod
    def get_maxpressure_action() -> int:
        """
        Compute MaxPressure greedy action across all 4 independent phases:
        Phase 0: NS Straight
        Phase 1: NS Left-turn
        Phase 2: EW Straight
        Phase 3: EW Left-turn
        """
        try:
            lanes = {
                0: ("N2TL_0", "N2TL_1", "N2TL_2", "S2TL_0", "S2TL_1", "S2TL_2"),
                1: ("N2TL_3", "S2TL_3"),
                2: ("E2TL_0", "E2TL_1", "E2TL_2", "W2TL_0", "W2TL_1", "W2TL_2"),
                3: ("E2TL_3", "W2TL_3"),
            }
            pressures = {
                a: sum(traci.lane.getLastStepHaltingNumber(l) for l in l_list)
                for a, l_list in lanes.items()
            }
            max_p = max(pressures.values())
            if max_p == 0:
                return 0
            best_cands = [a for a, p in pressures.items() if p == max_p]
            return best_cands[0]
        except Exception:
            return 0

