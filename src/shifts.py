"""
Shift Injection Engine for Phase 2.

Implements parameterized distribution shifts across all 4 canonical MDP components:
1. Observation Shift (S-shift): Bit-flip noise, sensor dropout / blind lanes.
2. Transition Shift (P-shift): Friction / max speed reduction, lane capacity reduction, surge demand.
3. Action Shift (A-shift): Actuator lag / latency queue, stuck phase probability, action execution noise.
4. Reward Shift (R-shift): Multi-objective queue imbalance penalty, switch penalty, scaled rewards.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
import numpy as np
import traci

from tlcs.constants import STATE_SIZE, ACTION_TO_TL_PHASE, TRAFFIC_LIGHT_ID, INCOMING_EDGES
from tlcs.env import Environment, EnvStats
from tlcs.episode import Record


class ShiftType(str, Enum):
    NOMINAL = "nominal"
    OBSERVATION = "observation"
    TRANSITION = "transition"
    ACTION = "action"
    REWARD = "reward"


@dataclass
class ShiftConfig:
    shift_type: ShiftType = ShiftType.NOMINAL
    # Observation shift params
    obs_bit_flip_prob: float = 0.0          # Bernoulli bit-flip noise on 80-cell occupancy
    obs_dropout_lanes: List[int] = field(default_factory=list)  # Lane groups to zero out [0..3]
    
    # Transition shift params
    trans_speed_factor: float = 1.0         # Scale max speed of vehicles in SUMO (e.g. 0.5 for wet/icy)
    trans_decel_factor: float = 1.0         # Scale deceleration (e.g. 0.5 for reduced braking)
    
    # Action shift params
    action_latency_steps: int = 0           # Buffer delay in decision steps (e.g. 1 or 2 steps delay)
    action_stuck_prob: float = 0.0          # Probability of ignoring action and staying in current phase
    
    # Reward shift params
    reward_queue_penalty_weight: float = 0.0 # Weight for maximum lane queue disparity penalty
    reward_switch_penalty: float = 0.0       # Penalty for switching traffic light phase


class ShiftedEnvironment(Environment):
    """
    Environment wrapper that cleanly injects parameterized MDP distribution shifts.
    """

    def __init__(
        self,
        n_cars_generated: int,
        max_steps: int,
        yellow_duration: int,
        green_duration: int,
        turn_chance: float,
        sumocfg_file: Any,
        gui: bool,
        shift_config: Optional[ShiftConfig] = None,
    ) -> None:
        super().__init__(
            n_cars_generated=n_cars_generated,
            max_steps=max_steps,
            yellow_duration=yellow_duration,
            green_duration=green_duration,
            turn_chance=turn_chance,
            sumocfg_file=sumocfg_file,
            gui=gui,
        )
        self.shift_config = shift_config or ShiftConfig()
        self.rng = np.random.RandomState(42)
        
        # Internal buffers for action latency
        self._action_queue: List[int] = []
        self._last_executed_action: int = 0
        self._transition_applied: bool = False

    def set_seed(self, seed: int) -> None:
        """Seed internal random number generator for reproducible noise."""
        self.rng = np.random.RandomState(seed)

    def activate(self) -> None:
        """Activate SUMO and apply dynamic transition shifts if configured."""
        super().activate()
        self._action_queue.clear()
        self._last_executed_action = 0
        self._transition_applied = False
        
        # Apply vehicle dynamics transition shift if configured
        if self.shift_config.shift_type == ShiftType.TRANSITION:
            self._apply_transition_shift()

    def _apply_transition_shift(self) -> None:
        """Modify vehicle physical parameters via TraCI for transition shifts."""
        cfg = self.shift_config
        try:
            # Scale incoming edge speed limits (nominal 13.89 m/s = 50 km/h)
            if cfg.trans_speed_factor != 1.0:
                for edge in INCOMING_EDGES:
                    traci.edge.setMaxSpeed(edge, 13.89 * cfg.trans_speed_factor)

            vtypes = traci.vehicletype.getIDList()
            for vt in vtypes:
                if cfg.trans_speed_factor != 1.0:
                    traci.vehicletype.setSpeedFactor(vt, cfg.trans_speed_factor)
                if cfg.trans_decel_factor != 1.0:
                    orig_decel = traci.vehicletype.getDecel(vt)
                    traci.vehicletype.setDecel(vt, orig_decel * cfg.trans_decel_factor)
            self._transition_applied = True
        except Exception as e:
            print(f"[ShiftEngine Warning] Failed to modify TraCI vehicle dynamics: {e}")

    def get_raw_state(self) -> np.ndarray:
        """Return the unperturbed ground-truth state from the simulator."""
        return super().get_state()

    def get_state(self) -> np.ndarray:
        """
        Return the state seen by the agent, applying observation shifts if configured.
        """
        state = super().get_state().copy()
        
        if self.shift_config.shift_type == ShiftType.OBSERVATION:
            cfg = self.shift_config
            # 1. Bit-flip noise
            if cfg.obs_bit_flip_prob > 0.0:
                flips = self.rng.rand(STATE_SIZE) < cfg.obs_bit_flip_prob
                state = np.where(flips, 1.0 - state, state)
            
            # 2. Sensor dropout / blind lane groups (each group has 20 cells: 0-19, 20-39, etc.)
            for group_idx in cfg.obs_dropout_lanes:
                start_idx = group_idx * 20
                end_idx = start_idx + 20
                if 0 <= start_idx < STATE_SIZE and end_idx <= STATE_SIZE:
                    state[start_idx:end_idx] = 0.0
                    
        return state

    def get_incoming_mean_speed(self) -> float:
        """Compute average speed across all incoming edges (in m/s)."""
        speeds = []
        for edge in INCOMING_EDGES:
            try:
                sp = float(traci.edge.getLastStepMeanSpeed(edge))
                n_veh = traci.edge.getLastStepVehicleNumber(edge)
                if n_veh > 0:
                    speeds.append(sp)
            except Exception:
                pass
        return float(np.mean(speeds)) if speeds else 13.89

    def get_incoming_free_flow_speed(self) -> float:
        """Compute average speed of moving vehicles (>1.0 m/s) on incoming edges."""
        moving_speeds = []
        for edge in INCOMING_EDGES:
            try:
                veh_ids = traci.edge.getLastStepVehicleIDs(edge)
                for vid in veh_ids:
                    v_spd = float(traci.vehicle.getSpeed(vid))
                    if v_spd > 1.0:
                        moving_speeds.append(v_spd)
            except Exception:
                pass
        return float(np.mean(moving_speeds)) if moving_speeds else 13.89

    def execute(self, action: int) -> list[EnvStats]:
        """
        Execute action with action shift injection (latency or stuck phase).
        """
        cfg = self.shift_config
        effective_action = action

        if cfg.shift_type == ShiftType.ACTION:
            # 1. Action stuck probability
            if cfg.action_stuck_prob > 0.0 and self.rng.rand() < cfg.action_stuck_prob:
                effective_action = self._last_executed_action
            # 2. Action latency queue
            elif cfg.action_latency_steps > 0:
                self._action_queue.append(action)
                if len(self._action_queue) <= cfg.action_latency_steps:
                    effective_action = self._last_executed_action
                else:
                    effective_action = self._action_queue.pop(0)

        self._last_executed_action = effective_action
        return super().execute(effective_action)

    def compute_reward(self, previous_total_wait: float, current_total_wait: float, previous_action: int, current_action: int) -> float:
        """
        Compute reward, applying reward shift modifications if configured.
        """
        # Nominal reward: -Δ(cumulative waiting time)
        base_reward = previous_total_wait - current_total_wait
        
        if self.shift_config.shift_type == ShiftType.REWARD:
            cfg = self.shift_config
            reward = base_reward
            
            # 1. Penalty for switching phase (encourages phase stability / reduces yellow phases)
            if cfg.reward_switch_penalty > 0.0 and previous_action != current_action:
                reward -= cfg.reward_switch_penalty
                
            # 2. Queue imbalance penalty
            if cfg.reward_queue_penalty_weight > 0.0:
                queues = [
                    traci.edge.getLastStepHaltingNumber("N2TL"),
                    traci.edge.getLastStepHaltingNumber("S2TL"),
                    traci.edge.getLastStepHaltingNumber("E2TL"),
                    traci.edge.getLastStepHaltingNumber("W2TL"),
                ]
                imbalance = float(max(queues) - min(queues))
                reward -= cfg.reward_queue_penalty_weight * imbalance
                
            return reward
            
        return base_reward


def run_shifted_episode(
    env: ShiftedEnvironment,
    agent: Any,
    seed: int,
) -> tuple[list[Record], list[EnvStats]]:
    """
    Run one episode in the ShiftedEnvironment and record transitions with reward computation.
    """
    env.set_seed(seed)
    env.generate_routefile(seed=seed)

    previous_total_wait = 0.0
    previous_action = 0
    history: list[Record] = []
    env_stats: list[EnvStats] = []

    env.activate()

    while not env.is_over():
        state = env.get_state()
        action = agent.choose_action(state)

        action_stats = env.execute(action)
        env_stats.extend(action_stats)

        current_total_wait = env.get_cumulated_waiting_time()
        reward = env.compute_reward(
            previous_total_wait=previous_total_wait,
            current_total_wait=current_total_wait,
            previous_action=previous_action,
            current_action=action,
        )
        previous_total_wait = current_total_wait
        previous_action = action

        record = Record(state=state, action=action, reward=reward)
        history.append(record)

    env.deactivate()
    return history, env_stats
