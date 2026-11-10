"""
Verification script for Phase 2 Shift Injection Engine (R1).

Verifies each of the 4 shift types against nominal behavior with raw numerical assertions:
1. Observation Shift: Raw vs perturbed state comparison (bit-flip noise & dropout).
2. Transition Shift: Vehicle dynamics modification verification via TraCI API.
3. Action Shift: Actuator latency & action queue verification.
4. Reward Shift: Objective penalty and mathematical delta verification.
"""

import os
import sys
from pathlib import Path
import numpy as np

# Path + SUMO setup (portable; all paths are relative to this file - see src/project_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from project_paths import setup_all, SETTINGS_FILE, VIDALI_DIR as REPO_DIR, MODELS_DIR, PHASE2_DIR, ROOT_DIR, output_dir  # noqa: E402
setup_all()

import traci
from tlcs.settings import load_training_settings
from shifts import ShiftedEnvironment, ShiftConfig, ShiftType


def test_observation_shift():
    print("\n" + "=" * 50)
    print("  TEST 1: OBSERVATION SHIFT (Bit-flip & Dropout)")
    print("=" * 50)
    settings = load_training_settings(SETTINGS_FILE)
    
    target_p = 0.20
    cfg = ShiftConfig(
        shift_type=ShiftType.OBSERVATION,
        obs_bit_flip_prob=target_p,
        obs_dropout_lanes=[0],  # Zero out lane group 0 (cells 0-19)
    )
    
    env = ShiftedEnvironment(
        n_cars_generated=100,
        max_steps=200,
        yellow_duration=settings.yellow_duration,
        green_duration=settings.green_duration,
        turn_chance=settings.turn_chance,
        sumocfg_file=settings.sumocfg_file,
        gui=False,
        shift_config=cfg,
    )
    
    env.set_seed(42)
    env.generate_routefile(seed=42)
    env.activate()
    
    # Evaluate over 20 decision steps (N = 20 steps * 60 non-dropout cells = 1200 trials)
    total_diffs = 0
    total_evaluated_cells = 0
    all_lane0_zero = True
    
    for _ in range(20):
        env.execute(0)
        raw_state = env.get_raw_state()
        obs_state = env.get_state()
        
        # Check lane 0 dropout
        if not np.all(obs_state[0:20] == 0.0):
            all_lane0_zero = False
            
        # Accumulate bit flips across cells 20-79
        diffs = int(np.sum(raw_state[20:] != obs_state[20:]))
        total_diffs += diffs
        total_evaluated_cells += len(raw_state[20:])
        
    env.deactivate()
    
    empirical_rate = total_diffs / float(total_evaluated_cells)
    
    # Statistical tolerance band (99% confidence interval for Binomial(N=1200, p=0.20)):
    # std_error = sqrt(0.20 * 0.80 / 1200) = 0.0115 (1.15%)
    # 99% CI: [0.20 - 2.58*0.0115, 0.20 + 2.58*0.0115] = [17.0%, 23.0%]
    tol_low = 0.170
    tol_high = 0.230
    
    print(f"  Lane 0 Dropout (Cells 0-19 across 20 steps): {'PASSED (All 0.0)' if all_lane0_zero else 'FAILED'}")
    print(f"  Evaluated Sample Size: {total_evaluated_cells} cells across 20 simulation steps")
    print(f"  Total Bit Flips: {total_diffs}/{total_evaluated_cells}")
    print(f"  Empirical Bit-Flip Rate: {empirical_rate:.2%} (Target: {target_p:.1%})")
    print(f"  Pass/Fail Tolerance Band: [{tol_low:.1%}, {tol_high:.1%}] (99% confidence interval)")
    
    assert all_lane0_zero, "Dropout failed: non-zero values found in masked lane"
    assert tol_low <= empirical_rate <= tol_high, f"Empirical rate {empirical_rate:.2%} outside tolerance [{tol_low:.1%}, {tol_high:.1%}]"
    print("  [VERDICT] Observation Shift Injection: VERIFIED OK")


def test_transition_shift():
    print("\n" + "=" * 50)
    print("  TEST 2: TRANSITION SHIFT (Dynamics Modification)")
    print("=" * 50)
    settings = load_training_settings(SETTINGS_FILE)
    
    # Nominal run to read base edge max speed & decel
    env_nom = ShiftedEnvironment(
        n_cars_generated=100,
        max_steps=100,
        yellow_duration=settings.yellow_duration,
        green_duration=settings.green_duration,
        turn_chance=settings.turn_chance,
        sumocfg_file=settings.sumocfg_file,
        gui=False,
        shift_config=ShiftConfig(shift_type=ShiftType.NOMINAL),
    )
    env_nom.generate_routefile(seed=12345)
    env_nom.activate()
    vtypes = traci.vehicletype.getIDList()
    base_speed = traci.lane.getMaxSpeed("N2TL_0")
    base_decel = traci.vehicletype.getDecel(vtypes[0])
    env_nom.deactivate()
    print(f"  Nominal Dynamics: Lane MaxSpeed = {base_speed:.2f} m/s, Decel = {base_decel:.2f} m/s^2")
    
    # Shifted run (50% speed, 50% decel)
    cfg = ShiftConfig(
        shift_type=ShiftType.TRANSITION,
        trans_speed_factor=0.50,
        trans_decel_factor=0.50,
    )
    env_shift = ShiftedEnvironment(
        n_cars_generated=100,
        max_steps=100,
        yellow_duration=settings.yellow_duration,
        green_duration=settings.green_duration,
        turn_chance=settings.turn_chance,
        sumocfg_file=settings.sumocfg_file,
        gui=False,
        shift_config=cfg,
    )
    env_shift.generate_routefile(seed=12345)
    env_shift.activate()
    shift_speed = traci.lane.getMaxSpeed("N2TL_0")
    shift_decel = traci.vehicletype.getDecel(vtypes[0])
    env_shift.deactivate()
    
    print(f"  Shifted Dynamics: Lane MaxSpeed = {shift_speed:.2f} m/s, Decel = {shift_decel:.2f} m/s^2")
    print(f"  Speed Ratio: {shift_speed / base_speed:.2f} (Target: 0.50)")
    print(f"  Decel Ratio: {shift_decel / base_decel:.2f} (Target: 0.50)")
    
    assert abs(shift_speed / base_speed - 0.50) < 1e-3, "Speed scaling failed"
    assert abs(shift_decel / base_decel - 0.50) < 1e-3, "Decel scaling failed"
    print("  [VERDICT] Transition Shift Injection: VERIFIED OK")


def test_action_shift():
    print("\n" + "=" * 50)
    print("  TEST 3: ACTION SHIFT (Actuator Latency Queue)")
    print("=" * 50)
    settings = load_training_settings(SETTINGS_FILE)
    
    cfg = ShiftConfig(
        shift_type=ShiftType.ACTION,
        action_latency_steps=1,  # 1-step lag
    )
    env = ShiftedEnvironment(
        n_cars_generated=100,
        max_steps=200,
        yellow_duration=settings.yellow_duration,
        green_duration=settings.green_duration,
        turn_chance=settings.turn_chance,
        sumocfg_file=settings.sumocfg_file,
        gui=False,
        shift_config=cfg,
    )
    env.generate_routefile(seed=12345)
    env.activate()
    
    # Action 0 at t=0 -> should execute initial 0 (buffer fill)
    env.execute(0)
    executed_0 = env._last_executed_action
    print(f"  Step 1: Commanded Action 0 -> Executed: {executed_0}")
    
    # Action 2 at t=1 -> with 1-step lag, should execute previous Action 0
    env.execute(2)
    executed_1 = env._last_executed_action
    print(f"  Step 2: Commanded Action 2 -> Executed: {executed_1} (Expected 0 due to 1-step lag)")
    
    # Action 3 at t=2 -> should execute Action 2
    env.execute(3)
    executed_2 = env._last_executed_action
    print(f"  Step 3: Commanded Action 3 -> Executed: {executed_2} (Expected 2 due to 1-step lag)")
    
    env.deactivate()
    
    assert executed_1 == 0, f"Expected executed 0, got {executed_1}"
    assert executed_2 == 2, f"Expected executed 2, got {executed_2}"
    print("  [VERDICT] Action Shift Injection: VERIFIED OK")


def test_reward_shift():
    print("\n" + "=" * 50)
    print("  TEST 4: REWARD SHIFT (Switch & Queue Imbalance Penalty)")
    print("=" * 50)
    settings = load_training_settings(SETTINGS_FILE)
    
    cfg = ShiftConfig(
        shift_type=ShiftType.REWARD,
        reward_switch_penalty=10.0,
        reward_queue_penalty_weight=2.0,
    )
    env = ShiftedEnvironment(
        n_cars_generated=100,
        max_steps=100,
        yellow_duration=settings.yellow_duration,
        green_duration=settings.green_duration,
        turn_chance=settings.turn_chance,
        sumocfg_file=settings.sumocfg_file,
        gui=False,
        shift_config=cfg,
    )
    env.generate_routefile(seed=12345)
    env.activate()
    
    # Step 1
    env.execute(0)
    
    # Nominal reward: prev_wait - curr_wait = 0 - 5 = -5
    # With switch (0 -> 1): switch penalty = -10.0
    # Queue imbalance: calculated dynamically
    r = env.compute_reward(
        previous_total_wait=0.0,
        current_total_wait=5.0,
        previous_action=0,
        current_action=1,
    )
    env.deactivate()
    
    print(f"  Base Waiting Delta: -5.0")
    print(f"  Switch Penalty: -10.0")
    print(f"  Computed Shifted Reward: {r:.2f}")
    
    # Base (-5) - Switch (10) - Queue Imbalance (>=0) <= -15.0
    assert r <= -15.0, f"Expected reward <= -15.0, got {r}"
    print("  [VERDICT] Reward Shift Injection: VERIFIED OK")


if __name__ == "__main__":
    print("=" * 60)
    print("  RUNNING PHASE 2 SHIFT INJECTION ENGINE VERIFICATION (R1)")
    print("=" * 60)
    test_observation_shift()
    test_transition_shift()
    test_action_shift()
    test_reward_shift()
    print("\n" + "=" * 60)
    print("  ALL 4 SHIFT INJECTION TESTS PASSED!")
    print("=" * 60)
