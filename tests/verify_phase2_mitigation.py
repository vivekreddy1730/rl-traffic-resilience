"""
Comprehensive Verification and Evaluation Harness for Phase 2 Mitigation Module (R4).

Evaluates:
1. Fallback Governor calibration on Nominal data (confirming False Fallback Rate <= 0.5%).
2. Quantitative Recovery Ratios (rho) across all 4 shift classes on Held-Out Seeds 4 & 5.
3. Single-shot confirmatory evaluation on fresh, untouched Seed 7.
4. End-to-End pipeline misrouting resilience test (verifying whether misdiagnosed/unnecessary mitigation causes harm).
"""

import os
import sys
import json
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import numpy as np
import torch

# Path + SUMO setup (portable; all paths are relative to this file - see src/project_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from project_paths import setup_all, SETTINGS_FILE, VIDALI_DIR as REPO_DIR, MODELS_DIR, PHASE2_DIR, ROOT_DIR, output_dir  # noqa: E402
setup_all()

from tlcs.settings import load_training_settings
from tlcs.agent import Agent
from shifts import ShiftedEnvironment, ShiftConfig, ShiftType
from monitor import ShiftMonitor, BaselineProfile
from diagnosis import DiagnosisClassifier, extract_diagnostic_feature_vector
from mitigation import (
    SpatialTemporalConsensusFilter,
    VelocityAdaptivePhaseStretcher,
    PredictiveLookaheadMitigator,
    RecedingHorizonLookahead,
    FallbackGovernor,
    compute_recovery_ratio,
)

RESULTS_DIR = PHASE2_DIR  # official inputs (read-only)
OUT_DIR = output_dir(PHASE2_DIR)  # where this script writes; set RESULTS_OUT_DIR to protect official files
CHECKPOINT_DIR = MODELS_DIR  # frozen DQN weights: models/seed_N/trained_model.pt


def run_episode_simulation(
    env: ShiftedEnvironment,
    agent: Agent,
    seed: int,
    apply_mitigation: bool = False,
    mitigation_type: Optional[str] = None,
    monitor: Optional[ShiftMonitor] = None,
    fallback_governor: Optional[FallbackGovernor] = None,
    force_misroute_type: Optional[str] = None,
    corrupt_policy: bool = False,
) -> Dict[str, float]:
    """Execute a single episode simulation and return performance metrics."""
    env.set_seed(seed)
    env.generate_routefile(seed=seed)

    obs_filter = SpatialTemporalConsensusFilter()
    vaps = VelocityAdaptivePhaseStretcher()
    action_mit = PredictiveLookaheadMitigator(agent=agent)

    previous_total_wait = 0.0
    previous_action = 0
    waiting_times = []
    delays = []
    queues = []
    fallback_triggers = 0
    total_steps = 0

    env.activate()
    state = env.get_state()

    while not env.is_over():
        total_steps += 1
        current_queue = float(env.get_queue_length())
        mean_speed = float(env.get_incoming_mean_speed())
        free_flow_speed = float(env.get_incoming_free_flow_speed())

        # Determine effective state seen by agent
        agent_state = state
        active_mitigation = force_misroute_type if force_misroute_type else mitigation_type

        if apply_mitigation and active_mitigation == "Observation":
            agent_state = obs_filter.filter(state)

        q_raw = agent.model.predict_one(agent_state)
        q_values = -q_raw if corrupt_policy else q_raw
        chosen_action = int(np.argmax(q_values))

        # Check Fallback Governor
        is_fallback = False
        if fallback_governor and monitor:
            # Monitor anomaly check
            dummy_next_q = agent.model.predict_one(agent_state)
            score, _, _ = monitor.update(
                state, chosen_action, 0.0, state, current_queue, mean_speed, q_values, dummy_next_q
            )
            if apply_mitigation:
                is_fallback = fallback_governor.check_and_update(current_queue, score)

        if is_fallback:
            fallback_triggers += 1
            effective_action = FallbackGovernor.get_maxpressure_action()
        elif apply_mitigation and active_mitigation == "Action":
            effective_action = action_mit.filter_action(state, chosen_action, agent=agent, step=total_steps)
        else:
            effective_action = chosen_action

        # Apply transition green phase stretch if applicable
        if apply_mitigation and active_mitigation == "Transition":
            extra_steps = vaps.get_extra_green_duration(free_flow_speed)
            if extra_steps > 0:
                env.green_duration += extra_steps
                action_stats = env.execute(effective_action)
                env.green_duration -= extra_steps
            else:
                action_stats = env.execute(effective_action)
        else:
            action_stats = env.execute(effective_action)

        current_total_wait = env.get_cumulated_waiting_time()
        prev_wait_delta = previous_total_wait - current_total_wait
        reward = env.compute_reward(
            previous_total_wait=previous_total_wait,
            current_total_wait=current_total_wait,
            previous_action=previous_action,
            current_action=effective_action,
        )
        previous_total_wait = current_total_wait
        previous_action = effective_action

        waiting_times.append(current_total_wait)
        queues.append(current_queue)
        state = env.get_state()

    env.deactivate()

    # Calculate average waiting time per step and average queue length
    avg_wait = float(np.mean(waiting_times)) if waiting_times else 0.0
    avg_queue = float(np.mean(queues)) if queues else 0.0
    fallback_rate = (fallback_triggers / total_steps) if total_steps > 0 else 0.0

    return {
        "avg_waiting_time": avg_wait,
        "avg_queue_length": avg_queue,
        "fallback_rate": fallback_rate,
        "total_steps": total_steps,
    }


def run_mitigation_verification():
    print("=" * 65)
    print("  PHASE 2 (R4): MITIGATION MODULE VERIFICATION & EVALUATION")
    print("=" * 65)

    settings = load_training_settings(SETTINGS_FILE)
    profile = BaselineProfile.load(RESULTS_DIR / "calibration_profile.json")

    # 1. Fallback Governor Calibration on Nominal Seeds 1-3
    print("\n[Step 1] Calibrating Fallback Governor on Nominal Seeds 1-3...")
    governor = FallbackGovernor(severity_threshold=2.5, queue_increase_window=4)

    nominal_fallback_rates = []
    for s in [1, 2, 3]:
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
        agent = Agent(settings=settings, epsilon=0.0, model_path=CHECKPOINT_DIR / f"seed_{s}")
        monitor = ShiftMonitor(profile=profile)
        governor.reset()
        res = run_episode_simulation(
            env, agent, ep_seed, apply_mitigation=True, monitor=monitor, fallback_governor=governor
        )
        nominal_fallback_rates.append(res["fallback_rate"])
        print(f"  Nominal Seed {s}: False Fallback Rate = {res['fallback_rate']:.2%}")

    mean_false_fallback = float(np.mean(nominal_fallback_rates))
    print(f"  Mean False Fallback Rate: {mean_false_fallback:.2%} (Target <= 0.50%)")
    assert mean_false_fallback <= 0.005, "False fallback rate exceeds 0.5%"

    # 2. Evaluate Recovery Ratios on Held-Out Seeds 4 & 5
    print("\n[Step 2] Evaluating Recovery Ratios on Held-Out Seeds 4 & 5...")

    test_shifts = {
        "Observation": ShiftConfig(
            shift_type=ShiftType.OBSERVATION, obs_bit_flip_prob=0.20, obs_dropout_lanes=[0]
        ),
        "Transition": ShiftConfig(
            shift_type=ShiftType.TRANSITION, trans_speed_factor=0.50, trans_decel_factor=0.50
        ),
        "Action": ShiftConfig(
            shift_type=ShiftType.ACTION, action_latency_steps=1
        ),
        "Policy_Collapse_Fallback": ShiftConfig(
            shift_type=ShiftType.NOMINAL
        ),
    }

    test_seeds = [4, 5]
    results_records = []

    for shift_name, s_cfg in test_shifts.items():
        print(f"\n  --- Testing Shift: {shift_name} ---")
        nom_waits, unmit_waits, mit_waits = [], [], []
        is_collapse_test = (shift_name == "Policy_Collapse_Fallback")

        for s in test_seeds:
            ep_seed = s * 10000
            agent = Agent(settings=settings, epsilon=0.0, model_path=CHECKPOINT_DIR / f"seed_{s}")

            # 1. Nominal Run
            env_nom = ShiftedEnvironment(
                n_cars_generated=settings.n_cars_generated,
                max_steps=settings.max_steps,
                yellow_duration=settings.yellow_duration,
                green_duration=settings.green_duration,
                turn_chance=settings.turn_chance,
                sumocfg_file=settings.sumocfg_file,
                gui=False,
                shift_config=ShiftConfig(shift_type=ShiftType.NOMINAL),
            )
            res_nom = run_episode_simulation(env_nom, agent, ep_seed)
            nom_waits.append(res_nom["avg_waiting_time"])

            # 2. Unmitigated Shift Run
            env_unmit = ShiftedEnvironment(
                n_cars_generated=settings.n_cars_generated,
                max_steps=settings.max_steps,
                yellow_duration=settings.yellow_duration,
                green_duration=settings.green_duration,
                turn_chance=settings.turn_chance,
                sumocfg_file=settings.sumocfg_file,
                gui=False,
                shift_config=s_cfg,
            )
            res_unmit = run_episode_simulation(
                env_unmit, agent, ep_seed, apply_mitigation=False, corrupt_policy=is_collapse_test
            )
            unmit_waits.append(res_unmit["avg_waiting_time"])

            # 3. Mitigated Shift Run
            env_mit = ShiftedEnvironment(
                n_cars_generated=settings.n_cars_generated,
                max_steps=settings.max_steps,
                yellow_duration=settings.yellow_duration,
                green_duration=settings.green_duration,
                turn_chance=settings.turn_chance,
                sumocfg_file=settings.sumocfg_file,
                gui=False,
                shift_config=s_cfg,
            )
            mit_type = "Observation" if "Observation" in shift_name else ("Transition" if "Transition" in shift_name else ("Action" if "Action" in shift_name else "Fallback"))
            monitor_inst = ShiftMonitor(profile=profile)
            gov_inst = FallbackGovernor(severity_threshold=2.5, queue_increase_window=4) if mit_type == "Fallback" else None

            res_mit = run_episode_simulation(
                env_mit,
                agent,
                ep_seed,
                apply_mitigation=True,
                mitigation_type=mit_type,
                monitor=monitor_inst,
                fallback_governor=gov_inst,
                corrupt_policy=is_collapse_test,
            )
            mit_waits.append(res_mit["avg_waiting_time"])

        mean_nom = float(np.mean(nom_waits))
        mean_unmit = float(np.mean(unmit_waits))
        mean_mit = float(np.mean(mit_waits))
        rho, rho_str = compute_recovery_ratio(mean_nom, mean_unmit, mean_mit)

        print(f"    Nominal Avg Wait:     {mean_nom:8.2f} s")
        print(f"    Unmitigated Avg Wait: {mean_unmit:8.2f} s")
        print(f"    Mitigated Avg Wait:   {mean_mit:8.2f} s")
        print(f"    Recovery Ratio (rho): {rho_str}")

        results_records.append({
            "shift_name": shift_name,
            "nominal_wait": mean_nom,
            "unmitigated_wait": mean_unmit,
            "mitigated_wait": mean_mit,
            "recovery_ratio": rho,
            "recovery_ratio_str": rho_str,
        })

    # 3. Single-Shot Confirmatory Evaluation on Fresh Seed 7
    print("\n[Step 3] Single-Shot Confirmatory Evaluation on Untouched Seed 7...")
    ep_seed_7 = 70000
    agent_7 = Agent(settings=settings, epsilon=0.0, model_path=CHECKPOINT_DIR / "seed_1")

    confirmatory_shift = ShiftConfig(
        shift_type=ShiftType.OBSERVATION, obs_bit_flip_prob=0.25, obs_dropout_lanes=[1]
    )

    env_s7_nom = ShiftedEnvironment(
        n_cars_generated=settings.n_cars_generated,
        max_steps=settings.max_steps,
        yellow_duration=settings.yellow_duration,
        green_duration=settings.green_duration,
        turn_chance=settings.turn_chance,
        sumocfg_file=settings.sumocfg_file,
        gui=False,
        shift_config=ShiftConfig(shift_type=ShiftType.NOMINAL),
    )
    s7_nom = run_episode_simulation(env_s7_nom, agent_7, ep_seed_7)["avg_waiting_time"]

    env_s7_unmit = ShiftedEnvironment(
        n_cars_generated=settings.n_cars_generated,
        max_steps=settings.max_steps,
        yellow_duration=settings.yellow_duration,
        green_duration=settings.green_duration,
        turn_chance=settings.turn_chance,
        sumocfg_file=settings.sumocfg_file,
        gui=False,
        shift_config=confirmatory_shift,
    )
    s7_unmit = run_episode_simulation(env_s7_unmit, agent_7, ep_seed_7, apply_mitigation=False)["avg_waiting_time"]

    env_s7_mit = ShiftedEnvironment(
        n_cars_generated=settings.n_cars_generated,
        max_steps=settings.max_steps,
        yellow_duration=settings.yellow_duration,
        green_duration=settings.green_duration,
        turn_chance=settings.turn_chance,
        sumocfg_file=settings.sumocfg_file,
        gui=False,
        shift_config=confirmatory_shift,
    )
    s7_mit = run_episode_simulation(env_s7_mit, agent_7, ep_seed_7, apply_mitigation=True, mitigation_type="Observation")["avg_waiting_time"]

    s7_rho, s7_rho_str = compute_recovery_ratio(s7_nom, s7_unmit, s7_mit)
    print(f"  Seed 7 Nominal:     {s7_nom:8.2f} s")
    print(f"  Seed 7 Unmitigated: {s7_unmit:8.2f} s")
    print(f"  Seed 7 Mitigated:   {s7_mit:8.2f} s")
    print(f"  Seed 7 Recovery Ratio: {s7_rho_str}")

    # 4. End-to-End Pipeline Misrouting Test
    print("\n[Step 4] End-to-End Pipeline Misrouting Resilience Check...")
    print("  Testing: Applying Action mitigation to an unperturbed Nominal episode (simulating Nominal misdiagnosed as Action)...")

    env_misroute_nom = ShiftedEnvironment(
        n_cars_generated=settings.n_cars_generated,
        max_steps=settings.max_steps,
        yellow_duration=settings.yellow_duration,
        green_duration=settings.green_duration,
        turn_chance=settings.turn_chance,
        sumocfg_file=settings.sumocfg_file,
        gui=False,
        shift_config=ShiftConfig(shift_type=ShiftType.NOMINAL),
    )
    res_clean = run_episode_simulation(env_misroute_nom, agent_7, ep_seed_7, apply_mitigation=False)

    env_misroute_act = ShiftedEnvironment(
        n_cars_generated=settings.n_cars_generated,
        max_steps=settings.max_steps,
        yellow_duration=settings.yellow_duration,
        green_duration=settings.green_duration,
        turn_chance=settings.turn_chance,
        sumocfg_file=settings.sumocfg_file,
        gui=False,
        shift_config=ShiftConfig(shift_type=ShiftType.NOMINAL),
    )
    res_misrouted = run_episode_simulation(
        env_misroute_act, agent_7, ep_seed_7, apply_mitigation=True, force_misroute_type="Action"
    )

    delta_misroute = res_misrouted["avg_waiting_time"] - res_clean["avg_waiting_time"]
    pct_impact = (delta_misroute / res_clean["avg_waiting_time"]) * 100.0
    print(f"  Unmitigated Clean Nominal Wait: {res_clean['avg_waiting_time']:8.2f} s")
    print(f"  Misrouted (Action Mit applied):  {res_misrouted['avg_waiting_time']:8.2f} s")
    print(f"  Misrouting Impact:              {delta_misroute:+8.2f} s ({pct_impact:+.2f}%)")

    # Checkpoint summary JSON
    summary_file = OUT_DIR / "mitigation_evaluation_summary_v2.json"
    with open(summary_file, "w") as f:
        json.dump({
            "mean_false_fallback_rate": mean_false_fallback,
            "held_out_shifts": results_records,
            "seed_7_confirmatory": {
                "nominal_wait": s7_nom,
                "unmitigated_wait": s7_unmit,
                "mitigated_wait": s7_mit,
                "recovery_ratio": s7_rho,
                "recovery_ratio_str": s7_rho_str,
            },
            "misrouting_check": {
                "clean_wait": res_clean["avg_waiting_time"],
                "misrouted_wait": res_misrouted["avg_waiting_time"],
                "delta": delta_misroute,
                "pct_impact": pct_impact,
            },
        }, f, indent=2)
    print(f"\n[Step 5] Checkpointed evaluation summary to: {summary_file}")
    print("\n" + "=" * 65)
    print("  PHASE 2 MITIGATION (R4) EVALUATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    run_mitigation_verification()
