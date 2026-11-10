"""
run_adaptive_demo.py - the full resilience pipeline on one episode.

    Monitor -> Diagnosis -> Mitigation -> Fallback

The trained DQN controls the junction. Part-way through the episode a
"distribution shift" is injected (something about the world changes that the
agent never saw in training). The AdaptiveController (src/adaptive_controller.py)
watches live telemetry, raises an alarm, guesses which kind of shift happened,
and applies the matching mitigation - or falls back to Max-Pressure if queues
keep growing.

This script uses the same episode loop, injection step (decision step 120),
shift settings and model (models/seed_1) as tests/run_phase3_pilot.py, so for
the pilot's seed/shift pairs the result can be compared with
results/phase3_pilot/phase3_pilot_summary.csv.

Shift types:
  observation  sensor noise: 20% of the 80 occupancy cells flipped + lane group 0 blind (pilot seed 7)
  transition   vehicle dynamics: 50% max speed and 50% braking (pilot seed 8)
  action       actuator lag: every signal command executes 1 decision late (pilot seed 9)
  reward       reward definition: switch penalty 10 + queue-imbalance penalty 2
               (the Phase 2 monitor test setting; NOT part of the Phase 3 pilot)

Examples:
    python run_adaptive_demo.py                       # observation shift, seed 7
    python run_adaptive_demo.py --shift transition
    python run_adaptive_demo.py --shift action --no-compare
"""

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from project_paths import (  # noqa: E402
    CALIB_PROFILE, DIAG_CLASSIFIER, ROOT_DIR, SETTINGS_FILE, checkpoint_dir, setup_all,
)

INJECTION_STEP = 120  # same as tests/run_phase3_pilot.py
DEFAULT_SEED = {"observation": 7, "transition": 8, "action": 9, "reward": 7}
PILOT_CSV = ROOT_DIR / "results" / "phase3_pilot" / "phase3_pilot_summary.csv"


def parse_args():
    p = argparse.ArgumentParser(
        description="Inject one distribution shift mid-episode and show what the adaptive "
                    "controller (Monitor -> Diagnosis -> Mitigation -> Fallback) detects and does.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--shift", choices=["observation", "transition", "action", "reward"],
                   default="observation", help="Which kind of shift to inject.")
    p.add_argument("--seed", type=int, default=None,
                   help="Traffic seed (default: the Phase 3 pilot seed for that shift: "
                        "observation=7, transition=8, action=9, reward=7).")
    p.add_argument("--no-compare", action="store_true",
                   help="Skip the second run with the plain (unprotected) DQN.")
    p.add_argument("--enable-action-mitigation", action="store_true",
                   help="Turn ON the experimental action mitigation. It is OFF by default because it "
                        "was measured as harmful (-46.62%% recovery, results/phase2_shifts).")
    p.add_argument("--quiet", action="store_true", help="Print only the summary.")
    p.add_argument("--show-sumo-warnings", action="store_true",
                   help="Show SUMO's own console warnings (hidden by default; they do not change results).")
    return p.parse_args()


def make_shift(name):
    from shifts import ShiftConfig, ShiftType
    if name == "observation":
        return ShiftConfig(shift_type=ShiftType.OBSERVATION, obs_bit_flip_prob=0.20, obs_dropout_lanes=[0])
    if name == "transition":
        return ShiftConfig(shift_type=ShiftType.TRANSITION, trans_speed_factor=0.50, trans_decel_factor=0.50)
    if name == "action":
        return ShiftConfig(shift_type=ShiftType.ACTION, action_latency_steps=1)
    return ShiftConfig(shift_type=ShiftType.REWARD, reward_switch_penalty=10.0, reward_queue_penalty_weight=2.0)


def run_episode(controller_type, seed, shift_name, settings, args):
    """Mirror of tests/run_phase3_pilot.py::run_episode (DQN_Unmitigated / DQN_Adaptive)."""
    import numpy as np
    from tlcs.agent import Agent
    from shifts import ShiftConfig, ShiftedEnvironment, ShiftType
    from adaptive_controller import AdaptiveController

    target_cfg = make_shift(shift_name)
    ep_seed = seed * 10000
    class QuietEnv(ShiftedEnvironment):
        """Same simulation; only hides SUMO console warnings (they do not affect results)."""

        def build_sumo_cmd(self):
            cmd = super().build_sumo_cmd()
            return cmd if args.show_sumo_warnings else cmd + ["--no-warnings", "true"]

    env = QuietEnv(
        n_cars_generated=settings.n_cars_generated, max_steps=settings.max_steps,
        yellow_duration=settings.yellow_duration, green_duration=settings.green_duration,
        turn_chance=settings.turn_chance, sumocfg_file=settings.sumocfg_file, gui=False,
        shift_config=ShiftConfig(shift_type=ShiftType.NOMINAL),
    )
    env.set_seed(ep_seed)
    env.generate_routefile(seed=ep_seed)
    agent = Agent(settings=settings, epsilon=0.0, model_path=checkpoint_dir(1))
    ctrl = None
    if controller_type == "DQN_Adaptive":
        ctrl = AdaptiveController(agent=agent, profile_path=CALIB_PROFILE, classifier_path=DIAG_CLASSIFIER,
                                  enable_action_mitigation=args.enable_action_mitigation)

    # The pilot loop passes prev_reward=0.0 to the monitor. A reward shift only changes the
    # reward signal, so for --shift reward we feed the real (shifted) reward instead -
    # otherwise the shift would be invisible to the monitor by construction.
    feed_reward = shift_name == "reward"
    waiting_times, queues = [], []
    previous_total_wait, previous_action, last_reward = 0.0, 0, 0.0
    total_steps, injected = 0, False

    env.activate()
    state = env.get_state()
    while not env.is_over():
        total_steps += 1
        current_queue = float(env.get_queue_length())
        mean_speed = float(env.get_incoming_mean_speed())
        free_flow_speed = float(env.get_incoming_free_flow_speed())

        if total_steps >= INJECTION_STEP and not injected:
            env.shift_config = target_cfg
            if target_cfg.shift_type == ShiftType.TRANSITION:
                env._apply_transition_shift()
            injected = True
            if not args.quiet and controller_type == "DQN_Adaptive":
                print(f"  [decision {total_steps:3d}, t={env.step:4d}s] >>> SHIFT INJECTED: {shift_name}")

        extra = 0
        if controller_type == "DQN_Unmitigated":
            action = int(np.argmax(agent.model.predict_one(state)))
        else:
            prev_wait_delta = previous_total_wait - env.get_cumulated_waiting_time()
            action, extra, info = ctrl.select_action(
                raw_state=state, queue_len=current_queue, mean_speed=mean_speed,
                free_flow_speed=free_flow_speed,
                prev_reward=last_reward if feed_reward else 0.0, prev_wait_delta=prev_wait_delta,
            )
            if not args.quiet:
                prev = ctrl.history[-2] if len(ctrl.history) > 1 else None
                changed = prev is None or (prev["active_diagnosis"], prev["is_fallback_active"]) != (
                    info["active_diagnosis"], info["is_fallback_active"])
                if changed and total_steps > 1:
                    print(f"  [decision {total_steps:3d}, t={env.step:4d}s] score={info['anomaly_score']:.2f} "
                          f"alarm={'YES' if info['anomaly_detected'] else 'no '} diagnosis={info['active_diagnosis']:<11s} "
                          f"fallback={'ON' if info['is_fallback_active'] else 'off'} action-taken={info['mitigation_applied']}")

        if extra > 0:
            env.green_duration += extra
            env.execute(action)
            env.green_duration -= extra
        else:
            env.execute(action)

        current_total_wait = env.get_cumulated_waiting_time()
        last_reward = env.compute_reward(previous_total_wait, current_total_wait, previous_action, action)
        previous_total_wait = current_total_wait
        previous_action = action
        waiting_times.append(current_total_wait)
        queues.append(current_queue)
        state = env.get_state()
    env.deactivate()

    return {
        "avg_wait": float(np.mean(waiting_times)),
        "avg_queue": float(np.mean(queues)),
        "history": ctrl.history if ctrl else [],
        "threshold": ctrl.profile.detection_threshold if ctrl else None,
    }


def pilot_reference(seed, shift_name):
    try:
        with open(PILOT_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if int(row["seed"]) == seed and row["shift"].lower().startswith(shift_name):
                    return float(row["DQN_Unmitigated_wait"]), float(row["DQN_Adaptive_wait"])
    except (OSError, KeyError, ValueError):
        pass
    return None


def main():
    args = parse_args()
    seed = args.seed or DEFAULT_SEED[args.shift]
    setup_all()
    from tlcs.settings import load_training_settings
    settings = load_training_settings(SETTINGS_FILE)

    print("=" * 78)
    print("  ADAPTIVE CONTROLLER DEMO  (Monitor -> Diagnosis -> Mitigation -> Fallback)")
    print("=" * 78)
    print(f"  Shift injected : {args.shift}  at decision step {INJECTION_STEP} (each decision = 10 s green + 4 s yellow if the phase changes)")
    print(f"  Traffic seed   : {seed}   | DQN model: models/seed_1 (frozen)")
    print(f"  Action mitigation: {'ON (experimental, measured harmful)' if args.enable_action_mitigation else 'OFF (default; measured harmful)'}")
    if args.shift == "reward":
        print("  Note: the reward shift is not part of the Phase 3 pilot; here the real shifted reward")
        print("        is fed to the monitor (the pilot loop feeds 0.0), so there is no stored reference.")
    print("-" * 78)
    print("  Live log (printed whenever the controller's state changes):")

    t0 = time.time()
    res = run_episode("DQN_Adaptive", seed, args.shift, settings, args)
    hist = res["history"]
    after = [h for h in hist if h["step"] >= INJECTION_STEP]
    before = [h for h in hist if h["step"] < INJECTION_STEP]
    first_alarm = next((h["step"] for h in after if h["anomaly_detected"]), None)
    false_alarms = sum(h["anomaly_detected"] for h in before)
    diag = Counter(h["active_diagnosis"] for h in after if h["anomaly_detected"])
    mitig = Counter(h["mitigation_applied"] for h in after)
    fallback_steps = sum(h["is_fallback_active"] for h in hist)

    print("-" * 78)
    print("  WHAT WAS DETECTED")
    print(f"    Detection threshold (3-sigma, from calibration): {res['threshold']:.4f}")
    print(f"    Max anomaly score after injection : {max((h['anomaly_score'] for h in after), default=0):.4f}")
    print(f"    Alarms before injection (false)   : {false_alarms} decision steps")
    if first_alarm is None:
        print("    Shift detected?                   : NO - the monitor never raised an alarm after the injection")
    else:
        print(f"    Shift detected?                   : YES, first alarm at decision {first_alarm} "
              f"({first_alarm - INJECTION_STEP} decisions after injection)")
        print(f"    Alarmed decision steps            : {sum(h['anomaly_detected'] for h in after)} of {len(after)}")
        print(f"    Diagnosis while alarmed           : " + ", ".join(f"{k}={v}" for k, v in diag.most_common()))
    print("  WHAT THE CONTROLLER DID (decisions after injection)")
    for k, v in mitig.most_common():
        print(f"    {k:<45s}: {v}")
    print(f"    Fallback to Max-Pressure (whole episode): {fallback_steps} decision steps")

    print("-" * 78)
    print("  RESULT (mean cumulative waiting time over the episode - the Phase 2/3 metric)")
    print(f"    DQN + adaptive controller : {res['avg_wait']:9.2f} s   (avg queue {res['avg_queue']:.2f} cars)")
    if not args.no_compare:
        plain = run_episode("DQN_Unmitigated", seed, args.shift, settings, args)
        change = (res["avg_wait"] - plain["avg_wait"]) / plain["avg_wait"] * 100
        print(f"    Plain DQN, no protection  : {plain['avg_wait']:9.2f} s   (avg queue {plain['avg_queue']:.2f} cars)")
        print(f"    Difference                : {change:+.2f}%  (negative = adaptive controller helped)")
    ref = pilot_reference(seed, args.shift)
    if ref:
        print(f"    Stored pilot result (results/phase3_pilot): plain DQN {ref[0]:.2f} s, adaptive {ref[1]:.2f} s")
    print(f"  Finished in {time.time() - t0:.1f} s")
    print("=" * 78)


if __name__ == "__main__":
    main()
