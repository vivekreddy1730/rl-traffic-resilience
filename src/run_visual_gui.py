"""
Visual SUMO-GUI Runner v2 using the exact canonical Environment.
Ensures true 80-cell state calculation for DQN, preventing traffic jams,
and provides smooth, realistic speed control (not too slow, not too fast).

Speed control:
  --delay        GUI refresh delay in ms  (slows the animation clock)
  --speed-factor fraction of road max-speed vehicles actually drive
                 0.25 = quarter speed, 0.5 = half speed, 1.0 = full speed
  Both work together: delay slows the frame rate; speed-factor limits how
  far each vehicle moves per simulation second.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Path + SUMO setup (portable; see src/project_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_paths import setup_all, SETTINGS_FILE, VIDALI_DIR as REPO_DIR, ROOT_DIR as WORKSPACE_DIR, checkpoint_dir  # noqa: E402
setup_all()

import sumolib
from sumolib import checkBinary
import traci
import numpy as np

from tlcs.settings import load_training_settings
from tlcs.agent import Agent
from tlcs.env import Environment, EnvStats
from tlcs.constants import (
    ACTION_TO_TL_PHASE,
    TRAFFIC_LIGHT_ID,
    TL_GREEN_TO_YELLOW,
    INCOMING_EDGES,
)
from run_baselines import FixedTimeController, MaxPressureController

PHASE_NAMES = {
    0: "NS Straight / Right (Green Phase 0)",
    1: "NS Left Turn         (Green Phase 2)",
    2: "EW Straight / Right (Green Phase 4)",
    3: "EW Left Turn         (Green Phase 6)",
}


class VisualEnvironment(Environment):
    """
    Subclass of the canonical Environment configured for smooth, realistic visual inspection
    in SUMO-GUI without altering simulation physics or state representations.
    """
    def __init__(self, delay_ms: int = 80, auto_start: bool = True, quit_on_end: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay_ms = delay_ms
        self.auto_start = auto_start
        self.quit_on_end = quit_on_end

    def build_sumo_cmd(self) -> list[str]:
        cmd = super().build_sumo_cmd()
        if self.auto_start:
            cmd.append("--start")
        if self.quit_on_end:
            cmd.append("--quit-on-end")
        if self.delay_ms > 0:
            cmd.extend(["--delay", str(self.delay_ms)])
        return cmd

    def _simulate(self, duration: int) -> list[EnvStats]:
        stats: list[EnvStats] = []
        steps_todo = min(duration, self.max_steps - self.step)

        for _ in range(steps_todo):
            try:
                traci.simulationStep()
            except Exception:
                self.step = self.max_steps
                break

            self.step += 1
            queue_length = self.get_queue_length()
            stats.append(EnvStats(queue_length=queue_length))

        return stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SUMO-GUI visual simulation for RL Traffic Signal Control."
    )
    parser.add_argument(
        "--controller",
        type=str,
        choices=["dqn", "maxpressure", "fixedtime"],
        default="dqn",
        help="Controller type to run (default: dqn).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Evaluation seed (default: 1; generates route with ep_seed = seed * 10000).",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=80,
        help="Simulation step delay in milliseconds (default: 80ms for realistic, smooth speed; not too fast, not too slow).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5400,
        help="Maximum simulation seconds (default: 5400).",
    )
    parser.add_argument(
        "--min-green",
        type=int,
        default=1,
        help="Minimum green duration in decision steps (each = 10s) for Max-Pressure (default: 1 step = 10s hold).",
    )
    parser.add_argument(
        "--speed-factor",
        type=float,
        default=0.5,
        help=(
            "Fraction of road max-speed that vehicles travel (default: 0.5 = half speed). "
            "1.0 = full SUMO speed, 0.25 = quarter speed (very slow demo). "
            "Applied to all edges via traci right after simulation starts."
        ),
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        default=True,
        help="Automatically start the simulation in GUI without waiting for Play button (default: True).",
    )
    parser.add_argument(
        "--quit-on-end",
        action="store_true",
        default=False,
        help="Automatically quit GUI when simulation completes (default: False).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    settings = load_training_settings(SETTINGS_FILE)

    ep_seed = args.seed * 10000
    print("=" * 70)
    print("  SUMO-GUI REAL-TIME VISUAL SIMULATION (CANONICAL ENVIRONMENT)")
    print("=" * 70)
    print(f"  Controller:     {args.controller.upper()}")
    print(f"  Seed:           {args.seed} (Route Seed: {ep_seed})")
    print(f"  GUI Delay:      {args.delay} ms")
    print(f"  Vehicle Speed:  {args.speed_factor}x of road max-speed  "
          f"({'half speed — comfortable' if args.speed_factor == 0.5 else 'quarter speed — very slow' if args.speed_factor <= 0.25 else 'full SUMO speed'})")
    print(f"  Max Steps:      {args.max_steps} s")
    if args.controller == "maxpressure":
        print(f"  Min Green:      {args.min_green} decision step(s) = {args.min_green * settings.green_duration} s hold")
    print("=" * 70)
    print("  NOTE: Vehicles queued on the RED side waiting is CORRECT behaviour.")
    print("        The controller holds green for the busiest lanes first.")
    print("        Use 'Delay' slider in the SUMO-GUI toolbar to adjust speed.")
    print("=" * 70)

    # 1. Setup Controller
    agent = None
    controller = None

    if args.controller == "dqn":
        ckpt_path = checkpoint_dir(args.seed)
        model_file = ckpt_path / "trained_model.pt"
        if not model_file.exists():
            print(f"[ERROR] Checkpoint not found: {model_file}")
            sys.exit(1)
        print(f"  Loading verified DQN model: {ckpt_path.name}/trained_model.pt (frozen policy, eps=0.0)")
        agent = Agent(settings=settings, epsilon=0.0, model_path=ckpt_path)
        agent.set_epsilon(0.0)

    elif args.controller == "maxpressure":
        print(f"  Initializing MaxPressureController (min_green={args.min_green} steps = {args.min_green * settings.green_duration}s hold)")
        controller = MaxPressureController(min_green=args.min_green)
        controller.reset()

    elif args.controller == "fixedtime":
        print("  Initializing FixedTimeController (10s green cycle)")
        controller = FixedTimeController(
            green_duration=settings.green_duration,
            yellow_duration=settings.yellow_duration,
        )
        controller.reset()

    # 2. Build Canonical Visual Environment
    # Note: max_steps is initialized with settings.max_steps (5400) so that route generation
    # uses the nominal, realistic traffic density (1000 cars over 5400s) instead of artificially choking.
    env = VisualEnvironment(
        delay_ms=args.delay,
        auto_start=args.auto_start,
        quit_on_end=args.quit_on_end,
        n_cars_generated=settings.n_cars_generated,
        max_steps=settings.max_steps,
        yellow_duration=settings.yellow_duration,
        green_duration=settings.green_duration,
        turn_chance=settings.turn_chance,
        gui=True,
        sumocfg_file=settings.sumocfg_file,
    )

    print(f"\n  Generating routefile for Seed {args.seed} ({settings.n_cars_generated} cars over {settings.max_steps}s nominal density)...")
    env.generate_routefile(seed=ep_seed)

    print(f"  Launching SUMO-GUI window...")
    env.activate()

    # Apply speed factor to all edges so vehicles travel at a fraction of the
    # road max-speed.  This makes movement visually smooth and easy to follow
    # without altering the RL state, queue counts, or physics.
    if abs(args.speed_factor - 1.0) > 0.01:
        try:
            edge_ids = traci.edge.getIDList()
            for eid in edge_ids:
                traci.edge.setMaxSpeed(eid, 13.89 * args.speed_factor)
            print(f"  Speed factor {args.speed_factor}x applied to {len(edge_ids)} edges "
                  f"(max {13.89 * args.speed_factor:.2f} m/s = "
                  f"{13.89 * args.speed_factor * 3.6:.1f} km/h).")
        except Exception as sf_err:
            print(f"  [WARN] Could not apply speed factor: {sf_err}")

    decision_step = 0
    phase_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    queue_history = []

    print("\n  [SIMULATION ACTIVE - Watch traffic lights dynamically adapt!]\n")

    try:
        while not env.is_over() and env.step < args.max_steps:
            # True 80-cell binary occupancy state from canonical environment
            state = env.get_state()

            if args.controller == "dqn":
                action = agent.choose_action(state)
            else:
                action = controller.choose_action(state)

            phase_counts[action] = phase_counts.get(action, 0) + 1
            decision_step += 1

            # Execute phase in environment (handles yellow + green durations)
            action_stats = env.execute(action)
            for s in action_stats:
                queue_history.append(s.queue_length)

            current_queue = queue_history[-1] if queue_history else 0
            current_total_wait = env.get_cumulated_waiting_time()

            if decision_step % 10 == 0 or env.is_over() or decision_step == 1:
                phase_label = PHASE_NAMES.get(action, f"Action {action}")
                print(
                    f"  [Sim Step {env.step:4d}/{args.max_steps}] Decision {decision_step:3d} | "
                    f"Selected {phase_label:<32s} | Queue: {current_queue:2d} veh | "
                    f"Total Wait: {current_total_wait:7.1f}s"
                )

    except Exception as e:
        print(f"\n[INFO] Simulation closed: {e}")

    finally:
        try:
            env.deactivate()
        except Exception:
            pass

    mean_queue = np.mean(queue_history) if queue_history else 0.0
    print("\n" + "=" * 70)
    print("  SIMULATION COMPLETE")
    print(f"  Simulation Seconds:     {env.step} s")
    print(f"  Decision Steps:         {decision_step} steps")
    print(f"  Mean Queue Length:      {mean_queue:.2f} vehicles")
    print("  Phase Breakdown:       ", {PHASE_NAMES[k].split()[0]: v for k, v in phase_counts.items()})
    print("=" * 70)


if __name__ == "__main__":
    main()
