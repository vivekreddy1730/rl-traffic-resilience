"""
run_demo.py - watch the trained DQN control a traffic light in SUMO.

Runs ONE simulated hour-and-a-half of traffic (1000 cars, 5400 simulated
seconds) at a single 4-way junction. The frozen, already-trained DQN agent
chooses which direction gets green every 10 seconds. No training happens.

At the end it prints the average waiting time per vehicle (from SUMO's own
trip statistics) and, for seeds 1-3, the value stored in
results/final_checks/metric_independent_baselines.csv for comparison.

Examples:
    python run_demo.py                 # opens the SUMO-GUI window
    python run_demo.py --nogui         # headless, prints the result (~10-30 s)
    python run_demo.py --seed 2 --delay 0
"""

import argparse
import csv
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from project_paths import ROOT_DIR, SETTINGS_FILE, checkpoint_dir, setup_all  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description="Run one SUMO episode controlled by the trained (frozen) DQN agent "
                    "and print the average waiting time per vehicle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--nogui", action="store_true",
                   help="Run headless (no SUMO-GUI window). Use this on servers or for a quick check.")
    p.add_argument("--seed", type=int, default=1,
                   help="Traffic seed. Cars are generated with seed*10000 and the DQN trained with "
                        "this seed (models/seed_N) is used. Valid: 1-5.")
    p.add_argument("--delay", type=int, default=20,
                   help="GUI only: milliseconds to pause per simulated second (bigger = slower animation).")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Stop early after this many simulated seconds (default: full 5400 s episode). "
                        "The stored reference value only applies to a full episode.")
    p.add_argument("--keep-open", action="store_true",
                   help="GUI only: keep the SUMO-GUI window open when the episode ends.")
    return p.parse_args()


def mean_wait_from_tripinfo(path: Path):
    trips = ET.parse(path).getroot().findall("tripinfo")
    if not trips:
        return 0, 0.0
    waits = [float(t.get("waitingTime", 0.0)) for t in trips]
    return len(trips), sum(waits) / len(waits)


def stored_reference(seed: int):
    """DQN per-vehicle waiting time stored on disk for this seed (seeds 1-3 only)."""
    csv_path = ROOT_DIR / "results" / "final_checks" / "metric_independent_baselines.csv"
    key = f"seed_{seed}_wait"
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["controller"] == "DQN Frozen Eval" and key in row:
                    return float(row[key])
    except (OSError, KeyError, ValueError):
        pass
    return None


def main():
    args = parse_args()
    if not 1 <= args.seed <= 5:
        sys.exit("--seed must be between 1 and 5 (one trained model per seed).")

    setup_all()  # finds SUMO, sets import paths, CWD = Vidali folder
    import numpy as np
    from tlcs.agent import Agent
    from tlcs.generator import generate_routefile
    from tlcs.settings import load_training_settings
    from shifts import ShiftConfig, ShiftedEnvironment, ShiftType

    settings = load_training_settings(SETTINGS_FILE)
    max_steps = args.max_steps or settings.max_steps
    ep_seed = args.seed * 10000
    tripinfo = Path(tempfile.mkdtemp(prefix="tsc_demo_")) / "tripinfo.xml"

    class DemoEnvironment(ShiftedEnvironment):
        """Nominal (no shift) environment + tripinfo output + GUI niceties."""

        def build_sumo_cmd(self):
            cmd = super().build_sumo_cmd()
            cmd += ["--tripinfo-output", str(tripinfo)]
            if self.gui:
                cmd += ["--start", "--delay", str(args.delay)]
                if not args.keep_open:
                    cmd += ["--quit-on-end"]
            return cmd

    print("=" * 70)
    print("  TRAFFIC SIGNAL CONTROL DEMO - trained DQN agent (frozen, epsilon = 0)")
    print("=" * 70)
    print(f"  Mode        : {'headless (--nogui)' if args.nogui else 'SUMO-GUI window'}")
    print(f"  Traffic     : {settings.n_cars_generated} cars, seed {args.seed} (route seed {ep_seed})")
    print(f"  Duration    : {max_steps} simulated seconds")
    print(f"  Model       : models/seed_{args.seed}/trained_model.pt")
    print("=" * 70)

    # Same procedure as results/final_checks/metric_independent_baseline_check.py::run_dqn_frozen
    generate_routefile(seed=ep_seed, n_cars_generated=settings.n_cars_generated,
                       max_steps=settings.max_steps, turn_chance=settings.turn_chance)
    agent = Agent(settings=settings, epsilon=0.0, model_path=checkpoint_dir(args.seed))
    env = DemoEnvironment(
        n_cars_generated=settings.n_cars_generated, max_steps=max_steps,
        yellow_duration=settings.yellow_duration, green_duration=settings.green_duration,
        turn_chance=settings.turn_chance, sumocfg_file=settings.sumocfg_file,
        gui=not args.nogui, shift_config=ShiftConfig(shift_type=ShiftType.NOMINAL),
    )
    env.set_seed(ep_seed)

    names = {0: "North-South straight", 1: "North-South left", 2: "East-West straight", 3: "East-West left"}
    t0 = time.time()
    env.activate()
    state = env.get_state()
    decisions = 0
    try:
        while not env.is_over():
            action = int(np.argmax(agent.model.predict_one(state)))
            env.execute(action)
            decisions += 1
            if decisions % 50 == 0:
                print(f"  t = {env.step:5d} s | green: {names[action]:<20s} | cars queued now: {env.get_queue_length()}")
            state = env.get_state()
    finally:
        try:
            env.deactivate()
        except Exception:
            pass

    n_veh, mean_wait = mean_wait_from_tripinfo(tripinfo)
    print("=" * 70)
    print(f"  Finished in {time.time() - t0:.1f} s of real time, {decisions} signal decisions.")
    print(f"  Vehicles that completed their trip : {n_veh}")
    print(f"  AVERAGE WAITING TIME PER VEHICLE   : {mean_wait:.2f} s")
    ref = stored_reference(args.seed)
    if ref is not None and max_steps == settings.max_steps:
        diff = mean_wait - ref
        print(f"  Stored result for seed {args.seed} (results/final_checks): {ref:.2f} s  "
              f"-> difference {diff:+.2f} s {'(exact match)' if abs(diff) < 0.01 else ''}")
    print("=" * 70)


if __name__ == "__main__":
    main()
