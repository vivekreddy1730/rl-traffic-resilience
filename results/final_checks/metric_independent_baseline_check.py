"""
Task 5: Metric-Independent Baseline Check on Seeds 1-3 using SUMO Tripinfo Output.
Reports mean per-vehicle waiting time and mean travel time for:
1. Fixed-Time
2. Corrected Max-Pressure (min_green=5)
3. Minimum-Green Max-Pressure Variant (min_green=1)
4. DQN Frozen Eval (epsilon=0.0)
5. SUMO Built-in Actuated Traffic Light

New script only. All outputs stored under results/final_checks/.
"""
import os
import sys
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

# Path + SUMO setup (portable; all paths are relative to this file - see src/project_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from project_paths import setup_all, SETTINGS_FILE, VIDALI_DIR as REPO_DIR, MODELS_DIR, PHASE2_DIR, ROOT_DIR, output_dir  # noqa: E402
setup_all()
ROOT, REPO = ROOT_DIR, REPO_DIR

import sumolib
import traci
from tlcs.settings import load_training_settings
from tlcs.agent import Agent
from tlcs.generator import generate_routefile
from run_baselines import FixedTimeController, MaxPressureController, ACTION_TO_TL_PHASE, TL_GREEN_TO_YELLOW, TRAFFIC_LIGHT_ID

SETTINGS_FILE = REPO / "settings" / "training_settings.yaml"
CHECKPOINT_DIR = MODELS_DIR  # frozen DQN weights: models/seed_N/trained_model.pt
ACTUATED_ADD_XML = ROOT / "results" / "final_checks" / "sumo_actuated.add.xml"
OUT_DIR = output_dir(ROOT / "results" / "final_checks")  # RESULTS_OUT_DIR env var redirects outputs
TRIPINFO_DIR = OUT_DIR / "tripinfos"
TRIPINFO_DIR.mkdir(parents=True, exist_ok=True)

settings = load_training_settings(SETTINGS_FILE)
SEEDS = [1, 2, 3]

def generate_routes_for_seed(ep_seed: int):
    generate_routefile(
        seed=ep_seed,
        n_cars_generated=settings.n_cars_generated,
        max_steps=settings.max_steps,
        turn_chance=settings.turn_chance,
    )

def parse_tripinfo(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    trips = root.findall("tripinfo")
    n_vehicles = len(trips)
    if n_vehicles == 0:
        return 0, 0.0, 0.0
    wait_times = [float(t.get("waitingTime", 0.0)) for t in trips]
    durations = [float(t.get("duration", 0.0)) for t in trips]
    return n_vehicles, float(np.mean(wait_times)), float(np.mean(durations))

# ---------------------------------------------------------------------------
# Runner Functions
# ---------------------------------------------------------------------------

def run_fixed_time(seed: int, tripinfo_file: Path):
    ep_seed = seed * 10000
    generate_routes_for_seed(ep_seed)
    ctrl = FixedTimeController(
        green_duration=settings.green_duration,
        yellow_duration=settings.yellow_duration,
    )
    ctrl.reset()

    sumo_cmd = [
        sumolib.checkBinary("sumo"),
        "-c", str(settings.sumocfg_file),
        "--no-step-log", "true",
        "--waiting-time-memory", str(settings.max_steps),
        "--tripinfo-output", str(tripinfo_file),
    ]
    traci.start(sumo_cmd)

    step = 0
    state = np.zeros(80, dtype=float)
    current_green_phase = 0
    green_duration = settings.green_duration
    yellow_duration = settings.yellow_duration

    while step < settings.max_steps:
        action = ctrl.choose_action(state)
        next_green_phase = ACTION_TO_TL_PHASE[action]

        if next_green_phase != current_green_phase:
            yellow_phase = TL_GREEN_TO_YELLOW[current_green_phase]
            traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, yellow_phase)
            for _ in range(min(yellow_duration, settings.max_steps - step)):
                traci.simulationStep()
                step += 1

        if step >= settings.max_steps:
            break

        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, next_green_phase)
        current_green_phase = next_green_phase

        for _ in range(min(green_duration, settings.max_steps - step)):
            traci.simulationStep()
            step += 1

        if traci.simulation.getMinExpectedNumber() <= 0:
            break

    traci.close()

def run_max_pressure(seed: int, min_green: int, tripinfo_file: Path):
    ep_seed = seed * 10000
    generate_routes_for_seed(ep_seed)
    ctrl = MaxPressureController(min_green=min_green)
    ctrl.reset()

    sumo_cmd = [
        sumolib.checkBinary("sumo"),
        "-c", str(settings.sumocfg_file),
        "--no-step-log", "true",
        "--waiting-time-memory", str(settings.max_steps),
        "--tripinfo-output", str(tripinfo_file),
    ]
    traci.start(sumo_cmd)

    step = 0
    state = np.zeros(80, dtype=float)
    current_green_phase = 0
    green_duration = settings.green_duration
    yellow_duration = settings.yellow_duration

    while step < settings.max_steps:
        action = ctrl.choose_action(state)
        next_green_phase = ACTION_TO_TL_PHASE[action]

        if next_green_phase != current_green_phase:
            yellow_phase = TL_GREEN_TO_YELLOW[current_green_phase]
            traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, yellow_phase)
            for _ in range(min(yellow_duration, settings.max_steps - step)):
                traci.simulationStep()
                step += 1

        if step >= settings.max_steps:
            break

        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, next_green_phase)
        current_green_phase = next_green_phase

        for _ in range(min(green_duration, settings.max_steps - step)):
            traci.simulationStep()
            step += 1

        if traci.simulation.getMinExpectedNumber() <= 0:
            break

    traci.close()

from shifts import ShiftedEnvironment, ShiftConfig, ShiftType

class TripinfoEnvironment(ShiftedEnvironment):
    def __init__(self, tripinfo_file: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tripinfo_file = tripinfo_file

    def build_sumo_cmd(self) -> list[str]:
        cmd = super().build_sumo_cmd()
        cmd.extend(["--tripinfo-output", str(self.tripinfo_file)])
        return cmd

def run_dqn_frozen(seed: int, tripinfo_file: Path):
    ep_seed = seed * 10000
    generate_routes_for_seed(ep_seed)

    ckpt_path = CHECKPOINT_DIR / f"seed_{seed}"
    agent = Agent(settings=settings, epsilon=0.0, model_path=ckpt_path)

    env = TripinfoEnvironment(
        tripinfo_file=tripinfo_file,
        n_cars_generated=settings.n_cars_generated,
        max_steps=settings.max_steps,
        yellow_duration=settings.yellow_duration,
        green_duration=settings.green_duration,
        turn_chance=settings.turn_chance,
        sumocfg_file=settings.sumocfg_file,
        gui=False,
        shift_config=ShiftConfig(shift_type=ShiftType.NOMINAL),
    )
    env.set_seed(ep_seed)
    env.activate()
    state = env.get_state()

    while not env.is_over():
        q_values = agent.model.predict_one(state)
        action = int(np.argmax(q_values))
        env.execute(action)
        state = env.get_state()

    env.deactivate()

def run_sumo_actuated(seed: int, tripinfo_file: Path):
    ep_seed = seed * 10000
    generate_routes_for_seed(ep_seed)

    sumo_cmd = [
        sumolib.checkBinary("sumo"),
        "-c", str(settings.sumocfg_file),
        "-a", str(ACTUATED_ADD_XML),
        "--no-step-log", "true",
        "--waiting-time-memory", str(settings.max_steps),
        "--tripinfo-output", str(tripinfo_file),
    ]
    traci.start(sumo_cmd)

    # SUMO's internal actuated controller controls the light autonomously
    step = 0
    while step < settings.max_steps:
        traci.simulationStep()
        step += 1
        if traci.simulation.getMinExpectedNumber() <= 0:
            break

    traci.close()

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("TASK 5: METRIC-INDEPENDENT BASELINE CHECK (SUMO TRIPINFO OUTPUT)")
    print("Seeds: 1, 2, 3")
    print("Controllers:")
    print("  1. Fixed-Time (pretimed cyclic)")
    print("  2. Corrected Max-Pressure (v2, min_green=5)")
    print("  3. Minimum-Green Max-Pressure Variant (min_green=1)")
    print("  4. DQN Frozen Eval (epsilon=0.0)")
    print("  5. SUMO Built-in Actuated Controller")
    print("=" * 80)

    controllers = [
        ("Fixed-Time", lambda s, p: run_fixed_time(s, p)),
        ("Corrected Max-Pressure (min_green=5)", lambda s, p: run_max_pressure(s, 5, p)),
        ("Max-Pressure Variant (min_green=1)", lambda s, p: run_max_pressure(s, 1, p)),
        ("DQN Frozen Eval", lambda s, p: run_dqn_frozen(s, p)),
        ("SUMO Built-in Actuated", lambda s, p: run_sumo_actuated(s, p)),
    ]

    results_table = []

    for name, runner_fn in controllers:
        print(f"\nEvaluating Controller: {name}")
        seed_waits = []
        seed_travels = []
        seed_counts = []

        for s in SEEDS:
            clean_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "_").replace("-", "_")
            tripinfo_path = TRIPINFO_DIR / f"tripinfo_{clean_name}_seed_{s}.xml"
            print(f"  Seed {s} -> {tripinfo_path.name} ...", end=" ", flush=True)
            runner_fn(s, tripinfo_path)
            n_veh, mean_wait, mean_dur = parse_tripinfo(tripinfo_path)
            seed_waits.append(mean_wait)
            seed_travels.append(mean_dur)
            seed_counts.append(n_veh)
            print(f"Done! Veh: {n_veh}, Mean Wait: {mean_wait:.2f}s, Mean Travel: {mean_dur:.2f}s")

        results_table.append({
            "controller": name,
            "seed_1_wait": seed_waits[0],
            "seed_2_wait": seed_waits[1],
            "seed_3_wait": seed_waits[2],
            "mean_wait": np.mean(seed_waits),
            "std_wait": np.std(seed_waits),
            "seed_1_travel": seed_travels[0],
            "seed_2_travel": seed_travels[1],
            "seed_3_travel": seed_travels[2],
            "mean_travel": np.mean(seed_travels),
            "std_travel": np.std(seed_travels),
            "vehicles_completed": int(np.mean(seed_counts)),
        })

    # Save summary CSV
    out_csv = OUT_DIR / "metric_independent_baselines.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_table[0].keys()))
        writer.writeheader()
        writer.writerows(results_table)
    print(f"\nSaved CSV summary to: {out_csv}")

    # Print Formatted Markdown Table
    print("\n" + "=" * 95)
    print("SUMMARY RESULTS: PER-VEHICLE TRIPINFO METRICS (MEAN ± STD ACROSS SEEDS 1-3)")
    print("=" * 95)
    print(f"| Controller Configuration | Mean Per-Vehicle Waiting Time (s) | Mean Travel Time (Duration, s) | Completed Vehicles |")
    print(f"| :--- | :---: | :---: | :---: |")
    for r in results_table:
        print(f"| **{r['controller']}** | {r['mean_wait']:.2f} ± {r['std_wait']:.2f} s | {r['mean_travel']:.2f} ± {r['std_travel']:.2f} s | {r['vehicles_completed']} / 1000 |")

    # Assess whether DQN's lead over Max-Pressure holds
    dqn_wait = next(r["mean_wait"] for r in results_table if r["controller"] == "DQN Frozen Eval")
    mp5_wait = next(r["mean_wait"] for r in results_table if r["controller"] == "Corrected Max-Pressure (min_green=5)")
    mp1_wait = next(r["mean_wait"] for r in results_table if r["controller"] == "Max-Pressure Variant (min_green=1)")
    ft_wait = next(r["mean_wait"] for r in results_table if r["controller"] == "Fixed-Time")
    act_wait = next(r["mean_wait"] for r in results_table if r["controller"] == "SUMO Built-in Actuated")

    dqn_lead_over_mp5 = (mp5_wait - dqn_wait) / mp5_wait * 100
    dqn_lead_over_mp1 = (mp1_wait - dqn_wait) / mp1_wait * 100

    print("\n" + "=" * 80)
    print("DQN LEAD ANALYSIS UNDER PER-VEHICLE TRIPINFO METRIC:")
    print("=" * 80)
    print(f"DQN Frozen Eval Mean Wait: {dqn_wait:.2f} s")
    print(f"Corrected Max-Pressure (min_green=5) Mean Wait: {mp5_wait:.2f} s")
    print(f"Max-Pressure Variant (min_green=1) Mean Wait: {mp1_wait:.2f} s")
    print(f"Fixed-Time Baseline Mean Wait: {ft_wait:.2f} s")
    print(f"SUMO Built-in Actuated Mean Wait: {act_wait:.2f} s")
    print(f"-> DQN Lead over Corrected Max-Pressure (min_green=5): {dqn_lead_over_mp5:+.1f}% delay reduction")
    print(f"-> DQN Lead over Max-Pressure Variant (min_green=1): {dqn_lead_over_mp1:+.1f}% delay reduction")
    if dqn_wait < mp5_wait:
        print("VERDICT: YES, DQN's lead over Max-Pressure HOLDS under the metric-independent per-vehicle waiting time metric!")
    else:
        print("VERDICT: NO, DQN's lead over Max-Pressure does NOT hold under the per-vehicle waiting time metric.")

if __name__ == "__main__":
    main()
