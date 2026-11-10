"""
Diagnostic script for Stages 2 & 3: Proving Max-Pressure Bug and Metric Consistency.
Runs solely for Seed 1, logs actions and tripinfo output, and saves to results/diagnostics/.
Does NOT modify or touch results/phase1_baseline/.
"""

import os
import sys
import json
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
import numpy as np

# Workspace paths
# Path + SUMO setup (portable; all paths are relative to this file - see src/project_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from project_paths import setup_all, SETTINGS_FILE, VIDALI_DIR as REPO_DIR, MODELS_DIR, PHASE2_DIR, ROOT_DIR, output_dir  # noqa: E402
setup_all()
WORKSPACE_DIR = ROOT_DIR
DIAG_DIR = ROOT_DIR / "results" / "diagnostics"
CHECKPOINT_DIR = MODELS_DIR

import traci
from tlcs.settings import load_training_settings
from tlcs.agent import Agent
from tlcs.generator import generate_routefile
from tlcs.constants import (
    ACTION_TO_TL_PHASE,
    TRAFFIC_LIGHT_ID,
    TL_GREEN_TO_YELLOW,
    INCOMING_EDGES,
)

SETTINGS_FILE = REPO_DIR / "settings" / "training_settings.yaml"
settings = load_training_settings(SETTINGS_FILE)
DIAG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------

class MaxPressureController:
    ACTION_TO_EDGES = {
        0: ("N2TL", "S2TL"),
        1: ("N2TL", "S2TL"),
        2: ("E2TL", "W2TL"),
        3: ("E2TL", "W2TL"),
    }

    def __init__(self, min_green: int = 5):
        self.min_green = min_green
        self.current_action = 0
        self.steps_in_action = 0

    def _get_pressure(self, edges: tuple) -> int:
        return sum(traci.edge.getLastStepHaltingNumber(e) for e in edges)

    def choose_action(self, _state) -> int:
        if self.steps_in_action < self.min_green:
            self.steps_in_action += 1
            return self.current_action

        pressures = {
            action: self._get_pressure(edges)
            for action, edges in self.ACTION_TO_EDGES.items()
        }
        best_action = max(pressures, key=pressures.__getitem__)

        if best_action != self.current_action:
            self.current_action = best_action
            self.steps_in_action = 0
        self.steps_in_action += 1
        return self.current_action

    def reset(self):
        self.current_action = 0
        self.steps_in_action = 0


class FixedTimeController:
    def __init__(self, green_duration: int = 10, yellow_duration: int = 4):
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.phase_index = 0
        self.steps_in_phase = 0

    def choose_action(self, _state) -> int:
        if self.steps_in_phase >= self.green_duration:
            self.phase_index = (self.phase_index + 1) % 4
            self.steps_in_phase = 0
        self.steps_in_phase += 1
        return self.phase_index

    def reset(self):
        self.phase_index = 0
        self.steps_in_phase = 0


def run_rule_based_diagnostic(name: str, controller, seed: int):
    print(f"\n{'='*60}")
    print(f"  RUNNING DIAGNOSTIC: {name} (Seed {seed})")
    print(f"{'='*60}")

    ep_seed = seed * 10000
    generate_routefile(
        seed=ep_seed,
        n_cars_generated=settings.n_cars_generated,
        max_steps=settings.max_steps,
        turn_chance=settings.turn_chance,
    )
    controller.reset()

    tripinfo_file = DIAG_DIR / f"tripinfo_{name.lower()}.xml"
    if tripinfo_file.exists():
        tripinfo_file.unlink()

    import sumolib
    sumo_binary = sumolib.checkBinary("sumo")
    sumo_cmd = [
        sumo_binary,
        "-c", str(settings.sumocfg_file),
        "--no-step-log", "true",
        "--waiting-time-memory", str(settings.max_steps),
        "--tripinfo-output", str(tripinfo_file),
    ]

    traci.start(sumo_cmd)

    step = 0
    actions_chosen = []
    queue_lengths = []
    rewards = []
    previous_total_wait = 0.0
    current_green_phase = 0
    state = np.zeros(80, dtype=float)

    teleports_started = 0

    while step < settings.max_steps:
        action = controller.choose_action(state)
        actions_chosen.append(int(action))
        next_green_phase = ACTION_TO_TL_PHASE[action]

        # Yellow transition
        if next_green_phase != current_green_phase:
            yellow_phase = TL_GREEN_TO_YELLOW[current_green_phase]
            traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, yellow_phase)
            for _ in range(min(settings.yellow_duration, settings.max_steps - step)):
                traci.simulationStep()
                step += 1
                q = sum(traci.edge.getLastStepHaltingNumber(e) for e in ("N2TL", "S2TL", "E2TL", "W2TL"))
                queue_lengths.append(q)
                teleports_started += traci.simulation.getStartingTeleportNumber()

        if step >= settings.max_steps:
            break

        # Green phase
        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, next_green_phase)
        current_green_phase = next_green_phase

        for _ in range(min(settings.green_duration, settings.max_steps - step)):
            traci.simulationStep()
            step += 1
            q = sum(traci.edge.getLastStepHaltingNumber(e) for e in ("N2TL", "S2TL", "E2TL", "W2TL"))
            queue_lengths.append(q)
            teleports_started += traci.simulation.getStartingTeleportNumber()

        # Cumulative wait computation
        wait = 0.0
        for car_id in traci.vehicle.getIDList():
            if traci.vehicle.getRoadID(car_id) in INCOMING_EDGES:
                wait += float(traci.vehicle.getAccumulatedWaitingTime(car_id))
        reward = previous_total_wait - wait
        previous_total_wait = wait
        rewards.append(reward)

    # Simulation vehicle counts
    departed = traci.simulation.getDepartedNumber()
    arrived = traci.simulation.getArrivedNumber()
    still_in_net = len(traci.vehicle.getIDList())

    traci.close()

    cumulative_wait = sum(queue_lengths)
    decision_count = len(rewards)
    pipeline_avg_wait = round(cumulative_wait / max(decision_count, 1), 4)

    return {
        "controller": name,
        "seed": seed,
        "ep_seed": ep_seed,
        "actions_chosen": actions_chosen,
        "action_histogram": dict(Counter(actions_chosen)),
        "vehicles_departed": departed,
        "vehicles_arrived": arrived,
        "vehicles_still_in_net": still_in_net,
        "teleports": teleports_started,
        "cumulative_wait": cumulative_wait,
        "decision_count": decision_count,
        "pipeline_avg_wait": pipeline_avg_wait,
        "tripinfo_file": str(tripinfo_file),
    }


def run_dqn_frozen_diagnostic(seed: int):
    print(f"\n{'='*60}")
    print(f"  RUNNING DIAGNOSTIC: DQN_Frozen_Eval (Seed {seed})")
    print(f"{'='*60}")

    ep_seed = seed * 10000
    generate_routefile(
        seed=ep_seed,
        n_cars_generated=settings.n_cars_generated,
        max_steps=settings.max_steps,
        turn_chance=settings.turn_chance,
    )

    ckpt_path = CHECKPOINT_DIR / f"seed_{seed}"
    agent = Agent(settings=settings, epsilon=0.0, model_path=ckpt_path)
    agent.set_epsilon(0.0)

    tripinfo_file = DIAG_DIR / "tripinfo_dqn_frozen_eval.xml"
    if tripinfo_file.exists():
        tripinfo_file.unlink()

    import sumolib
    sumo_binary = sumolib.checkBinary("sumo")
    sumo_cmd = [
        sumo_binary,
        "-c", str(settings.sumocfg_file),
        "--no-step-log", "true",
        "--waiting-time-memory", str(settings.max_steps),
        "--tripinfo-output", str(tripinfo_file),
    ]

    traci.start(sumo_cmd)

    step = 0
    actions_chosen = []
    queue_lengths = []
    rewards = []
    previous_total_wait = 0.0
    current_green_phase = 0
    teleports_started = 0

    from tlcs.env import Environment
    # We use Environment helper methods for state extraction
    env = Environment(
        n_cars_generated=settings.n_cars_generated,
        max_steps=settings.max_steps,
        yellow_duration=settings.yellow_duration,
        green_duration=settings.green_duration,
        turn_chance=settings.turn_chance,
        gui=False,
        sumocfg_file=settings.sumocfg_file,
    )
    env.step = 0

    while step < settings.max_steps:
        # Get 80-dim binary state
        state = env.get_state()
        action = int(agent.choose_action(state))
        actions_chosen.append(action)

        next_green_phase = ACTION_TO_TL_PHASE[action]

        # Yellow transition
        if next_green_phase != current_green_phase:
            yellow_phase = TL_GREEN_TO_YELLOW[current_green_phase]
            traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, yellow_phase)
            for _ in range(min(settings.yellow_duration, settings.max_steps - step)):
                traci.simulationStep()
                step += 1
                env.step = step
                q = env.get_queue_length()
                queue_lengths.append(q)
                teleports_started += traci.simulation.getStartingTeleportNumber()

        if step >= settings.max_steps:
            break

        # Green phase
        traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, next_green_phase)
        current_green_phase = next_green_phase

        for _ in range(min(settings.green_duration, settings.max_steps - step)):
            traci.simulationStep()
            step += 1
            env.step = step
            q = env.get_queue_length()
            queue_lengths.append(q)
            teleports_started += traci.simulation.getStartingTeleportNumber()

        current_total_wait = env.get_cumulated_waiting_time()
        reward = previous_total_wait - current_total_wait
        previous_total_wait = current_total_wait
        rewards.append(reward)

    departed = traci.simulation.getDepartedNumber()
    arrived = traci.simulation.getArrivedNumber()
    still_in_net = len(traci.vehicle.getIDList())

    traci.close()

    cumulative_wait = sum(queue_lengths)
    decision_count = len(rewards)
    pipeline_avg_wait = round(cumulative_wait / max(decision_count, 1), 4)

    return {
        "controller": "DQN_frozen_eval",
        "seed": seed,
        "ep_seed": ep_seed,
        "actions_chosen": actions_chosen,
        "action_histogram": dict(Counter(actions_chosen)),
        "vehicles_departed": departed,
        "vehicles_arrived": arrived,
        "vehicles_still_in_net": still_in_net,
        "teleports": teleports_started,
        "cumulative_wait": cumulative_wait,
        "decision_count": decision_count,
        "pipeline_avg_wait": pipeline_avg_wait,
        "tripinfo_file": str(tripinfo_file),
    }


def parse_tripinfo(xml_path: Path):
    if not xml_path.exists():
        return {"error": "File not found"}
    tree = ET.parse(xml_path)
    root = tree.getroot()

    trips = root.findall("tripinfo")
    if not trips:
        return {
            "vehicles_completed": 0,
            "mean_waiting_time": None,
            "mean_travel_time": None,
            "mean_time_loss": None,
        }

    waits = [float(t.get("waitingTime", 0.0)) for t in trips]
    durations = [float(t.get("duration", 0.0)) for t in trips]
    time_losses = [float(t.get("timeLoss", 0.0)) for t in trips]

    return {
        "vehicles_completed": len(trips),
        "mean_waiting_time_per_vehicle_s": round(float(np.mean(waits)), 2),
        "median_waiting_time_s": round(float(np.median(waits)), 2),
        "max_waiting_time_s": round(float(np.max(waits)), 2),
        "mean_travel_time_s": round(float(np.mean(durations)), 2),
        "mean_time_loss_s": round(float(np.mean(time_losses)), 2),
    }


if __name__ == "__main__":
    results = {}

    # 1. Max-Pressure
    mp_ctrl = MaxPressureController(min_green=5)
    mp_res = run_rule_based_diagnostic("MaxPressure", mp_ctrl, seed=1)
    mp_trip = parse_tripinfo(Path(mp_res["tripinfo_file"]))
    mp_res["tripinfo_stats"] = mp_trip
    results["MaxPressure"] = mp_res

    # 2. Fixed-Time
    ft_ctrl = FixedTimeController(green_duration=settings.green_duration, yellow_duration=settings.yellow_duration)
    ft_res = run_rule_based_diagnostic("FixedTime", ft_ctrl, seed=1)
    ft_trip = parse_tripinfo(Path(ft_res["tripinfo_file"]))
    ft_res["tripinfo_stats"] = ft_trip
    results["FixedTime"] = ft_res

    # 3. DQN Frozen Eval
    dqn_res = run_dqn_frozen_diagnostic(seed=1)
    dqn_trip = parse_tripinfo(Path(dqn_res["tripinfo_file"]))
    dqn_res["tripinfo_stats"] = dqn_trip
    results["DQN_frozen_eval"] = dqn_res

    # Save summary
    out_json = DIAG_DIR / "maxpressure_diagnostic_summary.json"
    # Convert actions_chosen to length only in summary to keep JSON clean
    summary_to_save = {}
    for k, v in results.items():
        summary_to_save[k] = {ik: iv for ik, iv in v.items() if ik != "actions_chosen"}
        summary_to_save[k]["total_actions_logged"] = len(v["actions_chosen"])

    with open(out_json, "w") as f:
        json.dump(summary_to_save, f, indent=2)

    # Save action histograms CSV
    hist_csv = DIAG_DIR / "action_histograms.csv"
    with open(hist_csv, "w") as f:
        f.write("controller,action_0_NS_straight,action_1_NS_left,action_2_EW_straight,action_3_EW_left,total_actions\n")
        for k, v in results.items():
            hist = v["action_histogram"]
            a0 = hist.get(0, 0)
            a1 = hist.get(1, 0)
            a2 = hist.get(2, 0)
            a3 = hist.get(3, 0)
            tot = len(v["actions_chosen"])
            f.write(f"{k},{a0},{a1},{a2},{a3},{tot}\n")

    print("\n" + "=" * 65)
    print("  DIAGNOSTIC RUN COMPLETED SUCCESSFULLY")
    print("=" * 65)
    print(f"Summary saved to: {out_json}")
    print(f"Action histograms saved to: {hist_csv}")
