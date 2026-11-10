"""
Check simulation seconds at decision step 120 for each controller.
"""
import os
import sys
from pathlib import Path
import numpy as np

# Path + SUMO setup (portable; all paths are relative to this file - see src/project_paths.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from project_paths import setup_all, SETTINGS_FILE, VIDALI_DIR as REPO_DIR, MODELS_DIR, PHASE2_DIR, ROOT_DIR, output_dir  # noqa: E402
setup_all()
ROOT, REPO = ROOT_DIR, REPO_DIR

import traci
from tlcs.settings import load_training_settings
from tlcs.agent import Agent
from shifts import ShiftedEnvironment, ShiftConfig, ShiftType
from adaptive_controller import AdaptiveController
from run_baselines import FixedTimeController, MaxPressureController

settings = load_training_settings(REPO / "settings" / "training_settings.yaml")
CHECKPOINT_DIR = MODELS_DIR  # frozen DQN weights: models/seed_N/trained_model.pt
CALIB_PROFILE = ROOT / "results" / "phase2_shifts" / "calibration_profile.json"
DIAG_CLASSIFIER = ROOT / "results" / "phase2_shifts" / "diagnosis_classifier.joblib"

controllers = ["Fixed_Time", "Max_Pressure", "DQN_Unmitigated", "DQN_Adaptive"]
seeds = [7, 8, 9]

results = {c: [] for c in controllers}

for c_name in controllers:
    print(f"\n--- Testing Controller: {c_name} ---")
    for s in seeds:
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
        env.set_seed(ep_seed)
        env.generate_routefile(seed=ep_seed)

        agent = Agent(settings=settings, epsilon=0.0, model_path=CHECKPOINT_DIR / "seed_1")
        adaptive_ctrl = None
        fixed_time_ctrl = None
        max_pressure_ctrl = None

        if c_name == "DQN_Adaptive":
            adaptive_ctrl = AdaptiveController(
                agent=agent,
                profile_path=CALIB_PROFILE,
                classifier_path=DIAG_CLASSIFIER,
            )
        elif c_name == "Fixed_Time":
            fixed_time_ctrl = FixedTimeController(
                green_duration=settings.green_duration,
                yellow_duration=settings.yellow_duration,
            )
        elif c_name == "Max_Pressure":
            max_pressure_ctrl = MaxPressureController(min_green=5)

        env.activate()
        state = env.get_state()
        steps = 0
        sim_sec_at_120 = None
        prev_wait = 0.0

        while not env.is_over():
            steps += 1
            if steps == 120:
                sim_sec_at_120 = env.step
                print(f"Seed {s}: At decision step 120, simulation second = {env.step} s")
                results[c_name].append(env.step)
                break

            current_queue = float(env.get_queue_length())
            mean_speed = float(env.get_incoming_mean_speed())
            free_flow_speed = float(env.get_incoming_free_flow_speed())
            extra_green_steps = 0

            if c_name == "DQN_Unmitigated":
                q = agent.model.predict_one(state)
                act = int(np.argmax(q))
            elif c_name == "DQN_Adaptive":
                prev_wait_delta = prev_wait - env.get_cumulated_waiting_time()
                act, extra_green_steps, info = adaptive_ctrl.select_action(
                    raw_state=state,
                    queue_len=current_queue,
                    mean_speed=mean_speed,
                    free_flow_speed=free_flow_speed,
                    prev_reward=0.0,
                    prev_wait_delta=prev_wait_delta,
                )
            elif c_name == "Fixed_Time":
                act = fixed_time_ctrl.choose_action(state)
            elif c_name == "Max_Pressure":
                act = max_pressure_ctrl.choose_action(state)

            if extra_green_steps > 0:
                env.green_duration += extra_green_steps
                env.execute(act)
                env.green_duration -= extra_green_steps
            else:
                env.execute(act)

            prev_wait = env.get_cumulated_waiting_time()
            state = env.get_state()

        env.deactivate()

print("\n" + "="*50)
print("SUMMARY: Simulation Seconds at Decision Step 120")
print("="*50)
for c_name in controllers:
    vals = results[c_name]
    print(f"{c_name:18s}: Seed 7 = {vals[0]}s, Seed 8 = {vals[1]}s, Seed 9 = {vals[2]}s | Mean = {np.mean(vals):.1f}s")
