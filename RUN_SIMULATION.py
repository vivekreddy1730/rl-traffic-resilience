"""
Interactive SUMO-GUI launcher (asks which controller and speed to show).
Run:  python RUN_SIMULATION.py      (or click Run in VS Code)
For a non-interactive demo use:  python run_demo.py
"""

import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# SUMO is located automatically by src/project_paths.py (SUMO_HOME env var,
# or the eclipse-sumo pip package). Nothing machine-specific is hardcoded here.
sys.path.insert(0, str(ROOT / "src"))
from project_paths import setup_sumo  # noqa: E402

setup_sumo()


def main():
    print("=" * 68)
    print("      TRAFFIC SIGNAL CONTROL - VISUAL SIMULATION (VS CODE)")
    print("=" * 68)
    print("What you will see:")
    print("  - Vehicles approach the intersection from all 4 directions")
    print("  - The controller dynamically decides which lanes get green")
    print("  - Queued vehicles wait (red side) while crossing vehicles move")
    print("  - This is CORRECT: the heavy side waits until the light switches")
    print("=" * 68)
    print("Choose Controller to watch:")
    print("  [1] DQN  - Trained AI (smart, adapts to queue pressure) [DEFAULT]")
    print("  [2] Max-Pressure - Rule-based (serves the most-loaded lane)")
    print("  [3] Fixed-Time   - Dumb clock (cycles every 10s no matter what)")
    print("=" * 68)

    choice = input("Enter 1, 2, or 3 (or press ENTER for DQN): ").strip()
    if choice not in ["1", "2", "3"]:
        choice = "1"

    print("\n" + "=" * 68)
    print("Choose Vehicle Speed (how fast cars visually move on screen):")
    print("  [1] Comfortable  (100 ms delay, half speed) — easy to follow [DEFAULT]")
    print("  [2] Very Slow    (160 ms delay, quarter speed) — for demos/review")
    print("  [3] Normal       (50 ms delay, full speed)  — real-time SUMO speed")
    print("=" * 68)
    print("  TIP: After the window opens you can also drag the 'Delay' slider")
    print("       in SUMO-GUI's toolbar to change speed at any time!")
    print("=" * 68)

    speed_choice = input("Enter 1, 2, or 3 (or press ENTER for Comfortable): ").strip()
    # delay_ms  speed_factor (applied via traci after launch)
    speed_map = {
        "1": ("100", "0.5"),   # comfortable: 100ms GUI delay + half vehicle speed
        "2": ("160", "0.25"),  # very slow:   160ms GUI delay + quarter vehicle speed
        "3": ("50",  "1.0"),   # normal:       50ms GUI delay + real speed
    }
    sim_delay, speed_factor = speed_map.get(speed_choice, ("100", "0.5"))

    python_exe = sys.executable
    script = ROOT / "src" / "run_visual_gui.py"

    if choice == "2":
        ctrl_name = "Max-Pressure (min_green=1, 10s hold)"
        ctrl_arg = "maxpressure"
        extra_args = ["--min-green", "1"]
    elif choice == "3":
        ctrl_name = "Fixed-Time Baseline"
        ctrl_arg = "fixedtime"
        extra_args = []
    else:
        ctrl_name = "Trained DQN (AI Controller)"
        ctrl_arg = "dqn"
        extra_args = []

    args = [
        python_exe,
        str(script),
        "--controller", ctrl_arg,
        "--delay", sim_delay,
        "--speed-factor", speed_factor,
        "--seed", "1",
    ] + extra_args

    print(f"\n>> Launching: {ctrl_name}")
    print(f">> GUI delay: {sim_delay} ms  |  Vehicle speed factor: {speed_factor}x")
    print(">> Opening SUMO-GUI window now. Please wait a few seconds...\n")
    subprocess.run(args)

if __name__ == "__main__":
    main()
