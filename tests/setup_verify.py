"""
Phase 1 Master Setup & Verify Script.

Run this FIRST after SUMO is installed to:
1. Verify SUMO is on PATH and SUMO_HOME is set
2. Install required Python packages into the active environment
3. Verify all imports work
4. Run a 1-episode smoke test

Usage (from project root, tsc_env activated):
    python setup_verify.py
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT / "src"))
from project_paths import setup_sumo, VIDALI_DIR as REPO_DIR, VIDALI_SRC as SRC_DIR, SETTINGS_FILE  # noqa: E402

# Locate SUMO portably: SUMO_HOME env var first, then the eclipse-sumo pip package
setup_sumo(exit_on_error=False)

REQUIRED_PACKAGES = [
    "torch",
    "numpy",
    "matplotlib",
    "pandas",
    "pydantic",
    "pyyaml",
    "typer",
    "rich",
]


def check_sumo() -> bool:
    """Verify SUMO binary is on PATH."""
    sumo_path = shutil.which("sumo")
    if not sumo_path:
        print("[FAIL] 'sumo' not found on PATH.")
        print("       Install SUMO from https://sumo.dlr.de/releases/")
        print("       Then re-run this script.")
        return False

    result = subprocess.run(["sumo", "--version"], capture_output=True, text=True, timeout=10)
    version_line = (result.stdout or result.stderr).strip().splitlines()[0]
    print(f"[OK]   SUMO found: {version_line}")
    print(f"       Binary: {sumo_path}")
    return True


def check_sumo_home() -> bool:
    """Verify SUMO_HOME is set and traci is importable."""
    sumo_home = os.environ.get("SUMO_HOME", "")
    if not sumo_home:
        # Try to find SUMO_HOME from sumo binary location
        sumo_path = shutil.which("sumo")
        if sumo_path:
            # On Windows SUMO installs to C:\Program Files\Eclipse\Sumo
            # SUMO_HOME should be the root above bin/
            sumo_home = str(Path(sumo_path).parent.parent)
            os.environ["SUMO_HOME"] = sumo_home
            print(f"[AUTO] SUMO_HOME set to: {sumo_home}")
        else:
            print("[WARN] SUMO_HOME not set and sumo not found on PATH.")
            return False

    tools_dir = Path(sumo_home) / "tools"
    if not tools_dir.exists():
        print(f"[WARN] SUMO tools dir not found at: {tools_dir}")
        print("       traci may not be importable.")
        return False

    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))

    try:
        import traci
        import sumolib
        print(f"[OK]   traci imported from: {tools_dir}")
        return True
    except ImportError as e:
        print(f"[FAIL] Cannot import traci/sumolib: {e}")
        return False


def install_packages() -> bool:
    """Install required Python packages via pip."""
    print("\n[...] Installing Python packages...")
    cmd = [sys.executable, "-m", "pip", "install"] + REQUIRED_PACKAGES
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("[FAIL] Package installation failed.")
        return False
    print("[OK]   Packages installed.")
    return True


def verify_imports() -> bool:
    """Verify all required packages can be imported."""
    import_map = {
        "torch": "torch",
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "pandas": "pandas",
        "pydantic": "pydantic",
        "yaml": "pyyaml",
        "typer": "typer",
        "rich": "rich",
        "traci": "traci (from SUMO tools)",
        "sumolib": "sumolib (from SUMO tools)",
    }

    all_ok = True
    for mod_name, display in import_map.items():
        try:
            __import__(mod_name)
            print(f"[OK]   {display}")
        except ImportError:
            print(f"[FAIL] {display}")
            all_ok = False
    return all_ok


def smoke_test() -> bool:
    """Run a single short episode to verify end-to-end."""
    print("\n[...] Running smoke test (1 episode, reduced steps)...")

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    os.chdir(REPO_DIR)

    try:
        from tlcs.settings import load_training_settings
        from tlcs.agent import Agent
        from tlcs.env import Environment
        from tlcs.episode import run_episode
        from tlcs.memory import Memory

        settings = load_training_settings(SETTINGS_FILE)

        # Override max_steps to keep smoke test short (5 minutes of sim)
        settings_dict = settings.model_dump()
        settings_dict["max_steps"] = 300
        settings_dict["n_cars_generated"] = 50
        settings_dict["total_episodes"] = 1

        from tlcs.settings import TrainingSettings
        settings_short = TrainingSettings.model_validate(settings_dict)

        memory = Memory(size_max=settings_short.memory_size_max, size_min=settings_short.memory_size_min)
        agent = Agent(settings=settings_short)
        agent.set_epsilon(1.0)

        env = Environment(
            n_cars_generated=settings_short.n_cars_generated,
            max_steps=settings_short.max_steps,
            yellow_duration=settings_short.yellow_duration,
            green_duration=settings_short.green_duration,
            turn_chance=settings_short.turn_chance,
            gui=False,
            sumocfg_file=settings_short.sumocfg_file,
        )

        episode_history, env_stats = run_episode(env=env, agent=agent, seed=42)

        queue_vals = [s.queue_length for s in env_stats]
        avg_q = sum(queue_vals) / max(len(queue_vals), 1)
        print(f"[OK]   Smoke test passed!")
        print(f"       Steps: {len(env_stats)} | Avg queue: {avg_q:.2f}")
        return True

    except Exception as e:
        print(f"[FAIL] Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("  Phase 1 — Setup & Verification")
    print("=" * 60)

    steps = [
        ("SUMO binary", check_sumo),
        ("SUMO_HOME / traci", check_sumo_home),
        # (packages are installed by setup.ps1 / `pip install -r requirements.txt` with pinned versions)
        ("Import verification", verify_imports),
        ("Smoke test (1 short episode)", smoke_test),
    ]

    results = {}
    for name, fn in steps:
        print(f"\n--- {name} ---")
        try:
            results[name] = fn()
        except Exception as e:
            print(f"[ERROR] {e}")
            results[name] = False

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, ok in results.items():
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {status} {name}")
        if not ok:
            all_pass = False

    if all_pass:
        print("\n  All checks passed. Ready to run Phase 1 experiments.")
        print("  Next: python run_demo.py --nogui")
    else:
        print("\n  Some checks failed. Fix the issues above before running experiments.")


if __name__ == "__main__":
    main()
