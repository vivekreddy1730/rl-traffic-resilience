"""
check_setup.py - confirm this computer can run the project. Called by setup.ps1 / setup.sh.

Checks: Python version, pinned packages, SUMO (sumo + sumo-gui), trained models,
diagnosis classifier, and a 300-second headless simulation with the trained DQN.
Prints "SETUP OK" at the end, or the exact steps to fix what failed.
"""

import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PINNED = {
    "torch": "2.13.0", "numpy": "1.26.4", "sklearn": "1.5.0", "scipy": "1.17.1",
    "joblib": "1.5.3", "yaml": "6.0.3", "pydantic": "2.13.4", "matplotlib": "3.11.1",
    "traci": None, "sumolib": None, "typer": None, "rich": None,
}
problems = []


def ok(msg):
    print(f"  [OK]   {msg}")


def bad(msg, fix):
    print(f"  [FAIL] {msg}")
    problems.append((msg, fix))


print("=" * 70)
print("  SETUP CHECK")
print("=" * 70)

# 1. Python
v = sys.version_info
if (3, 10) <= (v.major, v.minor) <= (3, 12):
    ok(f"Python {v.major}.{v.minor}.{v.micro} ({sys.executable})")
else:
    bad(f"Python {v.major}.{v.minor} is not supported (need 3.10, 3.11 or 3.12)",
        "Install Python 3.11 from https://www.python.org/downloads/release/python-3119/ and re-run setup.")

# 2. Packages
for mod, want in PINNED.items():
    try:
        m = importlib.import_module(mod)
        have = getattr(m, "__version__", "?")
        if want and not str(have).startswith(want):
            bad(f"{mod} {have} installed, {want} expected",
                "Run:  python -m pip install -r requirements.txt")
        else:
            ok(f"{mod} {have}")
    except Exception as e:  # noqa: BLE001
        bad(f"cannot import {mod}: {e}", "Run:  python -m pip install -r requirements.txt")

# 3. SUMO
from project_paths import DIAG_CLASSIFIER, MODELS_DIR, find_sumo_home, setup_all  # noqa: E402

home = find_sumo_home()
if home is None:
    bad("SUMO not found (no valid SUMO_HOME and no eclipse-sumo pip package)",
        "Run:  python -m pip install eclipse-sumo==1.27.1   (or install SUMO 1.27.1 and set SUMO_HOME)")
else:
    exe = ".exe" if sys.platform == "win32" else ""
    sumo_bin, gui_bin = home / "bin" / f"sumo{exe}", home / "bin" / f"sumo-gui{exe}"
    try:
        out = subprocess.run([str(sumo_bin), "--version"], capture_output=True, text=True, timeout=60).stdout
        first = out.strip().splitlines()[0] if out.strip() else "(no version output)"
        ok(f"{first}  [SUMO_HOME = {home}]")
        if "1.27.1" not in first:
            print("         note: results were produced with SUMO 1.27.1; other versions may give slightly different numbers")
    except Exception as e:  # noqa: BLE001
        bad(f"sumo found at {sumo_bin} but could not be started: {e}",
            "If Windows says 'blocked by Application Control / Smart App Control', see README > Troubleshooting.")
    if gui_bin.exists():
        ok(f"sumo-gui present ({gui_bin.name}) - needed only for the visual demo")
    else:
        print(f"  [WARN] sumo-gui not found in {gui_bin.parent} - use --nogui for the demos")

# 4. Models + classifier
missing = [s for s in range(1, 6) if not (MODELS_DIR / f"seed_{s}" / "trained_model.pt").exists()]
if missing:
    bad(f"trained models missing for seeds {missing}", "Re-download / re-clone the repository (models/ folder).")
else:
    ok("trained DQN models: models/seed_1 .. seed_5")
try:
    import joblib
    joblib.load(DIAG_CLASSIFIER)
    ok("diagnosis classifier loads (results/phase2_shifts/diagnosis_classifier.joblib)")
except Exception as e:  # noqa: BLE001
    bad(f"diagnosis classifier failed to load: {e}", "Make sure scikit-learn==1.5.0 is installed (requirements.txt).")

# 5. Smoke simulation
if not problems:
    try:
        setup_all()
        import numpy as np
        from tlcs.agent import Agent
        from tlcs.env import Environment
        from tlcs.settings import load_training_settings
        from project_paths import SETTINGS_FILE, checkpoint_dir

        s = load_training_settings(SETTINGS_FILE)
        env = Environment(n_cars_generated=50, max_steps=300, yellow_duration=s.yellow_duration,
                          green_duration=s.green_duration, turn_chance=s.turn_chance,
                          sumocfg_file=s.sumocfg_file, gui=False)
        env.generate_routefile(seed=42)
        agent = Agent(settings=s, epsilon=0.0, model_path=checkpoint_dir(1))
        env.activate()
        n = 0
        while not env.is_over():
            env.execute(int(np.argmax(agent.model.predict_one(env.get_state()))))
            n += 1
        env.deactivate()
        ok(f"smoke test: 300 s headless simulation with the trained DQN ({n} decisions)")
    except Exception as e:  # noqa: BLE001
        bad(f"smoke simulation failed: {e}", "See README > Troubleshooting.")

print("=" * 70)
if problems:
    print("  SETUP FAILED - fix these and run setup again:")
    for i, (msg, fix) in enumerate(problems, 1):
        print(f"   {i}. {msg}\n      -> {fix}")
    print("=" * 70)
    sys.exit(1)
print("  SETUP OK")
print("  Next:  python run_demo.py          (opens the SUMO-GUI window)")
print("         python run_demo.py --nogui  (no window)")
print("=" * 70)
