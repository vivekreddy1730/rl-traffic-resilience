"""
Shared project paths and SUMO discovery.

Every script imports this module first so that the project runs from any
folder, on any computer, without hardcoded absolute paths.

SUMO is located in this order:
  1. the SUMO_HOME environment variable (a normal SUMO install), then
  2. the `eclipse-sumo` pip package (installed by requirements.txt).
If neither is found, a clear error explains how to fix it.
"""

import os
import sys
from pathlib import Path

# Project layout (all relative to this file, so the repo can live anywhere)
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
VIDALI_DIR = ROOT_DIR / "Deep-QLearning-Agent-for-Traffic-Signal-Control"
VIDALI_SRC = VIDALI_DIR / "src"
SETTINGS_FILE = VIDALI_DIR / "settings" / "training_settings.yaml"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
PHASE2_DIR = RESULTS_DIR / "phase2_shifts"
CALIB_PROFILE = PHASE2_DIR / "calibration_profile.json"
DIAG_CLASSIFIER = PHASE2_DIR / "diagnosis_classifier.joblib"

SUMO_HELP = """
==========================================================================
 SUMO (the traffic simulator) could not be found.

 Easiest fix (all platforms) - install it with pip inside the venv:
     python -m pip install eclipse-sumo==1.27.1

 Or install SUMO 1.27.1 from https://sumo.dlr.de/docs/Downloads.php
 and set the SUMO_HOME environment variable to the install folder:
     Windows (PowerShell):  setx SUMO_HOME "C:\\path\\to\\sumo-1.27.1"
                            (then open a NEW terminal)
     Linux / macOS:         export SUMO_HOME=/path/to/sumo
==========================================================================
"""


def checkpoint_dir(seed: int) -> Path:
    """Folder holding the frozen DQN weights for one training seed."""
    return MODELS_DIR / f"seed_{seed}"


def output_dir(default: Path) -> Path:
    """Where a script should WRITE its results.

    Defaults to `default`. If the RESULTS_OUT_DIR environment variable is set,
    results go there instead, so re-running a verification script never
    overwrites the official result files shipped in results/.
    """
    override = os.environ.get("RESULTS_OUT_DIR", "").strip()
    out = Path(override) if override else default
    out.mkdir(parents=True, exist_ok=True)
    return out


def _sumo_bin_ok(home: Path) -> bool:
    exe = "sumo.exe" if os.name == "nt" else "sumo"
    return (home / "bin" / exe).exists()


def find_sumo_home() -> Path | None:
    """Return a usable SUMO_HOME folder, or None."""
    env_home = os.environ.get("SUMO_HOME", "").strip().strip('"')
    if env_home and _sumo_bin_ok(Path(env_home)):
        return Path(env_home)
    try:
        import sumo  # provided by the `eclipse-sumo` pip package

        pip_home = Path(sumo.SUMO_HOME)
        if _sumo_bin_ok(pip_home):
            return pip_home
    except Exception:
        pass
    return None


def setup_sumo(exit_on_error: bool = True) -> Path | None:
    """Configure SUMO_HOME, PATH and sys.path so traci/sumolib work."""
    home = find_sumo_home()
    if home is None:
        env_home = os.environ.get("SUMO_HOME", "")
        if env_home:
            print(f"[ERROR] SUMO_HOME is set to '{env_home}' but no sumo binary was found in its bin/ folder.")
        print(SUMO_HELP)
        if exit_on_error:
            sys.exit(2)
        return None

    os.environ["SUMO_HOME"] = str(home)
    bin_dir = str(home / "bin")
    if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    # Prefer the pip-installed traci/sumolib; fall back to SUMO's own tools folder.
    tools_dir = str(home / "tools")
    if tools_dir not in sys.path:
        sys.path.append(tools_dir)
    return home


def setup_python_path() -> None:
    """Make `tlcs` (Vidali base code), our src/ modules and baselines importable."""
    for p in (VIDALI_SRC, SRC_DIR, ROOT_DIR / "baselines", ROOT_DIR):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def setup_all(chdir_to_vidali: bool = True) -> Path | None:
    """One call that prepares everything a simulation script needs.

    The Vidali code writes the generated route file to the relative path
    intersection/episode_routes.rou.xml, so simulation scripts run with the
    Vidali folder as the working directory (same as the original project).
    """
    home = setup_sumo()
    setup_python_path()
    if chdir_to_vidali:
        os.chdir(VIDALI_DIR)
    return home
