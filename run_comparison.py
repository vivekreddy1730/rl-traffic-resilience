"""
run_comparison.py - Fixed-Time vs Max-Pressure vs DQN on the SAME traffic.

Every controller gets the identical car arrivals (same route seed). For each
controller SUMO records every finished trip, and we report the average waiting
time and travel time per vehicle. This uses exactly the runner functions of
results/final_checks/metric_independent_baseline_check.py, so for seeds 1-3 the
numbers can be compared with results/final_checks/metric_independent_baselines.csv
(the "stored" column).

Prints a table and saves a bar chart to results/demo_outputs/.

Examples:
    python run_comparison.py                     # seed 1, ~1 minute
    python run_comparison.py --seeds 1 2 3       # reproduces the stored 3-seed table
    python run_comparison.py --include-actuated  # also SUMO's built-in actuated light
"""

import argparse
import csv
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from project_paths import ROOT_DIR, setup_all  # noqa: E402

STORED_CSV = ROOT_DIR / "results" / "final_checks" / "metric_independent_baselines.csv"
OUT_DIR = ROOT_DIR / "results" / "demo_outputs"


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Fixed-Time, Max-Pressure and the trained DQN on the same traffic and "
                    "compare average waiting time per vehicle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[1],
                   help="Traffic seeds to run (1-5). Each seed = one full 5400 s episode per controller.")
    p.add_argument("--include-actuated", action="store_true",
                   help="Also run SUMO's built-in actuated traffic light.")
    p.add_argument("--no-plot", action="store_true", help="Do not save the bar chart.")
    return p.parse_args()


def load_stored():
    stored = {}
    try:
        with open(STORED_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                stored[row["controller"]] = row
    except OSError:
        pass
    return stored


def main():
    args = parse_args()
    for s in args.seeds:
        if not 1 <= s <= 5:
            sys.exit("--seeds must be between 1 and 5")

    setup_all()
    # Import the original evaluation module, but send its import-time output folder to a
    # temp dir so nothing in results/final_checks is created or overwritten.
    tmp = Path(tempfile.mkdtemp(prefix="tsc_compare_"))
    os.environ["RESULTS_OUT_DIR"] = str(tmp)
    sys.path.insert(0, str(ROOT_DIR / "results" / "final_checks"))
    import metric_independent_baseline_check as mib  # noqa: E402
    del os.environ["RESULTS_OUT_DIR"]

    controllers = [
        ("Fixed-Time", lambda s, p: mib.run_fixed_time(s, p)),
        ("Max-Pressure (min_green=5)", lambda s, p: mib.run_max_pressure(s, 5, p)),
        ("Max-Pressure (min_green=1)", lambda s, p: mib.run_max_pressure(s, 1, p)),
        ("DQN (frozen)", lambda s, p: mib.run_dqn_frozen(s, p)),
    ]
    stored_names = {
        "Fixed-Time": "Fixed-Time",
        "Max-Pressure (min_green=5)": "Corrected Max-Pressure (min_green=5)",
        "Max-Pressure (min_green=1)": "Max-Pressure Variant (min_green=1)",
        "DQN (frozen)": "DQN Frozen Eval",
        "SUMO actuated": "SUMO Built-in Actuated",
    }
    if args.include_actuated:
        controllers.append(("SUMO actuated", lambda s, p: mib.run_sumo_actuated(s, p)))

    stored = load_stored()
    rows = []
    print("=" * 78)
    print(f"  CONTROLLER COMPARISON on traffic seed(s) {args.seeds}  (1000 cars, 5400 s each)")
    print("=" * 78)
    for name, fn in controllers:
        for s in args.seeds:
            trip = tmp / f"tripinfo_{len(rows)}.xml"
            t0 = time.time()
            print(f"  Running {name:<28s} seed {s} ...", end=" ", flush=True)
            fn(s, trip)
            n, wait, travel = mib.parse_tripinfo(trip)
            ref_row = stored.get(stored_names[name], {})
            ref = float(ref_row[f"seed_{s}_wait"]) if f"seed_{s}_wait" in ref_row else None
            rows.append({"controller": name, "seed": s, "vehicles": n, "mean_wait_s": wait,
                         "mean_travel_s": travel, "stored_wait_s": ref})
            print(f"done in {time.time() - t0:5.1f} s")

    print()
    print(f"  {'Controller':<28s} {'Seed':>4s} {'Vehicles':>8s} {'Avg wait (s/veh)':>17s} "
          f"{'Avg travel (s)':>15s} {'Stored wait':>12s}")
    print("  " + "-" * 88)
    for r in rows:
        ref = f"{r['stored_wait_s']:.2f}" if r["stored_wait_s"] is not None else "n/a"
        print(f"  {r['controller']:<28s} {r['seed']:>4d} {r['vehicles']:>8d} {r['mean_wait_s']:>17.2f} "
              f"{r['mean_travel_s']:>15.2f} {ref:>12s}")
    print("  (Stored = results/final_checks/metric_independent_baselines.csv; only seeds 1-3 exist.)")
    print("  Lower waiting time is better.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_".join(str(s) for s in args.seeds)
    out_csv = OUT_DIR / f"comparison_seeds_{tag}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Saved table: {out_csv.relative_to(ROOT_DIR)}")

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [c[0] for c in controllers]
        means = [sum(r["mean_wait_s"] for r in rows if r["controller"] == n) / len(args.seeds) for n in names]
        fig, ax = plt.subplots(figsize=(9, 4.8))
        bars = ax.bar(names, means, color=["#8c8c8c", "#5b8ff9", "#3a5bbf", "#e8684a", "#6dc8ec"][: len(names)])
        ax.bar_label(bars, fmt="%.1f s")
        ax.set_ylabel("Average waiting time per vehicle (s)")
        ax.set_title(f"Same traffic, different controllers (seed(s) {', '.join(map(str, args.seeds))})")
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=12)
        plt.tight_layout()
        out_png = OUT_DIR / f"comparison_seeds_{tag}.png"
        plt.savefig(out_png, dpi=150)
        print(f"  Saved plot : {out_png.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
