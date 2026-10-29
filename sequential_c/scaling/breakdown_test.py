"""
Per-stage timestep breakdown for the sequential C++ SPH solver at a single particle count.

Same scene and measurement as scaling_test.py (hydrostatic scene at the dx that gives the
requested count, MAX_STEPS physics steps, 10-step window mean +/- stdev), but run with
validate.exe --breakdown so each stage is timed separately:
A) density neighbourhood search, B) density computation, C) force neighbourhood search,
D) force computation, E) pressure computation, F) boundary condition.

Usage:
    python sequential_c/scaling/breakdown_test.py              # N=2000
    python sequential_c/scaling/breakdown_test.py --count 1000
    python sequential_c/scaling/breakdown_test.py --replot
"""

import argparse
import json
import os
import subprocess
import time

from scaling_test import ELASTIC_EXE, HERE, SEQ_C, MAX_STEPS, HYDRO_W, HYDRO_FILL_H, generate_scene

STAGES = [
    ("A", "Density NBHD search"),
    ("B", "Density computation"),
    ("C", "Force NBHD search"),
    ("D", "Force computation"),
    ("E", "Pressure computation"),
    ("F", "Boundary condition"),
]


def run(n_target):
    scene_json, scene_csv = generate_scene(n_target)
    out_csv = os.path.join(HERE, "_scratch_out.csv")
    out_steps = os.path.join(HERE, "_scratch_steps.csv")
    proc = subprocess.run(
        [ELASTIC_EXE, scene_json, scene_csv, out_csv, out_steps, "--max-steps", str(MAX_STEPS), "--breakdown"],
        capture_output=True, text=True,
    )
    for p in (out_csv, out_steps):
        if os.path.exists(p):
            os.remove(p)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])

    fields = {}
    for line in proc.stdout.splitlines():
        if line.startswith(("done:", "window_stats:", "breakdown:")):
            for tok in line.split(":", 1)[1].split():
                k, v = tok.split("=")
                fields[k] = float(v)

    return {
        "particles_requested": n_target,
        "n_total": int(fields["n_fluid"]),
        "steps": int(fields["steps"]),
        "ms_per_step": fields["mean_ms_per_step"],
        "ms_per_step_stdev": fields["stdev_ms_per_step"],
        "stages": {k: {"name": name, "ms": fields[f"{k}_mean"], "stdev": fields[f"{k}_stdev"]} for k, name in STAGES},
    }


def make_plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
    labels = [f"{k}) {name}".replace(" ", "\n", 2).replace("\n", " ", 1) for k, name in STAGES]
    vals = [result["stages"][k]["ms"] for k, _ in STAGES]

    plt.figure(figsize=(8, 4.5), dpi=140)
    ax = plt.gca()
    xs = range(len(STAGES))
    ax.bar(xs, vals, color=colours, alpha=0.9, width=0.8)
    for x, v in zip(xs, vals):
        ax.text(x, v + max(vals) * 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_ylabel("Compute time per timestep (ms)", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--json")
    ap.add_argument("--plot")
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()

    json_path = args.json or os.path.join(SEQ_C, "data", f"breakdown_n{args.count}.json")
    plot_path = args.plot or os.path.join(SEQ_C, "data", f"breakdown_n{args.count}.png")

    if args.replot:
        with open(json_path) as f:
            result = json.load(f)["result"]
    else:
        result = run(args.count)
        with open(json_path, "w") as f:
            json.dump({
                "metric": "per-stage steady per-step cost, sequential C++ hydrostatic scene, elastic boundary",
                "source": "sequential_c/validate_main.cpp --breakdown",
                "max_steps_measured": MAX_STEPS,
                "hydrostatic_geometry": {"W": HYDRO_W, "fill_height": HYDRO_FILL_H},
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "result": result,
            }, f, indent=2)
        for k, name in STAGES:
            s = result["stages"][k]
            print(f"{k}) {name:<22} {s['ms']:>9.4f} ms  +/- {s['stdev']:.4f}")
        print(f"total (breakdown mode)     {result['ms_per_step']:>9.4f} ms  +/- {result['ms_per_step_stdev']:.4f}  n_total={result['n_total']}")

    make_plot(result, plot_path)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
