"""
Scaling sweep for the sequential C++ SPH solver (sequential_c/), across particle count.

Measures sequential_c/build/validate.exe (built from validate_main.cpp, elastic hard-wall
boundary) on the hydrostatic scene.

For each particle count in the sweep: generate a hydrostatic scene at the dx that gives
that particle count, run the solver for a fixed number of steps (not to the scene's full
t_end -- we only need steady per-step cost), and record ms/step as the mean +/- stdev of
10-step window averages (validate.exe's window_stats: line).

Raw results are written to data/scaling_results.json before any plotting happens, so the
underlying numbers survive independently of the plot.

Usage:
    python sequential_c/scaling/scaling_test.py                 # full default sweep
    python sequential_c/scaling/scaling_test.py --counts 10,50,100
    python sequential_c/scaling/scaling_test.py --replot         # regenerate the plot only
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SEQ_C = os.path.dirname(HERE)
REPO = os.path.dirname(SEQ_C)

ELASTIC_EXE = os.path.join(SEQ_C, "build", "validate.exe")
SCENES_PY = os.path.join(REPO, "validation", "scenes.py")

SCENES_DIR = os.path.join(HERE, "scenes")
DEFAULT_JSON = os.path.join(SEQ_C, "data", "scaling_results.json")
DEFAULT_PLOT = os.path.join(SEQ_C, "data", "scaling_plot.png")

DEFAULT_COUNTS = (list(range(10, 101, 10))
                  + list(range(200, 1001, 100))
                  + list(range(1200, 3001, 200)))

MAX_STEPS = 150     # physics steps per measurement -- enough for a stable ms/step average
                     # without paying for the scene's full (multi-thousand-step) t_end.

# hydrostatic geometry constants (validation/scenes.py's hydrostatic()): W x H_fill fill area.
HYDRO_W = 0.5
HYDRO_FILL_H = 0.3


def dx_for_particle_count(n_target):
    return math.sqrt(HYDRO_W * HYDRO_FILL_H / n_target)


def generate_scene(n_target):
    dx = dx_for_particle_count(n_target)
    proc = subprocess.run(
        [sys.executable, SCENES_PY, "hydrostatic", "--dx", f"{dx:.8f}", "--out", SCENES_DIR],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"scene generation failed for n={n_target}: {proc.stderr}")
    return os.path.join(SCENES_DIR, "hydrostatic.json"), os.path.join(SCENES_DIR, "hydrostatic.init.csv")


def parse_fields(stdout, prefix):
    fields = {}
    for line in stdout.splitlines():
        if line.startswith(prefix):
            for tok in line[len(prefix):].split():
                if "=" in tok:
                    k, v = tok.split("=")
                    fields[k] = v
    return fields


def run_one(n_target):
    scene_json, scene_csv = generate_scene(n_target)

    out_csv = os.path.join(HERE, "_scratch_out.csv")
    out_steps = os.path.join(HERE, "_scratch_steps.csv")

    t0 = time.perf_counter()
    proc = subprocess.run(
        [ELASTIC_EXE, scene_json, scene_csv, out_csv, out_steps, "--max-steps", str(MAX_STEPS)],
        capture_output=True, text=True,
    )
    wall_s = time.perf_counter() - t0

    for p in (out_csv, out_steps):
        if os.path.exists(p):
            os.remove(p)

    if proc.returncode != 0:
        return {"particles_requested": n_target, "error": proc.stderr[-2000:]}

    done = parse_fields(proc.stdout, "done:")
    window = parse_fields(proc.stdout, "window_stats:")

    return {
        "particles_requested": n_target,
        "n_total": int(done["n_fluid"]),
        "steps": int(done["steps"]),
        "wall_s": wall_s,
        "window_size": int(window["window_size"]),
        "n_windows": int(window["n_windows"]),
        "ms_per_step": float(window["mean_ms_per_step"]),
        "ms_per_step_stdev": float(window["stdev_ms_per_step"]),
    }


def sweep(counts, json_path):
    if not os.path.exists(ELASTIC_EXE):
        sys.exit(f"binary not found: {ELASTIC_EXE} (build it with: mingw32-make validate)")
    os.makedirs(SCENES_DIR, exist_ok=True)

    results = []
    print(f"{'N req':>7} {'n_total':>8} {'steps':>6} {'ms/step':>10} {'+/-stdev':>9}")
    print("-" * 44)

    for n_target in counts:
        r = run_one(n_target)
        results.append(r)
        if "error" in r:
            print(f"{n_target:>7}   FAILED: {r['error'].strip().splitlines()[-1] if r['error'].strip() else '(no stderr)'}")
            continue
        print(f"{n_target:>7} {r['n_total']:>8} {r['steps']:>6} "
              f"{r['ms_per_step']:>10.4f} {r['ms_per_step_stdev']:>9.4f}")

    document = {
        "metric": "steady per-step solver cost, sequential C++ hydrostatic scene, elastic boundary",
        "source": "sequential_c/validate_main.cpp",
        "max_steps_measured": MAX_STEPS,
        "hydrostatic_geometry": {"W": HYDRO_W, "fill_height": HYDRO_FILL_H},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
    }
    with open(json_path, "w") as f:
        json.dump(document, f, indent=2)
    print("-" * 44)
    print(f"Wrote {len(results)} runs to {json_path}")
    return results


def make_plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    c = "#4C72B0"

    plt.figure(figsize=(9, 5.5), dpi=140)
    ax = plt.gca()

    pts = [r for r in results if "error" not in r]
    if pts:
        xs = np.array([r["n_total"] for r in pts], dtype=float)
        ys = np.array([r["ms_per_step"] for r in pts], dtype=float)
        errs = np.array([r.get("ms_per_step_stdev") or 0.0 for r in pts], dtype=float)
        order = np.argsort(xs)
        xs, ys, errs = xs[order], ys[order], errs[order]

        ax.plot(xs, ys, color=c, linewidth=2.0, alpha=0.85)
        ax.fill_between(xs, ys-errs, ys+errs, color=c, alpha=0.15, linewidth=0)
        ax.scatter(xs, ys, color=c, s=16, label="Sequential C++ (elastic boundary)", alpha=0.95)

        if len(xs) >= 3:
            k = float(np.sum(ys * xs ** 2) / np.sum(xs ** 4))
            fine = np.linspace(0, xs.max(), 200)
            ax.plot(fine, k * fine ** 2, color=c, linewidth=1.2, linestyle="--", alpha=0.45)

    ax.set_xlabel("Particle count", fontsize=12)
    ax.set_ylabel("ms per physics step", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    ax.tick_params(axis="both", labelsize=10, width=1.2)
    ax.grid(False)
    ax.legend(frameon=False, fontsize=11)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", help="comma-separated particle counts (overrides the default sweep)")
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--plot", default=DEFAULT_PLOT)
    ap.add_argument("--replot", action="store_true", help="regenerate the plot from an existing results file")
    args = ap.parse_args()

    if args.replot:
        with open(args.json) as f:
            results = json.load(f)["results"]
        make_plot(results, args.plot)
        print(f"Replotted {len(results)} runs to {args.plot}")
        return

    counts = [int(c) for c in args.counts.split(",")] if args.counts else DEFAULT_COUNTS
    results = sweep(counts, args.json)
    make_plot(results, args.plot)
    print(f"Wrote scaling plot to {args.plot}")


if __name__ == "__main__":
    main()
