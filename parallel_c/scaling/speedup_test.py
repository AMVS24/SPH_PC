"""
Thread-count speedup sweep for the OpenMP SPH solver (parallel_c/), across particle count.

Same scene and measurement as sequential_c/scaling/scaling_test.py: for each particle
count, generate a hydrostatic scene at the dx that gives that count, run
parallel_c/build/validate.exe for MAX_STEPS steps, and take ms/step as the mean +/- stdev
of 10-step window averages (validate.exe's window_stats: line). Each count is run at every
thread count in THREADS (validate.exe --threads N); speedup = ms/step at 1 thread divided
by ms/step at N threads.

Raw results are written to data/speedup_results.json before any plotting happens.

Usage:
    python parallel_c/scaling/speedup_test.py
    python parallel_c/scaling/speedup_test.py --counts 500,1000 --threads 1,4,8
    python parallel_c/scaling/speedup_test.py --replot
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PAR_C = os.path.dirname(HERE)
REPO = os.path.dirname(PAR_C)

EXE = os.path.join(PAR_C, "build", "validate.exe")
SCENES_PY = os.path.join(REPO, "validation", "scenes.py")

SCENES_DIR = os.path.join(HERE, "scenes")
DEFAULT_JSON = os.path.join(PAR_C, "data", "speedup_results.json")
DEFAULT_PLOT = os.path.join(PAR_C, "data", "speedup_plot.png")

DEFAULT_COUNTS = list(range(100, 1001, 100)) + list(range(1200, 3001, 200))
DEFAULT_THREADS = [1, 2, 4, 8, 16, 32, 64]

MAX_STEPS = 150

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


def run_one(scene_json, scene_csv, n_target, threads):
    out_csv = os.path.join(HERE, "_scratch_out.csv")
    out_steps = os.path.join(HERE, "_scratch_steps.csv")
    proc = subprocess.run(
        [EXE, scene_json, scene_csv, out_csv, out_steps, "--max-steps", str(MAX_STEPS), "--threads", str(threads)],
        capture_output=True, text=True,
    )
    for p in (out_csv, out_steps):
        if os.path.exists(p):
            os.remove(p)
    if proc.returncode != 0:
        return {"particles_requested": n_target, "threads": threads, "error": proc.stderr[-2000:]}

    done = parse_fields(proc.stdout, "done:")
    window = parse_fields(proc.stdout, "window_stats:")
    return {
        "particles_requested": n_target,
        "threads": int(done["threads"]),
        "n_total": int(done["n_fluid"]),
        "steps": int(done["steps"]),
        "ms_per_step": float(window["mean_ms_per_step"]),
        "ms_per_step_stdev": float(window["stdev_ms_per_step"]),
    }


def sweep(counts, thread_counts, json_path):
    if not os.path.exists(EXE):
        sys.exit(f"binary not found: {EXE} (build it with: mingw32-make validate)")
    os.makedirs(SCENES_DIR, exist_ok=True)

    results = []
    print(f"{'N req':>7} {'n_total':>8} {'threads':>8} {'ms/step':>10} {'+/-stdev':>9} {'speedup':>8}")
    print("-" * 56)
    for n_target in counts:
        scene_json, scene_csv = generate_scene(n_target)
        base = None
        for t in thread_counts:
            r = run_one(scene_json, scene_csv, n_target, t)
            results.append(r)
            if "error" in r:
                print(f"{n_target:>7} {'':>8} {t:>8}   FAILED")
                continue
            if t == 1:
                base = r["ms_per_step"]
            sp = base / r["ms_per_step"] if base else float("nan")
            print(f"{n_target:>7} {r['n_total']:>8} {t:>8} {r['ms_per_step']:>10.4f} {r['ms_per_step_stdev']:>9.4f} {sp:>7.2f}x")

    document = {
        "metric": "steady per-step solver cost vs OpenMP thread count, parallel C++ hydrostatic scene, elastic boundary",
        "source": "parallel_c/validate_main.cpp --threads",
        "max_steps_measured": MAX_STEPS,
        "hydrostatic_geometry": {"W": HYDRO_W, "fill_height": HYDRO_FILL_H},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
    }
    with open(json_path, "w") as f:
        json.dump(document, f, indent=2)
    print("-" * 56)
    print(f"Wrote {len(results)} runs to {json_path}")
    return results


def make_plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"]

    ok = [r for r in results if "error" not in r]
    base = {r["particles_requested"]: r for r in ok if r["threads"] == 1}
    thread_counts = sorted({r["threads"] for r in ok})

    plt.figure(figsize=(9, 5.5), dpi=140)
    ax = plt.gca()

    for c, t in zip(colours, thread_counts):
        pts = [r for r in ok if r["threads"] == t and r["particles_requested"] in base]
        pts.sort(key=lambda r: r["n_total"])
        xs = np.array([r["n_total"] for r in pts], dtype=float)
        b = np.array([base[r["particles_requested"]]["ms_per_step"] for r in pts])
        bs = np.array([base[r["particles_requested"]]["ms_per_step_stdev"] for r in pts])
        m = np.array([r["ms_per_step"] for r in pts])
        ms = np.array([r["ms_per_step_stdev"] for r in pts])
        ys = b / m
        errs = ys * np.sqrt((bs / b) ** 2 + (ms / m) ** 2)

        ax.plot(xs, ys, color=c, linewidth=2.0, alpha=0.85)
        ax.fill_between(xs, ys - errs, ys + errs, color=c, alpha=0.15, linewidth=0)
        ax.scatter(xs, ys, color=c, s=16, alpha=0.95, label=f"{t} thread{'s' if t > 1 else ''}")

    ax.set_xlabel("Particle count", fontsize=12)
    ax.set_ylabel("Speedup vs 1 thread", fontsize=12)
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
    ap.add_argument("--counts", help="comma-separated particle counts")
    ap.add_argument("--threads", help="comma-separated thread counts (must include 1)")
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--plot", default=DEFAULT_PLOT)
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()

    if args.replot:
        with open(args.json) as f:
            results = json.load(f)["results"]
        make_plot(results, args.plot)
        print(f"Replotted {len(results)} runs to {args.plot}")
        return

    counts = [int(c) for c in args.counts.split(",")] if args.counts else DEFAULT_COUNTS
    thread_counts = [int(t) for t in args.threads.split(",")] if args.threads else DEFAULT_THREADS
    results = sweep(counts, thread_counts, args.json)
    make_plot(results, args.plot)
    print(f"Wrote speedup plot to {args.plot}")


if __name__ == "__main__":
    main()
