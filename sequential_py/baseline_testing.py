"""
Milestone 1 baseline harness for the naive O(N^2) SPH solver.

Sweeps particle counts and records *steady-state* per-timestep cost for each.
Particles are spawned in the middle of the box, so the first few hundred frames
are the fluid falling and sloshing; we wait for the step time to stop drifting
before we start measuring.

Physics time and render time are recorded separately -- "ms per timestep" means
the solver only (density -> force -> boundary -> integrate), which is the number
that matters for the parallelisation work. Throughput is reported as
particle-updates per second (N / t_step), so different N are directly comparable.

Each particle count runs in its own subprocess, because naive_py.py builds its
window, its box and its whole particle list at import time from sys.argv -- a
fresh process is the only clean way to get a fresh sim.

Outputs (written next to this script):
    results.json                 per-N steady-state timings + stage split
    baseline_scaling.png         ms/timestep vs particle count  (report Fig. 7 style)
    baseline_stage_breakdown.png section-wise bar charts        (report Fig. 2/4/5/6 style)

Usage:
    python sequential_py/baseline_testing.py                    # default 10 -> 1000 sweep
    python sequential_py/baseline_testing.py --counts 10,50,100
    python sequential_py/baseline_testing.py --run 50           # internal (one N)

naive_py.py is imported unmodified; its physics functions are called in exactly
the order its own update() calls them.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.path.join(HERE, "results.json")
DEFAULT_PLOT = os.path.join(HERE, "baseline_scaling.png")
DEFAULT_BARS = os.path.join(HERE, "baseline_stage_breakdown.png")

# 10..100 by 10, 200..1000 by 100, then 1200..2000 by 200. The naive solver is
# O(N^2), so the step widens as N grows -- a uniform step to 2000 would spend the
# entire sweep on the tail for no extra resolution.
DEFAULT_COUNTS = (list(range(10, 101, 10))
                  + list(range(200, 1001, 100))
                  + list(range(1200, 2001, 200)))

# --- steady-state detection knobs -------------------------------------------
MIN_WARMUP_FRAMES = 120      # never call it settled before this many frames
MAX_WARMUP_FRAMES = 3000     # frame ceiling on warmup
WARMUP_BUDGET_S = 45.0       # wall-clock ceiling on warmup (large N never settles)
WINDOW = 30                  # frames per comparison window
STABLE_TOL = 0.05            # <5% drift between consecutive windows
STABLE_WINDOWS = 3           # this many consecutive stable windows => settled

# --- measurement knobs ------------------------------------------------------
MEASURE_FRAMES = 300         # frame ceiling on the measured window
MEASURE_BUDGET_S = 15.0      # wall-clock ceiling on the measured window
MIN_MEASURE_FRAMES = 10      # ...but always take at least this many

STAGE_KEYS = [
    ("1. Density Calculation", "density_ms"),
    ("2. Force Calculation", "force_ms"),
    ("3. Box Collisions", "box_collision_ms"),
    ("4. Integration", "integration_ms"),
]
STAGE_LABELS = ["Density Calculation", "Force Calculation",
                "Boundary Condition", "Numerical Integrator"]
STAGE_FIELDS = ["density_ms", "force_ms", "box_collision_ms", "integration_ms"]


def run_one(n):
    """Drive naive_py's own update loop for `n` particles and return metrics."""
    # naive_py reads the particle count off sys.argv at import time.
    sys.argv = ["naive_py.py", str(n)]
    sys.path.insert(0, HERE)

    import naive_py  # noqa: E402  (import is deliberately late / argv-dependent)

    win = naive_py.win
    batch = naive_py.main_batch
    dt = naive_py.dt

    # Uncapped frame rate: with vsync on, flip() would peg every result at the
    # monitor refresh and we would measure the display, not the solver.
    try:
        win.set_vsync(False)
    except Exception:
        pass

    def frame():
        """One frame. Returns (physics_seconds, render_seconds)."""
        t0 = time.perf_counter()
        # naive_py.update() hard-exits at frame 5 (that is how the original
        # reports its 5-frame total). Reset the counter so the guard never
        # fires; the physics it performs is otherwise untouched.
        naive_py.frame_counter = 0
        naive_py.update(dt)
        t1 = time.perf_counter()

        win.switch_to()
        win.dispatch_events()
        win.clear()
        batch.draw()
        win.flip()
        return t1 - t0, time.perf_counter() - t1

    # --- warm up until the step time stops drifting -------------------------
    step_times = []
    prev_window = None
    stable_run = 0
    settled_at = None
    warmup_start = time.perf_counter()

    while len(step_times) < MAX_WARMUP_FRAMES:
        step_times.append(frame()[0])

        if len(step_times) >= MIN_WARMUP_FRAMES and len(step_times) % WINDOW == 0:
            cur_window = statistics.median(step_times[-WINDOW:])
            if prev_window is not None:
                drift = abs(cur_window - prev_window) / prev_window
                stable_run = stable_run + 1 if drift < STABLE_TOL else 0
                if stable_run >= STABLE_WINDOWS:
                    settled_at = len(step_times)
                    break
            prev_window = cur_window

        if time.perf_counter() - warmup_start > WARMUP_BUDGET_S:
            break  # out of time budget; measure anyway and flag settled=False

    warmup_frames = len(step_times)

    # --- measure ------------------------------------------------------------
    # Reset the profiler so its exponential average reflects the measured
    # window only, not the transient we just discarded.
    naive_py.profiler.timings = {}

    physics, render = [], []
    measure_start = time.perf_counter()
    while len(physics) < MEASURE_FRAMES:
        p, r = frame()
        physics.append(p)
        render.append(r)
        if (len(physics) >= MIN_MEASURE_FRAMES
                and time.perf_counter() - measure_start > MEASURE_BUDGET_S):
            break

    step_ms = statistics.mean(physics) * 1000.0
    render_ms = statistics.mean(render) * 1000.0
    stages = dict(naive_py.profiler.timings)

    return {
        "particles": n,
        # when this point was collected -- matters when merging runs from
        # different sessions, since machine state affects timings
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        # primary metric: solver cost per timestep
        "ms_per_timestep": step_ms,
        "ms_per_timestep_median": statistics.median(physics) * 1000.0,
        "ms_per_timestep_stdev": (statistics.stdev(physics) * 1000.0) if len(physics) > 1 else 0.0,
        "ms_per_timestep_min": min(physics) * 1000.0,
        "ms_per_timestep_max": max(physics) * 1000.0,
        # normalised throughput: how many particles the solver advances per second
        "particle_updates_per_sec": n / (step_ms / 1000.0) if step_ms > 0 else 0.0,
        # render is reported but excluded from the solver metrics above
        "render_ms": render_ms,
        "total_frame_ms": step_ms + render_ms,
        "fps_physics_only": 1000.0 / step_ms if step_ms > 0 else 0.0,
        "fps_with_render": 1000.0 / (step_ms + render_ms) if (step_ms + render_ms) > 0 else 0.0,
        "settled": settled_at is not None,
        "warmup_frames": warmup_frames,
        "measured_frames": len(physics),
        # naive_py's own per-stage profiler (exponentially smoothed, ms)
        "stages_ms": {out: stages.get(key) for key, out in STAGE_KEYS},
    }


def _report_axes(ax, cross_at_origin=True):
    """Textbook-style axes, matching final_plot.py.

    cross_at_origin pins both spines to data coordinate 0. That is right for a
    continuous x axis, but wrong for a categorical bar chart, where x=0 is the
    centre of the *first bar* -- pinning there draws the y axis straight through
    it. Bar charts pass cross_at_origin=False and keep the spines at the edges.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    if cross_at_origin:
        ax.spines["bottom"].set_position(("data", 0))
        ax.spines["left"].set_position(("data", 0))
    ax.tick_params(axis="both", labelsize=10, width=1.2)
    ax.grid(False)


def make_scaling_plot(results, path):
    """ms per timestep vs particle count, in the style of the report's Fig. 7."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    xs = np.array([r["particles"] for r in results], dtype=float)
    ys = np.array([r["ms_per_timestep"] for r in results], dtype=float)

    plt.figure(figsize=(9, 5.5), dpi=140)
    ax = plt.gca()
    naive_colour = "#C44E52"

    ax.plot(xs, ys, color=naive_colour, linewidth=2.2, alpha=0.9)
    ax.scatter(xs, ys, color=naive_colour, s=14, label="Naive Python (sequential baseline)",
               alpha=0.95)

    # Least-squares quadratic-through-origin fit, to show the O(N^2) trend the
    # baseline is meant to establish.
    if len(xs) >= 3:
        k = float(np.sum(ys * xs ** 2) / np.sum(xs ** 4))
        fine = np.linspace(0, xs.max(), 200)
        ax.plot(fine, k * fine ** 2, color=naive_colour, linewidth=1.4,
                linestyle="--", alpha=0.55, label="$O(N^2)$ fit")

    ax.set_xlabel("Particle Count", fontsize=12)
    ax.set_ylabel("Total Time per Frame (ms)", fontsize=12)
    ax.set_xlim(0, xs.max() * 1.1)
    ax.set_ylim(0, ys.max() * 1.15)
    _report_axes(ax)
    ax.legend(frameon=False, fontsize=11)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def make_stage_breakdown(results, path, max_panels=6):
    """Section-wise time-per-step bar charts, in the style of the report's Fig. 2/4/5/6."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Pick up to `max_panels` evenly spaced particle counts so the figure stays legible.
    if len(results) > max_panels:
        idx = np.linspace(0, len(results) - 1, max_panels).round().astype(int)
        panels = [results[i] for i in sorted(set(idx.tolist()))]
    else:
        panels = results

    cols = min(3, len(panels))
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.4 * rows), dpi=140)
    axes = np.atleast_1d(axes).ravel()

    colours = ["#C44E52", "#4C72B0", "#55A868", "#8172B2"]

    for ax, r in zip(axes, panels):
        vals = [r["stages_ms"].get(f) or 0.0 for f in STAGE_FIELDS]
        # wrap "Density Calculation" -> "Density\nCalculation" so four labels fit
        bars = ax.bar([l.replace(" ", "\n") for l in STAGE_LABELS], vals,
                      color=colours, width=0.6)

        ax.set_title(f"Particle Count: {r['particles']} "
                     f"({r['fps_physics_only']:.1f} FPS)", fontsize=11)

        ax.set_ylabel("Time per Step (ms)", fontsize=10)
        ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)
        _report_axes(ax, cross_at_origin=False)
        ax.tick_params(axis="x", labelsize=8)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    for ax in axes[len(panels):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def sweep(counts, json_path, plot_path, bars_path, append=False):
    # In append mode, keep whatever is already in the JSON and only run the
    # counts that are missing. Points already measured are not re-run -- their
    # timings came off a machine in a particular power/thermal state and
    # re-measuring them piecemeal would make the curve inconsistent.
    existing = []
    if append and os.path.exists(json_path):
        with open(json_path) as f:
            existing = json.load(f).get("results", [])
        have = {r["particles"] for r in existing}
        reused = sorted(c for c in counts if c in have)
        counts = [c for c in counts if c not in have]
        if reused:
            print(f"Reusing {len(reused)} existing points: {reused}")
        if not counts:
            print("Nothing new to measure.")

    results = []
    print(f"{'N':>6} {'ms/step':>10} {'upd/sec':>12} {'render ms':>10} {'settled':>8} "
          f"{'density':>9} {'force':>9} {'bound':>8} {'integ':>8}")
    print("-" * 88)

    for n in counts:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--run", str(n)],
            capture_output=True, text=True,
        )
        # The child prints one JSON line prefixed with RESULT:; anything else on
        # stdout (NumPy warnings, pyglet chatter) is ignored.
        payload = None
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT:"):
                payload = json.loads(line[len("RESULT:"):])
        if payload is None:
            print(f"{n:>6}   FAILED (exit {proc.returncode})")
            if proc.stderr.strip():
                print("         " + proc.stderr.strip().splitlines()[-1])
            continue

        st = payload["stages_ms"]
        fmt = lambda k, w: (f"{st[k]:>{w}.3f}" if st.get(k) is not None else f"{'-':>{w}}")  # noqa: E731
        print(f"{n:>6} {payload['ms_per_timestep']:>10.3f} "
              f"{payload['particle_updates_per_sec']:>12.1f} {payload['render_ms']:>10.3f} "
              f"{str(payload['settled']):>8} "
              f"{fmt('density_ms', 9)} {fmt('force_ms', 9)} "
              f"{fmt('box_collision_ms', 8)} {fmt('integration_ms', 8)}")

        results.append(payload)

    if not results and not existing:
        print("No successful runs; nothing written.")
        return

    # Merge: a freshly measured point replaces an existing one for the same N.
    merged = {r["particles"]: r for r in existing}
    merged.update({r["particles"]: r for r in results})
    results = [merged[k] for k in sorted(merged)]

    document = {
        "metric": "steady-state solver cost per timestep, naive O(N^2) SPH (render excluded)",
        "source": "sequential_py/naive_py.py",
        "physics": "pressure + Monaghan artificial viscosity + gravity (0, -40), forward Euler",
        "settle_criterion": {
            "min_warmup_frames": MIN_WARMUP_FRAMES,
            "max_warmup_frames": MAX_WARMUP_FRAMES,
            "warmup_budget_s": WARMUP_BUDGET_S,
            "window_frames": WINDOW,
            "relative_tolerance": STABLE_TOL,
            "consecutive_stable_windows": STABLE_WINDOWS,
        },
        "measure_limits": {
            "max_frames": MEASURE_FRAMES,
            "budget_s": MEASURE_BUDGET_S,
            "min_frames": MIN_MEASURE_FRAMES,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "results": results,
    }
    with open(json_path, "w") as f:
        json.dump(document, f, indent=2)
    print("-" * 88)
    print(f"Wrote {len(results)} runs to {json_path}")

    make_scaling_plot(results, plot_path)
    print(f"Wrote scaling plot to {plot_path}")
    make_stage_breakdown(results, bars_path)
    print(f"Wrote stage breakdown to {bars_path}")

    unsettled = [r["particles"] for r in results if not r["settled"]]
    if unsettled:
        print(f"NOTE: did not reach steady state within the warmup budget for N = {unsettled}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=int, help=argparse.SUPPRESS)  # internal child mode
    ap.add_argument("--counts", help="comma-separated particle counts (overrides the default sweep)")
    ap.add_argument("--replot", action="store_true",
                    help="regenerate both figures from an existing results.json, without re-running the sweep")
    ap.add_argument("--append", action="store_true",
                    help="keep points already in results.json and only measure the missing counts")
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--plot", default=DEFAULT_PLOT)
    ap.add_argument("--bars", default=DEFAULT_BARS)
    args = ap.parse_args()

    if args.run is not None:
        print("RESULT:" + json.dumps(run_one(args.run)))
        return

    if args.replot:
        with open(args.json) as f:
            results = json.load(f)["results"]
        make_scaling_plot(results, args.plot)
        make_stage_breakdown(results, args.bars)
        print(f"Replotted {len(results)} runs from {args.json}")
        print(f"  {args.plot}\n  {args.bars}")
        return

    counts = ([int(c) for c in args.counts.split(",")] if args.counts else DEFAULT_COUNTS)
    sweep(counts, args.json, args.plot, args.bars, append=args.append)


if __name__ == "__main__":
    main()
