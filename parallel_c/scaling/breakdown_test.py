"""
Per-stage compute-time breakdown (validate.exe --breakdown) for sequential, OpenMP
all-pairs (before spatial hash) and OpenMP + spatial hash.

Stages: A) density neighbourhood search, B) density computation, C) force neighbourhood
search, D) force computation, E) pressure computation, F) boundary condition. For the
spatial-hash variant, building the hash grid is counted in A.

Two outputs, both from data/breakdown_results.json (saved after every run, resumable):
    --vs-n     breakdown vs particle count  -> data/breakdown_vs_n_<variant>_t<threads>.png
    --count N  breakdown at one count       -> data/breakdown_n<N>_<variant>_t<threads>.png

Usage:
    python parallel_c/scaling/breakdown_test.py --variant hash --threads 24 --vs-n
    python parallel_c/scaling/breakdown_test.py --variant prehash --threads 24 --count 30000
    python parallel_c/scaling/breakdown_test.py --replot
"""

import argparse
import os
import time

from bench_common import (DATA, EXES, LABELS, ORDER, COLOURS, generate_scene, steps_for, run_validate,
                          load_json, save_json, style_axes)

DEFAULT_JSON = os.path.join(DATA, "breakdown_results.json")

STAGES = [
    ("A", "Density NBHD search"),
    ("B", "Density computation"),
    ("C", "Force NBHD search"),
    ("D", "Force computation"),
    ("E", "Pressure computation"),
    ("F", "Boundary condition"),
]

VS_N_COUNTS = {
    "sequential": list(range(500, 10001, 500)),
    "prehash": list(range(500, 10001, 500)) + list(range(20000, 100001, 10000)),
    "hash": list(range(4000, 100001, 4000)),
}
DEFAULT_CAP_MS = float("inf")


def key(variant, threads, n):
    return f"{variant}|{threads}|{n}"


def measure(doc, json_path, variant, threads, counts, cap_ms):
    if not os.path.exists(EXES[variant]):
        raise SystemExit(f"binary not found: {EXES[variant]}")
    if variant == "sequential":
        threads = 1
    results = doc["results"]
    last = None
    for r in sorted((r for r in results.values() if r["variant"] == variant and r["threads"] == threads and "error" not in r),
                    key=lambda r: r["particles_requested"]):
        last = (r["particles_requested"], r["ms_per_step"])

    for n in counts:
        if key(variant, threads, n) in results:
            continue
        est = last[1] * (n / last[0]) ** ORDER[variant] if last else None
        r = run_validate(variant, generate_scene(n), threads, steps_for(est), breakdown=True)
        r.update({"variant": variant, "threads": threads, "particles_requested": n})
        results[key(variant, threads, n)] = r
        doc["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        save_json(json_path, doc)
        if "error" in r:
            print(f"{variant} t={threads} N={n} FAILED: {r['error'].strip().splitlines()[-1]}")
            break
        last = (n, r["ms_per_step"])
        stages = "  ".join(f"{k}={r['stages'][k]['ms']:.3f}" for k, _ in STAGES)
        print(f"{variant:>10} t={threads:<3} N={n:<7} n_total={r['n_total']:<7} steps={r['steps']:<4} total={r['ms_per_step']:.3f} ms  {stages}", flush=True)
        if r["ms_per_step"] > cap_ms:
            break


def title_for(variant, threads):
    return LABELS[variant] if variant == "sequential" else f"{LABELS[variant]}, {threads} thread{'s' if threads > 1 else ''}"


def plot_vs_n(results, variant, threads, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pts = sorted((r for r in results.values() if r["variant"] == variant and r["threads"] == threads and "error" not in r),
                 key=lambda r: r["n_total"])
    if len(pts) < 2:
        return False
    xs = np.array([r["n_total"] for r in pts], dtype=float)

    plt.figure(figsize=(9, 5.5), dpi=140)
    ax = plt.gca()
    for c, (k, name) in zip(COLOURS, STAGES):
        ys = np.array([r["stages"][k]["ms"] for r in pts])
        es = np.array([r["stages"][k]["stdev"] for r in pts])
        ax.plot(xs, ys, color=c, linewidth=2.0, alpha=0.85)
        ax.fill_between(xs, ys - es, ys + es, color=c, alpha=0.15, linewidth=0)
        ax.scatter(xs, ys, color=c, s=12, alpha=0.95, label=f"{k}) {name}")

    ax.set_xlabel("Particle count", fontsize=12)
    ax.set_ylabel("Compute time per timestep (ms)", fontsize=12)
    ax.set_title(title_for(variant, threads), fontsize=12, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def plot_single(r, variant, threads, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{k}) {name}".replace(" ", "\n", 2).replace("\n", " ", 1) for k, name in STAGES]
    vals = [r["stages"][k]["ms"] for k, _ in STAGES]

    plt.figure(figsize=(8, 4.5), dpi=140)
    ax = plt.gca()
    xs = range(len(STAGES))
    ax.bar(xs, vals, color=COLOURS[:len(STAGES)], alpha=0.9, width=0.8)
    for x, v in zip(xs, vals):
        ax.text(x, v + max(vals) * 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_ylabel("Compute time per timestep (ms)", fontsize=11)
    ax.set_title(f"{title_for(variant, threads)}, N = {r['n_total']}", fontsize=11, loc="left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def make_plots(doc, out_dir, singles):
    results = doc["results"]
    series = {(r["variant"], r["threads"]) for r in results.values()}
    for v, t in series:
        plot_vs_n(results, v, t, os.path.join(out_dir, f"breakdown_vs_n_{v}_t{t}.png"))
    for n in singles:
        for v, t in series:
            r = results.get(key(v, t, n))
            if r and "error" not in r:
                plot_single(r, v, t, os.path.join(out_dir, f"breakdown_n{n}_{v}_t{t}.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=list(EXES))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--vs-n", action="store_true", help="sweep the default particle counts for this variant")
    ap.add_argument("--counts", help="comma-separated particle counts for --vs-n (overrides the defaults)")
    ap.add_argument("--count", type=int, help="single particle count for the bar chart")
    ap.add_argument("--cap-ms", type=float, default=DEFAULT_CAP_MS)
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--out-dir", default=DATA)
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()

    doc = load_json(args.json) or {"method": "hydrostatic scene via validate.exe --breakdown, 10-step window mean +/- stdev", "results": {}, "singles": []}
    doc.setdefault("singles", [])

    if not args.replot:
        if not args.variant:
            raise SystemExit("--variant is required unless --replot")
        if args.vs_n:
            counts = [int(c) for c in args.counts.split(",")] if args.counts else VS_N_COUNTS[args.variant]
            measure(doc, args.json, args.variant, args.threads, counts, args.cap_ms)
        if args.count:
            measure(doc, args.json, args.variant, args.threads, [args.count], float("inf"))
            if args.count not in doc["singles"]:
                doc["singles"].append(args.count)
                save_json(args.json, doc)

    make_plots(doc, args.out_dir, doc["singles"])
    print(f"Wrote plots to {args.out_dir}")


if __name__ == "__main__":
    main()
