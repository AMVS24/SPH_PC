"""
Particle-count scaling sweep: sequential vs OpenMP all-pairs (before spatial hash) vs
OpenMP + spatial hash, across thread counts.

Same scene and measurement as sequential_c/scaling/scaling_test.py (see bench_common.py).
Steps per run adapt to cost (MAX_STEPS for cheap runs, down to MIN_STEPS so no run takes
much more than RUN_BUDGET_MS), and each series stops growing N once a step costs more than
--cap-ms. Results are saved to data/scaling_results.json after every run, so an
interrupted sweep keeps everything measured so far and resumes where it left off.

Usage:
    python parallel_c/scaling/scaling_test.py
    python parallel_c/scaling/scaling_test.py --variants hash --threads 1,24
    python parallel_c/scaling/scaling_test.py --replot
"""

import argparse
import os
import time

from bench_common import (DATA, EXES, LABELS, ORDER, COLOURS, MAX_STEPS, generate_scene, steps_for,
                          run_validate, load_json, save_json, style_axes)

DEFAULT_JSON = os.path.join(DATA, "scaling_results.json")

COUNTS = {
    "sequential": list(range(250, 10001, 250)) + list(range(20000, 100001, 10000)),
    "prehash": list(range(250, 10001, 250)) + list(range(20000, 100001, 10000)),
    "hash": list(range(2000, 100001, 2000)),
}
DEFAULT_THREADS = [1, 2, 4, 8, 16, 24, 32]
DEFAULT_CAP_MS = float("inf")


def key(variant, threads, n):
    return f"{variant}|{threads}|{n}"


def series_list(variants, thread_counts):
    out = []
    for v in variants:
        if v == "sequential":
            out.append((v, 1))
        else:
            out += [(v, t) for t in thread_counts]
    return out


def sweep(variants, thread_counts, cap_ms, json_path, counts_override):
    for v in variants:
        if not os.path.exists(EXES[v]):
            raise SystemExit(f"binary not found: {EXES[v]}")

    doc = load_json(json_path) or {"method": "hydrostatic scene via validate.exe, 10-step window mean +/- stdev", "results": {}}
    results = doc["results"]
    series = series_list(variants, thread_counts)
    counts = {v: (counts_override or COUNTS[v]) for v in variants}
    active = {s: True for s in series}
    last = {}

    for s in series:
        prev = sorted((r for r in results.values() if (r["variant"], r["threads"]) == s and "error" not in r),
                      key=lambda r: r["particles_requested"])
        if prev:
            last[s] = (prev[-1]["particles_requested"], prev[-1]["ms_per_step"])
            if prev[-1]["ms_per_step"] > cap_ms and ORDER[s[0]] == 2:
                active[s] = False

    all_counts = sorted(set(n for v in variants for n in counts[v]))
    print(f"{'variant':>10} {'thr':>4} {'N req':>7} {'n_total':>8} {'steps':>6} {'ms/step':>11} {'+/-stdev':>9}")
    print("-" * 64)
    for n in all_counts:
        todo = [s for s in series if active[s] and n in counts[s[0]] and key(s[0], s[1], n) not in results]
        if not todo:
            continue
        scene = generate_scene(n)
        for s in todo:
            v, t = s
            est = None
            if s in last:
                n0, ms0 = last[s]
                est = ms0 * (n / n0) ** ORDER[v]
            r = run_validate(v, scene, t, steps_for(est))
            r.update({"variant": v, "threads": t, "particles_requested": n})
            results[key(v, t, n)] = r
            doc["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            save_json(json_path, doc)
            if "error" in r:
                print(f"{v:>10} {t:>4} {n:>7}   FAILED: {r['error'].strip().splitlines()[-1]}")
                active[s] = False
                continue
            last[s] = (n, r["ms_per_step"])
            print(f"{v:>10} {t:>4} {n:>7} {r['n_total']:>8} {r['steps']:>6} {r['ms_per_step']:>11.4f} {r['ms_per_step_stdev']:>9.4f}", flush=True)
            if r["ms_per_step"] > cap_ms and ORDER[v] == 2:
                active[s] = False
    return doc


def series_points(results, v, t):
    import numpy as np
    pts = sorted((r for r in results.values() if r["variant"] == v and r["threads"] == t and "error" not in r),
                 key=lambda r: r["n_total"])
    xs = np.array([r["n_total"] for r in pts], dtype=float)
    ys = np.array([r["ms_per_step"] for r in pts], dtype=float)
    es = np.array([r["ms_per_step_stdev"] for r in pts], dtype=float)
    return xs, ys, es


def series_label(v, t):
    return LABELS[v] if v == "sequential" else f"{LABELS[v]}, {t} thread{'s' if t > 1 else ''}"


def plot_linear(results, variants, path, fit_order):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(9, 5.5), dpi=140)
    ax = plt.gca()
    series = sorted({(r["variant"], r["threads"]) for r in results.values() if r["variant"] in variants},
                    key=lambda s: (s[0] != "sequential", s[1]))
    thread_idx = {t: i for i, t in enumerate(sorted({r["threads"] for r in results.values() if r["variant"] != "sequential"}))}
    for v, t in series:
        xs, ys, es = series_points(results, v, t)
        if len(xs) == 0:
            continue
        c = "#1f1f1f" if v == "sequential" else COLOURS[thread_idx[t] % len(COLOURS)]
        ax.plot(xs, ys, color=c, linewidth=2.0, alpha=0.85)
        ax.fill_between(xs, ys - es, ys + es, color=c, alpha=0.15, linewidth=0)
        ax.scatter(xs, ys, color=c, s=12, alpha=0.95, label=series_label(v, t))
        if len(xs) >= 3:
            k = float(np.sum(ys * xs ** fit_order) / np.sum(xs ** (2 * fit_order)))
            fine = np.linspace(0, xs.max(), 200)
            ax.plot(fine, k * fine ** fit_order, color=c, linewidth=1.2, linestyle="--", alpha=0.45)

    ax.set_xlabel("Particle count", fontsize=12)
    ax.set_ylabel("ms per physics step", fontsize=12)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_loglog(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5.5), dpi=140)
    ax = plt.gca()
    series = sorted({(r["variant"], r["threads"]) for r in results.values()},
                    key=lambda s: ({"sequential": 0, "prehash": 1, "hash": 2}[s[0]], s[1]))
    thread_idx = {t: i for i, t in enumerate(sorted({r["threads"] for r in results.values() if r["variant"] != "sequential"}))}
    for v, t in series:
        xs, ys, _ = series_points(results, v, t)
        if len(xs) == 0:
            continue
        c = "#1f1f1f" if v == "sequential" else COLOURS[thread_idx[t] % len(COLOURS)]
        ls = "--" if v == "prehash" else "-"
        ax.plot(xs, ys, color=c, linewidth=1.8, linestyle=ls, alpha=0.85, label=series_label(v, t))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Particle count", fontsize=12)
    ax.set_ylabel("ms per physics step", fontsize=12)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_comparison(results, series, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5.5), dpi=140)
    ax = plt.gca()
    for c, (v, t) in zip([COLOURS[0], COLOURS[1], COLOURS[2]], series):
        xs, ys, _ = series_points(results, v, t)
        if len(xs) == 0:
            continue
        ax.plot(xs, ys, color=c, linewidth=2.0, alpha=0.85)
        ax.scatter(xs, ys, color=c, s=12, alpha=0.95, label=series_label(v, t))

    ax.set_xlabel("Particle count", fontsize=12)
    ax.set_ylabel("Compute time per timestep (ms)", fontsize=12)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def make_plots(doc, out_dir):
    results = doc["results"]
    compare = [("sequential", 1), ("prehash", 16), ("hash", 16)]
    plot_comparison(results, compare, os.path.join(out_dir, "comparison_seq_vs_16t.png"))
    plot_comparison(results, [("prehash", 16), ("hash", 16)], os.path.join(out_dir, "comparison_16t_nohash_vs_hash.png"))
    plot_linear(results, {"sequential", "prehash"}, os.path.join(out_dir, "scaling_all_pairs.png"), 2)
    plot_linear(results, {"hash"}, os.path.join(out_dir, "scaling_spatial_hash.png"), 1)
    plot_loglog(results, os.path.join(out_dir, "scaling_loglog.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="sequential,prehash,hash")
    ap.add_argument("--threads", help="comma-separated thread counts (default 1,2,4,8,16,24,32)")
    ap.add_argument("--counts", help="comma-separated particle counts (overrides the per-variant defaults)")
    ap.add_argument("--cap-ms", type=float, default=DEFAULT_CAP_MS, help="stop growing an O(N^2) series once a step costs more than this")
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--out-dir", default=DATA)
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()

    if args.replot:
        doc = load_json(args.json)
    else:
        variants = args.variants.split(",")
        thread_counts = [int(t) for t in args.threads.split(",")] if args.threads else DEFAULT_THREADS
        counts = [int(c) for c in args.counts.split(",")] if args.counts else None
        doc = sweep(variants, thread_counts, args.cap_ms, args.json, counts)
    make_plots(doc, args.out_dir)
    print(f"Wrote plots to {args.out_dir}")


if __name__ == "__main__":
    main()
