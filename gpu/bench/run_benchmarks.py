"""
3D benchmark sweep: sequential all-pairs vs OpenMP all-pairs vs OpenMP + spatial hash vs GPU (CUDA, FP64 and FP32).

All variants run the same physics on the same scene (a cubic lattice at rest density, see
bench/bench_main.cpp). ms/step is the mean +/- stdev of 10-step window averages after a 2-step
warm-up. Steps per run adapt to cost (as in parallel_c/scaling/bench_common.py), and a CPU series
stops growing N once a step costs more than --cap-ms. Results are written to data/bench_results.json
after every run, so an interrupted sweep resumes where it left off.

The two GPU series come from the same cudaPhysics.cu: gpu_fp32 uses bench/build (float, the default
build) and gpu_fp64 uses bench/fp64/build (compiled with -DGPU_DOUBLE, see bench/build_fp64.bat).

Usage:
    python gpu/bench/run_benchmarks.py
    python gpu/bench/run_benchmarks.py --variants gpu_fp32,gpu_fp64 --no-breakdown
    python gpu/bench/run_benchmarks.py --replot
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
EXE = os.path.join(HERE, "build", "bench.exe")
EXES = {"gpu_fp64": os.path.join(HERE, "fp64", "build", "bench.exe")}
DEFAULT_JSON = os.path.join(DATA, "bench_results.json")

LABELS = {
    "sequential": "Sequential, all-pairs",
    "prehash": "OpenMP, all-pairs",
    "hash": "OpenMP + spatial hash",
    "gpu_fp64": "GPU (CUDA), FP64",
    "gpu_fp32": "GPU (CUDA), FP32",
}
ORDER = {"sequential": 2, "prehash": 2, "hash": 1, "gpu_fp64": 1, "gpu_fp32": 1}
SORT = {"sequential": 0, "prehash": 1, "hash": 2, "gpu_fp64": 3, "gpu_fp32": 4}
GPU_COLOURS = {"gpu_fp64": "#7B3294", "gpu_fp32": "#C44E52"}

ALLPAIRS_COUNTS = [1000, 2000, 3000, 4000, 6000, 8000, 10000, 15000, 20000, 30000, 40000, 60000, 80000, 100000]
FAST_COUNTS = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 4000000, 8000000]
COUNTS = {"sequential": ALLPAIRS_COUNTS, "prehash": ALLPAIRS_COUNTS, "hash": FAST_COUNTS,
          "gpu_fp64": FAST_COUNTS, "gpu_fp32": FAST_COUNTS}
DEFAULT_THREADS = [1, 2, 4, 8, 16, 24]
BREAKDOWN_COUNTS = [10000, 30000, 100000, 300000, 1000000]
STAGES = ["drift_boundary", "grid_build", "density", "pressure", "forces", "kick"]
STAGE_LABELS = ["drift + boundary", "grid build", "density", "pressure", "forces", "kick"]

WINDOW = 10
MIN_STEPS = 2 * WINDOW
MAX_STEPS = 150
RUN_BUDGET_MS = 20000.0
SLOW_STEP_MS = 5000.0
DEFAULT_CAP_MS = 5000.0

COLOURS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]


def is_gpu(v):
    return v.startswith("gpu")


def exe_for(v):
    return EXES.get(v, EXE)


def steps_for(est_ms):
    if est_ms is None or est_ms <= 0:
        return MAX_STEPS
    if est_ms > SLOW_STEP_MS:
        return WINDOW
    s = int(RUN_BUDGET_MS / est_ms)
    s -= s % WINDOW
    return max(MIN_STEPS, min(MAX_STEPS, s))


def parse_fields(line):
    out = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def run_bench(variant, n, threads, steps, breakdown=False):
    cmd = [exe_for(variant), "--variant", variant, "--n", str(n), "--threads", str(threads), "--steps", str(steps)]
    if breakdown:
        cmd.append("--breakdown")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if proc.returncode != 0:
        return {"error": (proc.stderr[-1500:] or f"exit code {proc.returncode}")}
    tag = "STAGES" if breakdown else "RESULT"
    for line in proc.stdout.splitlines():
        if line.startswith(tag):
            f = parse_fields(line)
            if breakdown:
                return {"stages": {s: {"ms": float(f[f"{s}_mean"]), "stdev": float(f[f"{s}_stdev"])} for s in STAGES}}
            return {"steps": int(f["steps"]), "ms_per_step": float(f["mean_ms"]), "ms_per_step_stdev": float(f["stdev_ms"])}
    return {"error": "no result line in output"}


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_json(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2)
    os.replace(tmp, path)


def key(v, t, n):
    return f"{v}|{t}|{n}"


def series_list(variants, thread_counts):
    out = []
    for v in variants:
        if v == "sequential" or is_gpu(v):
            out.append((v, 1))
        else:
            out += [(v, t) for t in thread_counts]
    return out


def sweep(variants, thread_counts, cap_ms, json_path, counts_override):
    for v in variants:
        if not os.path.exists(exe_for(v)):
            raise SystemExit(f"binary not found: {exe_for(v)}  (run: mingw32-make bench, and bench/build_fp64.bat for gpu_fp64)")
    doc = load_json(json_path) or {"method": "3D lattice at rest density, 10-step window mean +/- stdev, 2-step warm-up", "results": {}, "breakdown": {}}
    results = doc["results"]
    series = series_list(variants, thread_counts)
    counts = {v: (counts_override or COUNTS[v]) for v in variants}
    active = {s: True for s in series}
    last = {}
    for s in series:
        prev = sorted((r for r in results.values() if (r["variant"], r["threads"]) == s and "error" not in r), key=lambda r: r["n"])
        if prev:
            last[s] = (prev[-1]["n"], prev[-1]["ms_per_step"])
            if prev[-1]["ms_per_step"] > cap_ms and not is_gpu(s[0]):
                active[s] = False

    all_counts = sorted(set(n for v in variants for n in counts[v]))
    print(f"{'variant':>10} {'thr':>4} {'N':>9} {'steps':>6} {'ms/step':>11} {'+/-stdev':>9}", flush=True)
    print("-" * 56, flush=True)
    for n in all_counts:
        for s in series:
            v, t = s
            if not (active[s] and n in counts[v] and key(v, t, n) not in results):
                continue
            est = None
            if s in last:
                n0, ms0 = last[s]
                est = ms0 * (n / n0) ** ORDER[v]
            r = run_bench(v, n, t, steps_for(est))
            r.update({"variant": v, "threads": t, "n": n})
            results[key(v, t, n)] = r
            doc["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            save_json(json_path, doc)
            if "error" in r:
                print(f"{v:>10} {t:>4} {n:>9}   FAILED: {r['error'].strip().splitlines()[-1]}", flush=True)
                active[s] = False
                continue
            last[s] = (n, r["ms_per_step"])
            print(f"{v:>10} {t:>4} {n:>9} {r['steps']:>6} {r['ms_per_step']:>11.4f} {r['ms_per_step_stdev']:>9.4f}", flush=True)
            if r["ms_per_step"] > cap_ms and not is_gpu(v):
                active[s] = False
    return doc


def breakdown_sweep(json_path, cpu_threads, gpu_variants):
    doc = load_json(json_path)
    bd = doc.setdefault("breakdown", {})
    print("\nstage breakdown", flush=True)
    for n in BREAKDOWN_COUNTS:
        for v, t in [(g, 1) for g in gpu_variants] + [("hash", cpu_threads)]:
            k = key(v, t, n)
            if k in bd and "error" not in bd[k]:
                continue
            r = run_bench(v, n, t, 20, breakdown=True)
            r.update({"variant": v, "threads": t, "n": n})
            bd[k] = r
            save_json(json_path, doc)
            if "error" in r:
                print(f"{v:>9} n={n}: FAILED", flush=True)
            else:
                total = sum(x["ms"] for x in r["stages"].values())
                print(f"{v:>9} t={t:<2} n={n:>8}: total {total:9.3f} ms  " + "  ".join(f"{s}={r['stages'][s]['ms']:.3f}" for s in STAGES), flush=True)
    return doc


def pts(results, v, t):
    import numpy as np
    p = sorted((r for r in results.values() if r["variant"] == v and r["threads"] == t and "error" not in r), key=lambda r: r["n"])
    return (np.array([r["n"] for r in p], float), np.array([r["ms_per_step"] for r in p], float),
            np.array([r["ms_per_step_stdev"] for r in p], float))


def label(v, t):
    if v == "sequential" or is_gpu(v):
        return LABELS[v]
    return f"{LABELS[v]}, {t} thread{'s' if t > 1 else ''}"


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    ax.tick_params(axis="both", labelsize=10, width=1.2)
    ax.grid(False)


def finish(fig, ax, path, xlabel="Particle count", ylabel="ms per physics step", legend_size=9):
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    style(ax)
    ax.legend(frameon=False, fontsize=legend_size)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")


def make_plots(doc, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    results = doc["results"]
    threads = sorted({r["threads"] for r in results.values() if r["variant"] in ("prehash", "hash")})
    tcol = {t: COLOURS[i % len(COLOURS)] for i, t in enumerate(threads)}

    def colour(v, t):
        if v == "sequential":
            return "#1f1f1f"
        if is_gpu(v):
            return GPU_COLOURS[v]
        return tcol[t]

    fig, ax = plt.subplots(figsize=(9.5, 6), dpi=140)
    series = sorted({(r["variant"], r["threads"]) for r in results.values()}, key=lambda s: (SORT[s[0]], s[1]))
    for v, t in series:
        x, y, _ = pts(results, v, t)
        if len(x):
            ax.plot(x, y, color=colour(v, t), linewidth=3.0 if is_gpu(v) else 1.8, linestyle="--" if v == "prehash" else "-", alpha=0.9, label=label(v, t))
    ax.set_xscale("log")
    ax.set_yscale("log")
    finish(fig, ax, os.path.join(out_dir, "scaling_loglog.png"), legend_size=7)
    plt.close(fig)

    def comparison(selection, name, loglog):
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
        for v, t in selection:
            x, y, e = pts(results, v, t)
            if len(x):
                c = "#1f1f1f" if v == "sequential" else GPU_COLOURS[v] if is_gpu(v) else {"prehash": "#4C72B0", "hash": "#55A868"}.get(v, "#8C8C8C")
                if v == "hash" and t == 1:
                    c = "#8DB596"
                ax.plot(x, y, color=c, linewidth=2.0, alpha=0.9)
                ax.scatter(x, y, color=c, s=14, label=label(v, t))
        if loglog:
            ax.set_xscale("log")
            ax.set_yscale("log")
        finish(fig, ax, os.path.join(out_dir, name), ylabel="Compute time per timestep (ms)", legend_size=10)
        plt.close(fig)

    ref_t = 16 if 16 in threads else (threads[-1] if threads else 1)
    sel = [("sequential", 1), ("prehash", ref_t), ("hash", ref_t), ("gpu_fp64", 1), ("gpu_fp32", 1)]
    comparison(sel, "comparison_all.png", False)
    comparison(sel, "comparison_all_loglog.png", True)
    comparison([("hash", 1), ("hash", ref_t), ("gpu_fp64", 1), ("gpu_fp32", 1)], "comparison_hash_vs_gpu_loglog.png", True)

    for variants, order, name in (({"sequential", "prehash"}, 2, "scaling_all_pairs.png"), ({"hash", "gpu_fp64", "gpu_fp32"}, 1, "scaling_spatial_hash.png")):
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
        for v, t in series:
            if v not in variants:
                continue
            x, y, e = pts(results, v, t)
            if not len(x):
                continue
            c = colour(v, t)
            ax.plot(x, y, color=c, linewidth=2.0, alpha=0.85)
            ax.fill_between(x, y - e, y + e, color=c, alpha=0.15, linewidth=0)
            ax.scatter(x, y, color=c, s=12, label=label(v, t))
            if len(x) >= 3:
                k = float(np.sum(y * x ** order) / np.sum(x ** (2 * order)))
                fine = np.linspace(0, x.max(), 200)
                ax.plot(fine, k * fine ** order, color=c, linewidth=1.1, linestyle="--", alpha=0.4)
        finish(fig, ax, os.path.join(out_dir, name))
        plt.close(fig)

    ns = sorted({r["n"] for r in results.values() if r["variant"] == "hash" and r["threads"] == 1})
    for target in (20000, 100000):
        if target not in ns:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 5), dpi=140)
        for v, c in (("hash", "#55A868"), ("prehash", "#4C72B0")):
            xs, ys = [], []
            base = results.get(key(v, 1, target))
            if not base or "error" in base:
                continue
            for t in threads:
                r = results.get(key(v, t, target))
                if r and "error" not in r:
                    xs.append(t)
                    ys.append(base["ms_per_step"] / r["ms_per_step"])
            ax.plot(xs, ys, color=c, marker="o", linewidth=2, label=LABELS[v])
        if threads:
            ax.plot([1, max(threads)], [1, max(threads)], color="#aaaaaa", linestyle=":", label="ideal")
        finish(fig, ax, os.path.join(out_dir, f"speedup_vs_threads_n{target}.png"), xlabel="OpenMP threads", ylabel="Speedup over 1 thread")
        plt.close(fig)

    bd = doc.get("breakdown", {})
    panels = [("hash", None, "OpenMP + spatial hash"), ("gpu_fp64", 1, "GPU (CUDA), FP64"), ("gpu_fp32", 1, "GPU (CUDA), FP32")]
    if any("error" not in r for r in bd.values()):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=140)
        for ax, (v, t, ttl) in zip(axes, panels):
            rows = [r for r in bd.values() if r["variant"] == v and "error" not in r and (t is None or r["threads"] == t)]
            rows.sort(key=lambda r: r["n"])
            if not rows:
                continue
            bottoms = np.zeros(len(rows))
            xs = np.arange(len(rows))
            for si, sname in enumerate(STAGES):
                vals = np.array([r["stages"][sname]["ms"] for r in rows])
                ax.bar(xs, vals, bottom=bottoms, color=COLOURS[si], label=STAGE_LABELS[si])
                bottoms += vals
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{r['n']:,}" for r in rows], fontsize=9)
            ax.set_title(ttl + (f", {rows[0]['threads']} threads" if v == "hash" else ""), fontsize=12)
            ax.set_xlabel("Particle count", fontsize=11)
            ax.set_ylabel("ms per step", fontsize=11)
            style(ax)
            ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "stage_breakdown.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def print_table(doc):
    results = doc["results"]
    threads = sorted({r["threads"] for r in results.values() if r["variant"] == "hash"})
    ref = 24 if 24 in threads else (threads[-1] if threads else 1)
    ns = sorted({r["n"] for r in results.values() if is_gpu(r["variant"])})

    def ms(v, t, n):
        r = results.get(key(v, t, n))
        return r["ms_per_step"] if r and "error" not in r else None

    f = lambda x: f"{x:>11.3f}" if x is not None else f"{'-':>11}"
    print(f"\n{'N':>9} {'seq':>11} {'omp all-pairs':>14} {'omp hash':>11} {'gpu fp64':>10} {'gpu fp32':>10} {'fp32/hash':>10} {'fp32/fp64':>10}   (omp rows use {ref} threads; ms/step)")
    for n in ns:
        g64, g32, h, p, s = ms("gpu_fp64", 1, n), ms("gpu_fp32", 1, n), ms("hash", ref, n), ms("prehash", ref, n), ms("sequential", 1, n)
        print(f"{n:>9} {f(s)} {f(p):>14} {f(h)} {f(g64):>10} {f(g32):>10} " +
              (f"{h/g32:>9.1f}x" if h and g32 else f"{'-':>10}") + (f"{g64/g32:>9.1f}x" if g64 and g32 else f"{'-':>10}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="sequential,prehash,hash,gpu_fp64,gpu_fp32")
    ap.add_argument("--threads", help="comma-separated OpenMP thread counts (default 1,2,4,8,16,24)")
    ap.add_argument("--counts", help="comma-separated particle counts (overrides the per-variant defaults)")
    ap.add_argument("--cap-ms", type=float, default=DEFAULT_CAP_MS, help="stop growing a CPU series once a step costs more than this")
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--out-dir", default=DATA)
    ap.add_argument("--no-breakdown", action="store_true")
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()

    if args.replot:
        doc = load_json(args.json)
    else:
        variants = args.variants.split(",")
        thread_counts = [int(t) for t in args.threads.split(",")] if args.threads else DEFAULT_THREADS
        counts = [int(c) for c in args.counts.split(",")] if args.counts else None
        doc = sweep(variants, thread_counts, args.cap_ms, args.json, counts)
        if not args.no_breakdown:
            gpu_variants = [v for v in variants if is_gpu(v)]
            doc = breakdown_sweep(args.json, 24 if 24 in thread_counts else thread_counts[-1], gpu_variants)
    os.makedirs(args.out_dir, exist_ok=True)
    make_plots(doc, args.out_dir)
    print_table(doc)
    print(f"\nWrote results and plots to {args.out_dir}")


if __name__ == "__main__":
    main()
