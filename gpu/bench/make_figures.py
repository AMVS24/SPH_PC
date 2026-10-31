"""
Extra figures for the GPU port:
  1. throughput_vs_particles.png   - particle count vs 1/step time (steps per second), all variants
  2. breakdown_10M.png             - A-F stage breakdown (search vs computation) at 10M particles
  3. gpu_time_complexity.png       - GPU ms/step vs N (linear fit + log-log exponent)
  4. gpu_space_complexity.png      - GPU device memory vs N (bytes/particle + fixed cost)
  5. transfer_overhead.png         - CPU<->GPU copy cost vs GPU step time

Measurements that don't come from the main sweep are collected into data/extra_results.json (resumable).
Timing measurements must only be collected on AC power in a performance power mode.

Usage:
    python gpu/bench/make_figures.py collect-mem      # device memory (power-independent)
    python gpu/bench/make_figures.py collect-transfer         # copy timings + PCIe bandwidth (timing!)
    python gpu/bench/make_figures.py collect-breakdown-gpu    # 10M A-F breakdown, GPU (timing!)
    python gpu/bench/make_figures.py collect-breakdown-hash   # 10M A-F breakdown, CPU hash (timing!)
    python gpu/bench/make_figures.py plot
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

import run_benchmarks as rb

EXTRA = os.path.join(rb.DATA, "extra_results.json")
PCIE_EXE = os.path.join(rb.HERE, "build", "pcie_bench.exe")

BD2_N = 10_000_000
MEM_COUNTS = [1000, 10000, 100000] + [m*1000000 for m in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32)]
TRANSFER_COUNTS = [100000, 1000000, 10000000]
DEVICE_POS_BYTES = {"gpu_fp32": 12, "gpu_fp64": 24}
DISK_MBPS = 250.0
VRAM_BYTES = 12227 * 1024 * 1024

STAGE_NAMES = ["A", "B", "C", "D", "E", "F"]
STAGE_TICKS = ["A) Density\nNBHD\nsearch", "B) Density\ncomp.", "C) Force\nNBHD\nsearch",
               "D) Force\ncomp.", "E) Pressure\ncomp.", "F) Drift,\nboundary\n& kick"]
BAR_COLOURS = ["#6B7FB5", "#E39A5C", "#7BB07F", "#C8605F", "#9C8CC4", "#A58A74"]


def load_extra():
    return rb.load_json(EXTRA) or {"mem": {}, "breakdown2": {}, "transfer": {}, "pcie": []}


def run(cmd, timeout=3600):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.stdout, p.returncode, p.stderr


def collect_mem(extra):
    for v in ("gpu_fp32", "gpu_fp64"):
        for n in MEM_COUNTS:
            k = f"{v}|{n}"
            if k in extra["mem"]:
                continue
            out, rc, err = run([rb.exe_for(v), "--variant", v, "--n", str(n), "--mem"])
            for line in out.splitlines():
                if line.startswith("MEM"):
                    extra["mem"][k] = int(rb.parse_fields(line)["device_bytes"])
            rb.save_json(EXTRA, extra)
            print(f"mem {v:>9} n={n:>9}: {extra['mem'].get(k, 'FAILED')}", flush=True)


def collect_breakdown(extra, only):
    for v, t in (("gpu_fp32", 1), ("gpu_fp64", 1), ("hash", 24)):
        if v not in only:
            continue
        k = f"{v}|{t}|{BD2_N}"
        if k in extra["breakdown2"]:
            continue
        steps = 10 if v == "hash" else 20
        out, rc, err = run([rb.exe_for(v), "--variant", v, "--n", str(BD2_N), "--threads", str(t), "--steps", str(steps), "--breakdown2"])
        for line in out.splitlines():
            if line.startswith("STAGES2"):
                f = rb.parse_fields(line)
                extra["breakdown2"][k] = {s: {"ms": float(f[f"{s}_mean"]), "stdev": float(f[f"{s}_stdev"])} for s in STAGE_NAMES}
        rb.save_json(EXTRA, extra)
        print(f"breakdown2 {k}: {'ok' if k in extra['breakdown2'] else 'FAILED ' + err[-300:]}", flush=True)


def collect_transfer(extra):
    for v in ("gpu_fp32", "gpu_fp64"):
        for n in TRANSFER_COUNTS:
            k = f"{v}|{n}"
            if k in extra["transfer"]:
                continue
            out, rc, err = run([rb.exe_for(v), "--variant", v, "--n", str(n), "--transfer"])
            for line in out.splitlines():
                if line.startswith("TRANSFER"):
                    f = rb.parse_fields(line)
                    extra["transfer"][k] = {x: float(f[x]) for x in ("first_step_ms", "step_ms", "fetch_ms")}
            rb.save_json(EXTRA, extra)
            print(f"transfer {k}: {extra['transfer'].get(k, 'FAILED')}", flush=True)

    if not extra["pcie"] and os.path.exists(PCIE_EXE):
        out, rc, err = run([PCIE_EXE])
        for line in out.splitlines():
            if line.startswith("PCIE"):
                extra["pcie"].append({k: float(v) for k, v in rb.parse_fields(line).items()})
        rb.save_json(EXTRA, extra)
        print("pcie:", extra["pcie"], flush=True)


def axes_style(ax):
    rb.style(ax)


def fig_throughput(doc, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = doc["results"]
    series = [("sequential", 1, "Sequential, all-pairs (1 thread)", "#1f1f1f"),
              ("prehash", 24, "OpenMP, all-pairs (24 threads)", "#4C72B0"),
              ("hash", 24, "OpenMP + spatial hash (24 threads)", "#55A868"),
              ("gpu_fp64", 1, "GPU (CUDA), FP64", "#7B3294"),
              ("gpu_fp32", 1, "GPU (CUDA), FP32", "#C44E52")]
    fig, ax = plt.subplots(figsize=(9.5, 6), dpi=140)
    for v, t, lab, c in series:
        x, y, _ = rb.pts(res, v, t)
        if len(x):
            ax.plot(x, 1000.0 / y, color=c, linewidth=2.2, marker="o", markersize=4, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    rb.finish(fig, ax, out, xlabel="Particle count", ylabel="Timesteps per second  (1 / step time)", legend_size=10)
    plt.close(fig)


def fig_breakdown(extra, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("hash|24", f"OpenMP + spatial hash, 24 threads, N = {BD2_N:,}"),
              ("gpu_fp64|1", f"GPU (CUDA) FP64, N = {BD2_N:,}"),
              ("gpu_fp32|1", f"GPU (CUDA) FP32, N = {BD2_N:,}")]
    have = [(k, t) for k, t in panels if f"{k}|{BD2_N}" in extra["breakdown2"]]
    if not have:
        print("no breakdown2 data yet; skipping breakdown_10M.png")
        return False
    fig, axes = plt.subplots(1, len(have), figsize=(7.2*len(have), 5.0), dpi=140)
    axes = np.atleast_1d(axes)
    for ax, (k, title) in zip(axes, have):
        d = extra["breakdown2"][f"{k}|{BD2_N}"]
        vals = [d[s]["ms"] for s in STAGE_NAMES]
        bars = ax.bar(range(6), vals, color=BAR_COLOURS, width=0.75)
        top = max(vals)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + top*0.012, f"{v:,.2f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylim(0, top*1.12)
        ax.set_xticks(range(6))
        ax.set_xticklabels(STAGE_TICKS, fontsize=8)
        ax.set_ylabel("Compute time per timestep (ms)", fontsize=11)
        ax.set_title(title, fontsize=11, loc="left")
        ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        axes_style(ax)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax.text(0.99, 0.97, f"total {sum(vals):,.1f} ms", transform=ax.transAxes, ha="right", va="top", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def fit_linear(x, y):
    a, b = np.polyfit(x, y, 1)
    pred = a*x + b
    r2 = 1 - np.sum((y - pred)**2)/np.sum((y - y.mean())**2)
    return a, b, r2


LABEL_GPU = {"gpu_fp32": "GPU (CUDA), FP32", "gpu_fp64": "GPU (CUDA), FP64"}
GPU_COL = {"gpu_fp32": "#C44E52", "gpu_fp64": "#7B3294"}


def two_panel(plot_fn, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), dpi=140)
    summary = {}
    for ax, v in zip(axes, ("gpu_fp32", "gpu_fp64")):
        summary[v] = plot_fn(ax, v)
        axes_style(ax)
        ax.legend(frameon=False, fontsize=10, loc="upper left")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return summary


def fig_time_complexity(doc, out, nmin=200000):
    res = doc["results"]

    def plot(ax, v):
        x, y, e = rb.pts(res, v, 1)
        m = x >= nmin
        a, b, r2 = fit_linear(x[m], y[m])
        ax.errorbar(x/1e6, y, yerr=e, fmt="o", color=GPU_COL[v], markersize=5, capsize=2, label="measured")
        xs = np.linspace(0, x.max(), 100)
        ax.plot(xs/1e6, a*xs + b, color="black", linestyle="--", linewidth=1.2,
                label=f"linear fit (N >= 200k): {a*1e6:.1f} ns/particle + {b:.2f} ms,  R$^2$ = {r2:.5f}")
        ax.set_title(LABEL_GPU[v], fontsize=12, loc="left")
        ax.set_xlabel("Particle count (millions)", fontsize=12)
        ax.set_ylabel("ms per physics step", fontsize=12)
        return dict(ns_per_particle=a*1e6, r2=r2, exponent=float("nan"), fixed_ms=b)

    return two_panel(plot, out)


def fig_space_complexity(extra, out):
    mem = extra["mem"]

    def plot(ax, v):
        pts_ = sorted((int(k.split("|")[1]), b) for k, b in mem.items() if k.startswith(v + "|"))
        x = np.array([p[0] for p in pts_], float)
        y = np.array([p[1] for p in pts_], float)/1e6
        a, b, r2 = fit_linear(x, y)
        ax.plot(x/1e6, y, "o", color=GPU_COL[v], markersize=5, label="measured (bytes allocated on device)")
        xs = np.linspace(0, x.max(), 100)
        ax.plot(xs/1e6, a*xs + b, color="black", linestyle="--", linewidth=1.2,
                label=f"linear fit: {a*1e6:.1f} B/particle + {b:.1f} MB,  R$^2$ = {r2:.6f}")
        ax.set_title(LABEL_GPU[v], fontsize=12, loc="left")
        ax.set_xlabel("Particle count (millions)", fontsize=12)
        ax.set_ylabel("Device memory allocated (MB)", fontsize=12)
        return dict(bytes_per_particle=a*1e6, fixed_mb=b, r2=r2, n_max_12gb=(VRAM_BYTES/1e6 - b)/a)

    return two_panel(plot, out)


def fig_transfer(extra, out, doc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not extra["pcie"]:
        print("no pcie data yet; skipping transfer_overhead.png")
        return None
    pc = sorted(extra["pcie"], key=lambda r: r["bytes"])

    def gbs(nbytes, key):
        return min(pc, key=lambda r: abs(np.log(r["bytes"]/nbytes)))[key]

    ns = sorted(int(k.split("|")[1]) for k in extra["transfer"] if k.startswith("gpu_fp32|"))
    rows = []
    for n in ns:
        sw = doc["results"][f"gpu_fp32|1|{n}"]
        up_bytes = 2*n*12
        dn_bytes = n*12
        up_ms = up_bytes/(gbs(up_bytes, "h2d_pageable")*1e9)*1e3
        dn_ms = dn_bytes/(gbs(dn_bytes, "d2h_pageable")*1e9)*1e3
        rows.append((n, sw["ms_per_step"], up_ms, dn_ms))
    labels = ["GPU step (compute only, every step)", "Upload positions + velocities (host to device, once at start)", "Download positions (device to host, only when read back)"]
    cols = ["#C44E52", "#4C72B0", "#55A868"]
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=140)
    w = 0.26
    for j in range(3):
        vals = [r[1 + j] for r in rows]
        bars = ax.bar(np.arange(len(rows)) + (j - 1)*w, vals, w, color=cols[j], label=labels[j])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v*1.08, f"{v:,.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{r[0]:,}" for r in rows])
    ax.set_xlabel("Particle count (GPU FP32)", fontsize=12)
    ax.set_ylabel("Time (ms, log scale)", fontsize=12)
    axes_style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["collect-mem", "collect-transfer", "collect-breakdown-gpu", "collect-breakdown-hash", "plot"])
    args = ap.parse_args()
    extra = load_extra()
    if args.cmd == "collect-mem":
        collect_mem(extra)
    elif args.cmd == "collect-transfer":
        collect_transfer(extra)
    elif args.cmd == "collect-breakdown-gpu":
        collect_breakdown(extra, ("gpu_fp32", "gpu_fp64"))
    elif args.cmd == "collect-breakdown-hash":
        collect_breakdown(extra, ("hash",))
    else:
        doc = rb.load_json(rb.DEFAULT_JSON)
        os.makedirs(rb.DATA, exist_ok=True)
        fig_throughput(doc, os.path.join(rb.DATA, "throughput_vs_particles.png"))
        fig_breakdown(extra, os.path.join(rb.DATA, "breakdown_10M.png"))
        ts = fig_time_complexity(doc, os.path.join(rb.DATA, "gpu_time_complexity.png"))
        for v, s in ts.items():
            print(f"time  {v}: {s['ns_per_particle']:.2f} ns/particle, R2={s['r2']:.5f}, log-log exponent={s['exponent']:.3f}, fixed={s['fixed_ms']:.3f} ms")
        ss = fig_space_complexity(extra, os.path.join(rb.DATA, "gpu_space_complexity.png"))
        for v, s in ss.items():
            print(f"space {v}: {s['bytes_per_particle']:.2f} B/particle + {s['fixed_mb']:.2f} MB fixed, R2={s['r2']:.6f}, N_max(12 GB) ~ {s['n_max_12gb']/1e6:.0f}M")
        rows = fig_transfer(extra, os.path.join(rb.DATA, "transfer_overhead.png"), doc)
        if rows:
            print("transfer (fp32): N, step ms, upload once ms, download ms")
            for r in rows:
                print("  %9d %9.2f %9.2f %9.2f" % r)
    rb.save_json(EXTRA, extra)


if __name__ == "__main__":
    main()
