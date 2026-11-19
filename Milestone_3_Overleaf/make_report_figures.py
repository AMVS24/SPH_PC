"""
Generates the diagrams and the report-specific scaling plot for Milestone 3.

    python Milestone_3_Overleaf/make_report_figures.py

Writes PDFs into Milestone_3_Overleaf/figures/. The measured-data plots (throughput, stage breakdown,
complexity, transfer) are produced by gpu/bench/make_figures.py and copied alongside.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "gpu", "bench"))

C_ELEM = "#9DB7DD"
C_GRID = "#F0BC8A"
C_NBR = "#E8A09E"
C_HOST = "#D9D9D9"
C_DEV = "#F4F7FB"
EDGE = "#333333"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def box(ax, x, y, w, h, text, fc, fs=8.5, bold=False, ec=EDGE, lw=1.0, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=lw, ls=ls)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")
    return p


def arrow(ax, x0, y0, x1, y1, text=None, ls="-", color=EDGE, rad=0.0, fs=7.5, tx=0, ty=0, ha="center"):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, lw=1.1, color=color, ls=ls,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if text:
        ax.text((x0 + x1)/2 + tx, (y0 + y1)/2 + ty, text, ha=ha, va="center", fontsize=fs, color=color)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    box(ax, 0.1, 3.7, 1.7, 1.2, "Host (CPU)\npositions, velocities\n(double)", C_HOST, fs=8.5, bold=False)
    ax.add_patch(FancyBboxPatch((2.55, 0.55), 8.3, 4.55, boxstyle="round,pad=0.02,rounding_size=0.1", fc=C_DEV, ec="#5577AA", lw=1.6, ls="--"))
    ax.text(2.7, 4.88, "Device memory (RTX 5070 Ti): everything below stays resident for the whole run", fontsize=8.5, color="#335588", va="center")
    ax.text(2.7, 4.58, "pos, vel, acc, density, pressure, cell_id, sorted_keys, cell_particles, cell_start, sort workspace", fontsize=7.5, color="#335588", va="center")

    arrow(ax, 1.8, 4.3, 2.55, 4.3, None, color="#1F4E8C")
    ax.text(2.18, 4.62, "upload\nonce", ha="center", fontsize=7.5, color="#1F4E8C", va="bottom")

    w, h = 1.52, 0.78
    ys1, ys2 = 3.45, 1.75
    xs = [2.75 + i*1.62 for i in range(5)]
    row1 = [("1  drift", C_ELEM), ("2  boundary", C_ELEM), ("3  cell id", C_ELEM), ("4  radix sort\n(CUB)", C_GRID), ("5  cell start\n(binary search)", C_GRID)]
    for (t, c), x in zip(row1, xs):
        box(ax, x, ys1, w, h, t, c)
    for i in range(4):
        arrow(ax, xs[i] + w, ys1 + h/2, xs[i+1], ys1 + h/2)

    row2 = [("6  density", C_NBR), ("7  pressure", C_ELEM), ("8  forces", C_NBR), ("9  kick", C_ELEM)]
    xs2 = [2.75 + i*1.62 for i in range(4)]
    for (t, c), x in zip(row2, xs2):
        box(ax, x, ys2, w, h, t, c)
    for i in range(3):
        arrow(ax, xs2[i] + w, ys2 + h/2, xs2[i+1], ys2 + h/2)
    ymid = (ys1 + ys2 + h)/2
    ax.plot([xs[4] + w/2, xs[4] + w/2, xs2[0] + w/2], [ys1, ymid, ymid], color=EDGE, lw=1.1)
    arrow(ax, xs2[0] + w/2, ymid, xs2[0] + w/2, ys2 + h, None)

    ax.text(xs2[1] + w + 0.04, ys2 - 0.28, "needs every density\nfinished first", fontsize=7, ha="center", va="top", color="#8A2B2B")
    ax.text(xs2[2] + w + 0.04, ys2 - 0.28, "reads old velocities;\nkick must not overlap", fontsize=7, ha="center", va="top", color="#8A2B2B")

    arrow(ax, xs2[3] + w/2, ys2, xs2[3] + w/2, 0.88, None)
    box(ax, xs2[3] - 0.6, 0.62, 2.7, 0.26, "cudaDeviceSynchronize (once per step)", "#FFFFFF", fs=7.5, ec="#555555")
    ax.text(xs2[3] + w + 1.05, 1.7, "repeated\nevery\ntimestep", fontsize=7.5, color="#555555", ha="center", va="center")

    box(ax, 0.1, 0.62, 1.7, 0.95, "Host (CPU)\npositions back\n(only when saved\nor rendered)", C_HOST, fs=8)
    arrow(ax, 2.55, 1.1, 1.8, 1.1, None, ls=(0, (4, 3)), color="#1F4E8C")
    ax.text(2.18, 1.35, "download\non demand", ha="center", fontsize=7.5, color="#1F4E8C")

    lg = [(C_ELEM, "elementwise, one thread per particle"), (C_GRID, "grid construction"), (C_NBR, "neighbour-loop kernels (hot spots)")]
    for k, (c, t) in enumerate(lg):
        ax.add_patch(Rectangle((2.75 + k*2.9, 0.08), 0.22, 0.22, fc=c, ec=EDGE, lw=0.8))
        ax.text(3.05 + k*2.9, 0.19, t, fontsize=7.5, va="center")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "gpu_pipeline.pdf"))
    plt.close(fig)


def fig_launch():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.05, 1]})
    a1.set_xlim(0, 10)
    a1.set_ylim(0, 6)
    a1.axis("off")
    a1.set_title("Launch geometry (block = 256 threads)", fontsize=10, loc="left")
    a1.text(0.1, 5.35, "particle index i = blockIdx.x * 256 + threadIdx.x", fontsize=8.5)
    nb = 4
    bw = 2.2
    for b in range(nb):
        x = 0.2 + b*(bw + 0.2)
        last = b == nb - 1
        a1.add_patch(Rectangle((x, 3.2), bw, 1.3, fc="#DCE6F5" if not last else "#EFEFEF", ec=EDGE, lw=1.1))
        a1.text(x + bw/2, 4.15, f"block {b}" if not last else "last block", ha="center", fontsize=8, fontweight="bold")
        for t in range(8):
            tx = x + 0.12 + t*(bw - 0.2)/8
            idle = last and t >= 4
            a1.add_patch(Rectangle((tx, 3.35), (bw - 0.2)/8 - 0.03, 0.5, fc="#FFFFFF" if idle else "#9DB7DD", ec="#777777", lw=0.5, ls=":" if idle else "-"))
        a1.text(x + bw/2, 3.0, "256 threads" if not last else "tail: threads with i >= N\nreturn immediately", ha="center", va="top", fontsize=7.2)
    a1.text(5, 1.55, "grid size = ceil(N / 256) blocks;  N = 10,000,000  ->  39,063 blocks", ha="center", fontsize=8.5)
    a1.text(5, 1.0, "no thread writes to another particle's output, so no atomics or locks", ha="center", fontsize=8.5, color="#335588")

    a2.set_xlim(0, 10)
    a2.set_ylim(0, 6)
    a2.axis("off")
    a2.set_title("Work of one thread (density and force kernels)", fontsize=10, loc="left")
    steps = [("read own cell id -> (cx, cy, cz)", C_ELEM),
             ("for each of 27 neighbouring cells\n(per-axis bounds check)", C_GRID),
             ("for k in cell_start[c] .. cell_start[c+1]:\nj = cell_particles[k]", C_GRID),
             ("r^2 = |x_j - x_i|^2;  skip if r^2 >= (2H)^2\n(before the sqrt and divides)", C_NBR),
             ("accumulate in registers\n(density sum / acceleration)", C_NBR),
             ("write result[i] once", C_ELEM)]
    for k, (t, c) in enumerate(steps):
        y = 5.15 - k*0.93
        box(a2, 0.4, y, 9.2, 0.7, t, c, fs=8)
        if k < len(steps) - 1:
            arrow(a2, 5, y, 5, y - 0.23, None)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "gpu_launch.pdf"))
    plt.close(fig)


def fig_gridbuild():
    fig, ax = plt.subplots(figsize=(11, 4.7))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.7)
    ax.axis("off")
    cols = ["#9DB7DD", "#F0BC8A", "#A8D5A2", "#D8A6D8"]
    cell_of = [2, 0, 2, 1, 0, 3, 2]
    keys_sorted = [0, 0, 1, 2, 2, 2, 3]
    idx_sorted = [1, 4, 3, 0, 2, 6, 5]

    def row(y, vals, colour_by, label, sub=None, x0=2.9, cw=0.78):
        ax.text(0.1, y + 0.28, label, fontsize=9, fontweight="bold", va="center")
        if sub:
            ax.text(0.1, y + 0.02, sub, fontsize=7.3, va="center", color="#444444")
        for i, (v, cb) in enumerate(zip(vals, colour_by)):
            ax.add_patch(Rectangle((x0 + i*cw, y), cw - 0.04, 0.56, fc=cols[cb], ec=EDGE, lw=0.9))
            ax.text(x0 + i*cw + cw/2 - 0.02, y + 0.28, str(v), ha="center", va="center", fontsize=9)

    ax.text(2.9, 4.45, "position in array:", fontsize=7.5, color="#555555")
    for i in range(7):
        ax.text(2.9 + i*0.78 + 0.37, 4.2, str(i), ha="center", fontsize=7.5, color="#555555")
    row(3.4, list(range(7)), cell_of, "particle id", "(unsorted)")
    row(2.55, cell_of, cell_of, "cell_id[i]", "1  cellIdKernel (one thread per particle)")
    arrow(ax, 5.6, 3.35, 5.6, 3.15, None)
    row(1.55, keys_sorted, keys_sorted, "sorted keys", "2  CUB radix sort of (cell_id, i) pairs")
    row(0.75, idx_sorted, keys_sorted, "cell_particles", "values carried through the sort")
    ax.annotate("", xy=(5.6, 2.1), xytext=(5.6, 2.5), arrowprops=dict(arrowstyle="-|>", lw=1.1, color=EDGE))

    ax.text(8.75, 3.75, "3  cellStartKernel", fontsize=9, fontweight="bold", va="center")
    ax.text(8.75, 3.45, "one thread per cell c (plus sentinel):", fontsize=7.8, va="center")
    ax.text(8.75, 3.2, "binary search for the first sorted key >= c", fontsize=7.8, va="center")
    starts = [0, 2, 3, 6, 7]
    ax.text(8.75, 2.62, "cell_start", fontsize=9, fontweight="bold", va="center")
    for c, s in enumerate(starts):
        ax.add_patch(Rectangle((8.75 + c*0.4, 2.0), 0.36, 0.45, fc=cols[c] if c < 4 else "#FFFFFF", ec=EDGE, lw=0.9))
        ax.text(8.75 + c*0.4 + 0.18, 2.225, str(s), ha="center", va="center", fontsize=9)
    ax.text(8.75, 1.78, "cell 0..3 and end sentinel", fontsize=7, color="#444444", va="center")
    ax.text(8.75, 1.25, "cell 2 holds cell_particles[3:6]\n= particles 0, 2, 6", fontsize=7.8, va="center")
    ax.text(8.75, 0.55, "Same result as the CPU counting-sort\n(count, prefix-sum, scatter) of Milestone 2;\nthe sort is stable, so the order within\na cell is identical.", fontsize=7.4, va="center", color="#335588")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "gpu_gridbuild.pdf"))
    plt.close(fig)


def fig_scaling():
    import run_benchmarks as rb
    doc = rb.load_json(rb.DEFAULT_JSON)
    res = doc["results"]
    series = [("sequential", 1, "Sequential, all-pairs (1 thread)", "#1f1f1f", "-"),
              ("prehash", 24, "OpenMP, all-pairs (24 threads)", "#4C72B0", "-"),
              ("hash", 24, "OpenMP + spatial hash (24 threads)", "#55A868", "-"),
              ("gpu_fp64", 1, "GPU (CUDA), FP64", "#7B3294", "-"),
              ("gpu_fp32", 1, "GPU (CUDA), FP32", "#C44E52", "-")]
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for v, t, lab, c, ls in series:
        x, y, _ = rb.pts(res, v, t)
        if len(x):
            ax.plot(x, y, color=c, linewidth=2.2, marker="o", markersize=3.5, label=lab, ls=ls)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Particle count", fontsize=11)
    ax.set_ylabel("ms per physics step", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "scaling_ms_loglog.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline()
    fig_launch()
    fig_gridbuild()
    fig_scaling()
    print("wrote", sorted(os.listdir(FIG)))
