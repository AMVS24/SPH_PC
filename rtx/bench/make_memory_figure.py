"""
Device memory of the grid CUDA solver vs the RT-core solver.

    python rtx/bench/make_memory_figure.py

Grid points: gpu/bench/data/extra_results.json (gpu_fp32). RT points: rtx/bench/data/rtx_memory.json.
Writes rtx/bench/data/memory_grid_vs_rt.png.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GREYS = ["#8c8a85", "#a9a7a1", "#c5c3bd", "#dddbd5"]


def fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs)/n, sum(ys)/n
    a = sum((x - mx)*(y - my) for x, y in zip(xs, ys))/sum((x - mx)**2 for x in xs)
    return a, my - a*mx


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


def main():
    extra = json.load(open(os.path.join(ROOT, "gpu", "bench", "data", "extra_results.json")))
    grid = sorted((int(k.split("|")[1]), v) for k, v in extra["mem"].items() if k.startswith("gpu_fp32|"))
    rt = sorted((int(k), v) for k, v in json.load(open(os.path.join(HERE, "data", "rtx_memory.json")))["points"].items())
    ga, gb = fit([n for n, _ in grid], [b for _, b in grid])
    ra, rb = fit([n for n, _ in rt], [b for _, b in rt])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150, gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor(SURFACE)

    style(a1)
    a1.plot([n/1e6 for n, _ in grid], [b/1e9 for _, b in grid], color=BLUE, marker="o", markersize=6, linewidth=2,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    a1.plot([n/1e6 for n, _ in rt], [b/1e9 for _, b in rt], color=ORANGE, marker="o", markersize=6, linewidth=2,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    for cap, label in ((12.0, "12 GB laptop GPU"), (32.0, "32 GB RTX 5000 Ada")):
        a1.axhline(cap, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        a1.text(43.5, cap + 0.35, label, ha="right", va="bottom", fontsize=8.5, color=INK2)
    a1.text(grid[-1][0]/1e6 + 0.5, grid[-1][1]/1e9, f"grid: {ga:.0f} B/particle", fontsize=9.5, color=INK, va="center")
    a1.text(rt[-1][0]/1e6 + 0.5, rt[-1][1]/1e9, f"RT: {ra:.0f} B/particle", fontsize=9.5, color=INK, va="center")
    a1.set_xlim(0, 44)
    a1.set_ylim(0, 34)
    a1.set_xlabel("particles (millions)", fontsize=10, color=INK2)
    a1.set_ylabel("device memory allocated (GB)", fontsize=10, color=INK2)
    a1.set_title("Memory vs particle count (both linear)", loc="left", fontsize=11, color=INK)
    a1.text(0.02, 0.86, f"Max particles in 12 GB:  grid ~{12e9/ga/1e6:.0f}M,  RT ~{12e9/ra/1e6:.0f}M\n"
            f"Max particles in 32 GB:  grid ~{32e9/ga/1e6:.0f}M,  RT ~{32e9/ra/1e6:.0f}M\n(upper bounds; ignore CUDA context and other GPU users)",
            transform=a1.transAxes, va="top", fontsize=8.5, color=INK2)

    style(a2)
    a2.grid(False)
    rt_parts = [("neighbour list\n(128 slots x 4 B)", 512.0, ORANGE),
                ("BVH + sort workspace", ra - 512 - 44 - 48 - 24 - 4, GREYS[0]),
                ("reorder buffers + ids", 48.0, GREYS[1]),
                ("particle state", 44.0, GREYS[2]),
                ("AABBs + counts", 28.0, GREYS[3])]
    grid_parts = [("particle state", 44.0, GREYS[2]), ("grid index arrays", 20.0, GREYS[1]), ("sort workspace", ga - 64.0, GREYS[0])]
    for y, parts in ((1, rt_parts), (0, grid_parts)):
        left = 0.0
        for label, w, col in parts:
            a2.barh(y, w, left=left, height=0.5, color=col, edgecolor=SURFACE, linewidth=2)
            if w >= 200:
                a2.text(left + w/2, y, f"{label}\n{w:.0f} B", ha="center", va="center", fontsize=8.5, color="#ffffff")
            left += w
        a2.text(left + 10, y, f"{left:.0f} B", va="center", fontsize=10, color=INK, fontweight="bold")
        small = [f"{lab.splitlines()[0]} {w:.0f} B" for lab, w, _ in parts if w < 200]
        a2.text(0, y - 0.36, "grey segments: " + ",  ".join(small), fontsize=8, color=INK2, ha="left", va="center")
    a2.set_yticks([0, 1])
    a2.set_yticklabels(["Grid (CUDA)", "RT cores"], fontsize=10, color=INK)
    a2.set_xlim(0, 860)
    a2.set_ylim(-0.6, 1.6)
    a2.set_xlabel("bytes per particle", fontsize=10, color=INK2)
    a2.set_title("Where the bytes go (per particle)", loc="left", fontsize=11, color=INK)

    fig.suptitle("Device memory: grid CUDA solver vs RT-core solver (FP32)", fontsize=12.5, color=INK, x=0.02, ha="left", y=0.985)
    fig.text(0.02, 0.915, "Measured bytes allocated by each solver. The RT solver's 10x cost is mostly its stored neighbour list; "
             "the grid solver searches on the fly and stores none.", fontsize=8.5, color=INK2, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = os.path.join(HERE, "data", "memory_grid_vs_rt.png")
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}  grid {ga:.2f} B/particle + {gb/1e6:.2f} MB, RT {ra:.2f} B/particle + {rb/1e6:.2f} MB")


if __name__ == "__main__":
    main()
