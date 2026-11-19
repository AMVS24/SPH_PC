"""
Nsight Compute profile of the first (FP64) GPU kernels at 1M particles, from the Milestone 3 report table.

    python gpu/bench/make_ncu_figure.py

Writes gpu/bench/data/ncu_fp64_profile.png.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

METRICS = [
    ("FP64 pipe\nutilisation", 88.3, 88.6),
    ("Memory\nthroughput", 3.4, 3.0),
    ("DRAM\nthroughput", 0.05, 0.08),
    ("Warp efficiency\n(active lanes / 32)", 26.1/32*100, 18.0/32*100),
    ("Theoretical\noccupancy", 83, 50),
    ("Achieved\noccupancy", 76, 40),
]


def main():
    fig, ax = plt.subplots(figsize=(11, 5.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)

    w = 0.36
    for i, (label, dens, force) in enumerate(METRICS):
        for off, val, col in ((-w/2, dens, BLUE), (w/2, force, ORANGE)):
            ax.bar(i + off, val, width=w, color=col, edgecolor=SURFACE, linewidth=1.5)
            txt = f"{val:.2f}%" if val < 1 else f"{val:.0f}%" if val >= 10 else f"{val:.1f}%"
            ax.text(i + off, val + 1.5, txt, ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([m[0] for m in METRICS], fontsize=9, color=INK)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of peak", fontsize=10, color=INK2)
    ax.axvspan(-0.5, 0.5, color="#f3e3da", zorder=0)
    ax.text(0, 100, "the limiter", ha="center", fontsize=9, color=INK, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE)]
    ax.legend(handles, ["density kernel (47 registers/thread)", "force kernel (79 registers/thread)"],
              frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Nsight Compute: first GPU version (FP64), 1M particles — arithmetic-bound, memory nearly idle",
                 fontsize=12.5, color=INK, x=0.02, ha="left", y=0.985)
    fig.text(0.02, 0.915, "FP64 pipe is ~88% busy while memory throughput is ~3% and DRAM <0.1%: the kernels wait on double-precision math, "
             "not data.\nThis GPU runs FP64 at 1/64 of its FP32 rate, which motivated the switch to single precision (~17-20x faster).",
             fontsize=8.5, color=INK2, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    out = os.path.join(HERE, "data", "ncu_fp64_profile.png")
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
