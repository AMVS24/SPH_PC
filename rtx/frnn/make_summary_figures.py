"""
Summary figures from the frozen-particle neighbour-search benchmark.

    python rtx/frnn/make_summary_figures.py

Writes, next to this script in data/:
  orcs_scenarios_lattice.png, orcs_scenarios_disordered.png, orcs_scenarios_cluster.png
      RT speedup over the uniform grid for r = 1, r = 160 and log-normal radius, geometric mean over
      N = 50k, 200k and 1M (whisker = min..max over the three N).
  multires_poc.png
      Multi-resolution SPH density stage: grid and RT time vs h ratio, and the resulting speedup.
"""

import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

NS = [50000, 200000, 1000000]
RADII = [("const:1", "r = 1\n(tiny radius, sparse box)"),
         ("const:160", "r = 160\n(large constant radius)"),
         ("lognormal:1:2:1:330", "log-normal radius\n(mostly small, a few huge)")]
DISTS = [("lattice", "Lattice"), ("disordered", "Disordered"), ("cluster", "Cluster")]


def base_style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


def log_ticks(ax, ticks, fmt):
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))


def speedup_label(v):
    return f"{v:.1f}x" if v >= 1 else f"{v:.2f}x"


def scenario_figures():
    doc = json.load(open(os.path.join(DATA, "frnn_box1000.json")))
    for dist, dname in DISTS:
        means, lows, highs = [], [], []
        for spec, _ in RADII:
            vals = [doc[f"{n}|{dist}|{spec}"]["grid_total_ms"]/doc[f"{n}|{dist}|{spec}"]["rt_total_ms"] for n in NS]
            means.append(math.exp(sum(math.log(v) for v in vals)/len(vals)))
            lows.append(min(vals))
            highs.append(max(vals))

        fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        base_style(ax)
        ax.set_yscale("log")
        ax.set_ylim(0.03, 100)
        log_ticks(ax, [0.1, 1, 10, 100], lambda v, _: f"{v:g}x")
        ax.axhline(1.0, color=INK, linewidth=1.2, zorder=3)
        xs = list(range(len(RADII)))
        for x, m, lo, hi in zip(xs, means, lows, highs):
            ax.bar(x, m - 1.0, bottom=1.0, width=0.55, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=2)
            ax.plot([x, x], [lo, hi], color=INK2, linewidth=1.2, zorder=4)
            ax.plot([x - 0.06, x + 0.06], [lo, lo], color=INK2, linewidth=1.2, zorder=4)
            ax.plot([x - 0.06, x + 0.06], [hi, hi], color=INK2, linewidth=1.2, zorder=4)
            y = max(m, hi)*1.35 if m >= 1 else min(m, lo)/1.35
            ax.text(x + 0.31, m, speedup_label(m), ha="left", va="center", fontsize=10.5, color=INK, fontweight="bold", zorder=5)
        ax.set_xticks(xs)
        ax.set_xticklabels([lab for _, lab in RADII], fontsize=9, color=INK)
        ax.set_xlim(-0.6, len(RADII) - 0.4)
        ax.set_ylabel("RT speedup over uniform grid", fontsize=10, color=INK2)
        ax.text(0.01, 0.98, "RT faster ↑", transform=ax.transAxes, va="top", fontsize=8.5, color=INK2)
        ax.text(0.01, 0.02, "grid faster ↓", transform=ax.transAxes, va="bottom", fontsize=8.5, color=INK2)
        fig.suptitle(f"{dname} particles: RT-core vs uniform-grid neighbour search", fontsize=12.5, color=INK, x=0.02, ha="left", y=0.985)
        fig.text(0.02, 0.915, "ORCS scenarios, box 1000, RTX 5000 Ada. Bar = geometric mean over N = 50k, 200k, 1M; line = range over those three.",
                 fontsize=8.5, color=INK2, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        out = os.path.join(DATA, f"orcs_scenarios_{dist}.png")
        fig.savefig(out, facecolor=SURFACE)
        plt.close(fig)
        print("wrote", out, " means:", ", ".join(f"{lab.splitlines()[0]}={m:.2f}x" for (_, lab), m in zip(RADII, means)))


def multires_figure():
    rows = []
    pat = re.compile(r"ratio=([\d.]+)\s+density\s+grid=\s*([\d.]+) ms\s+rt=\s*([\d.]+) ms")
    for line in open(os.path.join(DATA, "multires_poc.txt")):
        m = pat.search(line)
        if m:
            rows.append(tuple(float(g) for g in m.groups()))
    ratios = [r[0] for r in rows]
    grid = [r[1] for r in rows]
    rt = [r[2] for r in rows]
    speed = [g/r for g, r in zip(grid, rt)]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.9), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    base_style(a1)
    a1.set_yscale("log")
    a1.set_ylim(1, 1000)
    log_ticks(a1, [1, 10, 100, 1000], lambda v, _: f"{v:g}")
    a1.plot(ratios, grid, color=BLUE, linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    a1.plot(ratios, rt, color=ORANGE, linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    a1.text(ratios[-1] + 0.25, grid[-1], "uniform grid", color=INK, fontsize=9.5, va="center")
    a1.text(ratios[-1] + 0.25, rt[-1], "RT cores", color=INK, fontsize=9.5, va="center")
    for x, g, r in zip(ratios, grid, rt):
        g_up = g > r
        a1.text(x, g*1.35 if g_up else g/1.35, f"{g:.1f}", ha="center", va="bottom" if g_up else "top", fontsize=8, color=INK2)
        a1.text(x, r/1.35 if g_up else r*1.35, f"{r:.1f}", ha="center", va="top" if g_up else "bottom", fontsize=8, color=INK2)
    a1.set_xticks(ratios)
    a1.set_xticklabels([f"{r:g}:1" for r in ratios])
    a1.set_xlim(0.5, ratios[-1] + 1.6)
    a1.set_xlabel("h ratio (coarse ÷ fine particle size)", fontsize=10, color=INK2)
    a1.set_ylabel("ms per SPH density pass (log scale)", fontsize=10, color=INK2)
    a1.set_title("Time per density pass", loc="left", fontsize=11, color=INK)

    base_style(a2)
    a2.set_yscale("log")
    a2.set_ylim(0.1, 100)
    log_ticks(a2, [0.1, 1, 10, 100], lambda v, _: f"{v:g}x")
    a2.axhline(1.0, color=INK, linewidth=1.2, zorder=3)
    xs = list(range(len(ratios)))
    for x, s in zip(xs, speed):
        a2.bar(x, s - 1.0, bottom=1.0, width=0.55, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=2)
        a2.text(x, s*1.3 if s >= 1 else s/1.3, speedup_label(s), ha="center", va="bottom" if s >= 1 else "top",
                fontsize=10.5, color=INK, fontweight="bold")
    a2.set_xticks(xs)
    a2.set_xticklabels([f"{r:g}:1" for r in ratios])
    a2.set_xlabel("h ratio (coarse ÷ fine particle size)", fontsize=10, color=INK2)
    a2.set_ylabel("RT speedup over uniform grid", fontsize=10, color=INK2)
    a2.set_title("RT speedup", loc="left", fontsize=11, color=INK)
    a2.text(0.01, 0.98, "RT faster ↑", transform=a2.transAxes, va="top", fontsize=8.5, color=INK2)
    a2.text(0.01, 0.02, "grid faster ↓", transform=a2.transAxes, va="bottom", fontsize=8.5, color=INK2)

    fig.suptitle("Multi-resolution SPH (proof of concept): grid vs RT-core neighbour search in the density pass",
                 fontsize=12.5, color=INK, x=0.02, ha="left", y=0.985)
    fig.text(0.02, 0.905, "1M particles, RTX 5000 Ada, frozen particles. Coarse particles fill the box; a central fine region uses h smaller by the given ratio;\n"
             "both populations have ~80 neighbours. The uniform grid must size its cells for the coarse particles.\n"
             "Grid and RT densities agree per particle to ~1e-6 (relative). Ratio 1:1 = ordinary single-resolution SPH.",
             fontsize=8.5, color=INK2, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.8))
    out = os.path.join(DATA, "multires_poc.png")
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    scenario_figures()
    multires_figure()
