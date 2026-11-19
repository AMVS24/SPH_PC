"""
Speedup figures: ORCS (Meneses et al., arXiv:2601.15633, Table 2, wall BC) and our ORCS-scenario sweep.

    python rtx/frnn/make_speedup_figures.py

Speedup = grid time / RT time (above 1x: RT faster; below 1x: grid faster).
Writes data/orcs_speedup.png and data/our_speedup.png next to this script.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

DISTS = [("lattice", "Lattice"), ("disordered", "Disordered"), ("cluster", "Cluster")]
RADII = [("r1", "r = 1"), ("r160", "r = 160"), ("U", "U[1,160]"), ("LN", "LN(1,2)")]

# ORCS Table 2, wall BC, ms per step: (50k, 1M); None = out of memory or not applicable
ORCS = {
    "lattice": {
        "GPU-CELL":    {"r1": (0.415, 2.97), "r160": (43.6, 3185), "U": (27.1, 2163), "LN": (130, 7767)},
        "RT-REF":      {"r1": (0.165, 0.599), "r160": (17.6, 2235), "U": (6.99, 835), "LN": (10.6, None)},
        "ORCS-forces": {"r1": (0.166, 0.612), "r160": (19.5, 4836), "U": (4.99, 1323), "LN": (2.33, 548)},
        "ORCS-perse":  {"r1": (0.127, 0.506), "r160": (19.6, 5201), "U": (None, None), "LN": (None, None)},
    },
    "disordered": {
        "GPU-CELL":    {"r1": (0.433, 3.49), "r160": (48.6, 4569), "U": (28.4, 2697), "LN": (131, 8073)},
        "RT-REF":      {"r1": (0.163, 0.658), "r160": (18.6, 2861), "U": (7.08, 972), "LN": (8.73, None)},
        "ORCS-forces": {"r1": (0.153, 0.660), "r160": (22.3, 5398), "U": (5.41, 1464), "LN": (2.59, 674)},
        "ORCS-perse":  {"r1": (0.117, 0.552), "r160": (22.5, 6017), "U": (None, None), "LN": (None, None)},
    },
    "cluster": {
        "GPU-CELL":    {"r1": (0.638, 6.82), "r160": (333, 7119), "U": (278, 6114), "LN": (229, 12241)},
        "RT-REF":      {"r1": (0.200, 4.77), "r160": (251, None), "U": (182, None), "LN": (145, None)},
        "ORCS-forces": {"r1": (0.176, 6.32), "r160": (223, 19150), "U": (119, 6796), "LN": (40.8, 2586)},
        "ORCS-perse":  {"r1": (0.146, 7.44), "r160": (232, 23445), "U": (None, None), "LN": (None, None)},
    },
}

OUR_RADII = [("const:1", "r = 1"), ("const:160", "r = 160"), ("uniform:1:160", "U[1,160]"),
             ("lognormal:1:2:1:330", "LN(1,2)"), ("SPH", "SPH r")]

YMIN, YMAX = 1e-3, 1e2


def style_axis(ax):
    ax.set_yscale("log")
    ax.set_ylim(YMIN, YMAX)
    ticks = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.axhline(1.0, color=INK, linewidth=1.2, zorder=3)
    ax.text(1.0, 1.0, "  1x (equal)", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8, color=INK2)
    ax.text(0.003, 0.97, "RT faster ↑", transform=ax.transAxes, va="top", fontsize=8, color=INK2)
    ax.text(0.003, 0.03, "grid faster ↓", transform=ax.transAxes, va="bottom", fontsize=8, color=INK2)


def ratio_bar(ax, x, value, width, colour, hatch=None):
    ax.bar(x, value - 1.0, bottom=1.0, width=width, color=colour, edgecolor=SURFACE, linewidth=1.5,
           hatch=hatch, zorder=2)


def note(ax, x, text):
    ax.text(x, 1.25, text, rotation=90, ha="center", va="bottom", fontsize=6.5, color=INK2, zorder=4)


def group_axis(ax, radii, n_groups_per_dist):
    xs, labels = [], []
    for di, (_, dname) in enumerate(DISTS):
        for ri, (_, rlabel) in enumerate(radii):
            xs.append(di*(n_groups_per_dist + 1) + ri)
            labels.append(rlabel)
        mid = di*(n_groups_per_dist + 1) + (n_groups_per_dist - 1)/2
        ax.text(mid, -0.13, dname, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=10, color=INK, fontweight="bold")
        if di:
            ax.axvline(di*(n_groups_per_dist + 1) - 1, color=GRID, linewidth=1.0, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlim(-0.7, xs[-1] + 0.7)
    return xs


def orcs_figure(path):
    methods = [("RT-REF", "RT-REF (RT fills neighbour list, then force kernel)", BLUE, None),
               ("ORCS-forces", "ORCS-forces (forces inside RT pipeline, no list)", ORANGE, None),
               ("ORCS-perse", "ORCS-perse (whole step inside RT pipeline, constant r only)", AQUA, "///")]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.6), dpi=150, sharex=False)
    fig.patch.set_facecolor(SURFACE)
    w = 0.26
    for pi, (ax, nlabel) in enumerate(zip(axes, ("N = 50,000", "N = 1,000,000"))):
        style_axis(ax)
        xs = group_axis(ax, RADII, len(RADII))
        k = 0
        for dist, _ in DISTS:
            for rkey, _ in RADII:
                cell = ORCS[dist]["GPU-CELL"][rkey][pi]
                for mi, (m, _, col, hatch) in enumerate(methods):
                    t = ORCS[dist][m][rkey][pi]
                    x = xs[k] + (mi - 1)*w
                    if t is None:
                        note(ax, x, "OOM" if m == "RT-REF" else "n/a")
                    else:
                        ratio_bar(ax, x, cell/t, w, col, hatch)
                k += 1
        ax.set_title(nlabel, loc="left", fontsize=10.5, color=INK)
        ax.set_ylabel("speedup over GPU-CELL", fontsize=9, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, hatch=h, ec=SURFACE) for _, _, c, h in methods]
    fig.legend(handles, [lab for _, lab, _, _ in methods], loc="upper center", ncol=1, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("ORCS (Meneses et al. 2026): RT-core methods vs GPU cell list — full Lennard-Jones simulation step, wall BC, RTX Pro 6000 Blackwell",
                 fontsize=11.5, color=INK, y=0.995)
    fig.text(0.5, 0.005, "Speedup = GPU-CELL ms per step / RT-method ms per step (ORCS Table 2). OOM = RT-REF ran out of memory; n/a = ORCS-perse supports constant radius only.",
             ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.885))
    fig.subplots_adjust(hspace=0.42)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def our_figure(path):
    doc = json.load(open(os.path.join(DATA, "frnn_box1000.json")))

    def find(n, dist, spec):
        if spec == "SPH":
            for k, r in doc.items():
                kn, kd, ks = k.split("|", 2)
                if int(kn) == n and kd == dist and ks.startswith("const:") and ks not in ("const:1", "const:160"):
                    return r
            return None
        return doc.get(f"{n}|{dist}|{spec}")

    ns = [(50000, "N = 50,000"), (200000, "N = 200,000"), (1000000, "N = 1,000,000")]
    fig, axes = plt.subplots(3, 1, figsize=(12, 11.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    w = 0.55
    for ax, (n, nlabel) in zip(axes, ns):
        style_axis(ax)
        xs = group_axis(ax, OUR_RADII, len(OUR_RADII))
        k = 0
        for dist, _ in DISTS:
            for spec, _ in OUR_RADII:
                r = find(n, dist, spec)
                if r is not None:
                    ratio_bar(ax, xs[k], r["grid_total_ms"]/r["rt_total_ms"], w, ORANGE)
                k += 1
        ax.set_title(nlabel, loc="left", fontsize=10.5, color=INK)
        ax.set_ylabel("RT speedup over grid", fontsize=9, color=INK2)
    fig.suptitle("Ours: RT-core vs uniform-grid neighbour search on ORCS's scenarios — neighbour search only (frozen particles), box 1000, RTX 5000 Ada",
                 fontsize=11.5, color=INK, y=0.995)
    fig.text(0.5, 0.965, "Each bar: grid time (build cell index + one search pass) ÷ RT time (Morton reorder + BVH rebuild + one trace)",
             ha="center", fontsize=9.5, color=INK2)
    fig.text(0.5, 0.005, "Mean of 3-10 repetitions. Every run checked per particle against the grid's neighbour counts.",
             ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.015, 1, 0.95))
    fig.subplots_adjust(hspace=0.38)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    orcs_figure(os.path.join(DATA, "orcs_speedup.png"))
    our_figure(os.path.join(DATA, "our_speedup.png"))
    print("wrote", os.path.join(DATA, "orcs_speedup.png"), "and", os.path.join(DATA, "our_speedup.png"))
