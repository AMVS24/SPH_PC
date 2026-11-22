import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.edgecolor": INK2, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK})


def style(ax):
    ax.grid(True, axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2, pad=1.5, labelsize=6.5)


def times_fmt(v, _):
    return f"{v:g}$\\times$"


def ksweep():
    k = [1, 4, 16, 32, 80, 128, 256, 512]
    kr = [1, 4, 16, 80, 256]
    data = {
        "RTX 5070 Ti Laptop (Blackwell)": ([3.19, 3.80, 3.36, 3.43, 3.64, 3.64, 4.15, 5.16], [1.02, 1.65, 2.11, 3.14, 3.94],
                                            [0.65, 1.25, 1.87, 2.34, 3.02, 3.22, 3.88, 4.99]),
        "RTX 5000 Ada": ([3.46, 4.10, 3.76, 3.76, 3.60, 3.31, 3.80, 5.12], [1.16, 1.68, 2.17, 2.96, 3.57],
                         [0.56, 1.12, 1.82, 2.33, 2.83, 2.82, 3.48, 4.89]),
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.0), sharey=True)
    for ax, (name, (reb, ref, tr)) in zip(axes, data.items()):
        style(ax)
        ax.axvspan(50, 100, color="#f1f0ec", zorder=0)
        ax.text(70.7, 6.3, "SPH", ha="center", va="top", fontsize=6.5, color=INK2)
        ax.axhline(1, color=INK, lw=0.8, zorder=1)
        ax.plot(k, reb, "-o", color=BLUE, lw=1.6, ms=3.5, zorder=3)
        ax.plot(kr, ref, "--s", color=ORANGE, lw=1.6, ms=3.5, zorder=3)
        ax.plot(k, tr, ":^", color=AQUA, lw=1.6, ms=3.8, zorder=3)
        ax.text(16, reb[2] * 1.13, "rebuilt BVH", color=INK, fontsize=6.5, ha="center", va="bottom")
        ax.text(300, ref[-1] * 0.86, "refitted BVH", color=INK, fontsize=6.5, va="top")
        ax.text(1.35, tr[0], "search only (no BVH cost)", color=INK, fontsize=6.5, va="center")
        ax.text(1900, 0.93, "RT and grid equal", fontsize=6.5, color=INK2, ha="right", va="top")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.8, 2000)
        ax.set_ylim(0.4, 7)
        ax.xaxis.set_major_locator(FixedLocator(k))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(times_fmt))
        ax.set_title(name, fontsize=7, pad=3)
        ax.set_xlabel("Neighbours per particle", labelpad=1)
    axes[0].set_ylabel("RT time / grid time\n(above 1: RT slower)", labelpad=2)
    fig.tight_layout(pad=0.3, w_pad=1.2)
    fig.savefig(os.path.join(HERE, "rt_ksweep.pdf"))
    plt.close(fig)


def multires():
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.45, 2.05))
    r = [1, 2, 4, 8, 16]
    single = [1.82, 6.79, 38.56, 290.32, 2635.15]
    ml = [4.04, 4.79, 4.58, 4.49, 4.40]
    rt = [6.95, 9.50, 8.66, 8.56, 8.67]
    style(a)
    a.plot(r, single, ":^", color=AQUA, lw=1.6, ms=3.8)
    a.plot(r, rt, "-o", color=BLUE, lw=1.6, ms=3.5)
    a.plot(r, ml, "--s", color=ORANGE, lw=1.6, ms=3.5)
    a.set_xscale("log", base=2)
    a.set_yscale("log")
    a.xaxis.set_major_locator(FixedLocator(r))
    a.xaxis.set_minor_locator(NullLocator())
    a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_ylim(1, 5000)
    a.text(5.2, 700, "single grid", fontsize=6.3, ha="right")
    a.text(16, 11.5, "RT", fontsize=6.3, ha="right")
    a.text(16, 3.1, "multi-level grid", fontsize=6.3, ha="right", va="top")
    a.set_xlabel("Coarse / fine $h$ ratio", labelpad=1)
    a.set_ylabel("Time (ms)", labelpad=1)
    a.set_title("(a) Two-level scene", fontsize=7, pad=3)
    h = [100, 333, 1000, 3332, 9999]
    ml2 = [5.39, 5.16, 5.08, 4.93, 4.95]
    rt2 = [7.28, 7.58, 8.08, 8.65, 9.12]
    style(b)
    b.plot(h, rt2, "-o", color=BLUE, lw=1.6, ms=3.5)
    b.plot(h, ml2, "--s", color=ORANGE, lw=1.6, ms=3.5)
    b.set_xscale("log")
    b.xaxis.set_major_locator(FixedLocator([100, 1000, 10000]))
    b.xaxis.set_minor_locator(NullLocator())
    b.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"$10^{{{len(str(int(v))) - 1}}}$"))
    b.set_ylim(0, 10)
    b.text(100, 7.75, "RT", fontsize=6.3)
    b.text(100, 4.2, "multi-level grid", fontsize=6.3)
    b.text(9999, 9.12 + 0.35, "1.84$\\times$", fontsize=6.3, ha="right", color=INK2)
    b.text(100, 7.28 - 0.9, "1.35$\\times$", fontsize=6.3, color=INK2)
    b.set_xlabel("Range of $h$ (max / min)", labelpad=1)
    b.set_title("(b) Continuous $h$", fontsize=7, pad=3)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(os.path.join(HERE, "rt_multires.pdf"))
    plt.close(fig)


def confounds():
    steps = ["Fair:\nboth FP32,\nGPU-resident", "+ grid in\nFP64", "+ grid\nre-uploads\npositions", "+ grid copies\nall lists\nto host"]
    lap = [0.27, 3.47, 3.68, 6.52]
    ada = [0.27, 3.34, 3.69, 7.75]
    fig, ax = plt.subplots(figsize=(3.45, 2.1))
    style(ax)
    x = range(len(steps))
    w = 0.36
    ax.bar([i - w/2 - 0.01 for i in x], lap, w, color=BLUE, label="RTX 5070 Ti Laptop", zorder=3)
    ax.bar([i + w/2 + 0.01 for i in x], ada, w, color=ORANGE, label="RTX 5000 Ada", zorder=3)
    for i in x:
        ax.text(i - w/2 - 0.01, lap[i] * 1.08, f"{lap[i]:.2f}", ha="center", va="bottom", fontsize=6, color=INK)
        ax.text(i + w/2 + 0.01, ada[i] * 1.08, f"{ada[i]:.2f}", ha="center", va="bottom", fontsize=6, color=INK)
    ax.axhline(1, color=INK, lw=0.8, zorder=4)
    ax.text(3.62, 1.08, "equal", fontsize=6.3, color=INK2, ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(0.15, 20)
    ax.yaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2, 4, 8]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(times_fmt))
    ax.set_xticks(list(x))
    ax.set_xticklabels(steps, fontsize=6.3)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Apparent RT speed-up\n(grid time / RT time)", labelpad=1)
    ax.legend(loc="upper left", fontsize=6.3, frameon=False, handlelength=1.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(HERE, "rt_confounds.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    ksweep()
    multires()
    confounds()
