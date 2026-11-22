import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run_csv, scene_json, out = sys.argv[1], sys.argv[2], sys.argv[3]
plt.rcParams.update({"font.size": 7, "font.family": "serif"})

p = json.load(open(scene_json))
xmin, ymin, xmax, ymax = p["domain"]
d = np.genfromtxt(run_csv, delimiter=",", names=True)
times = np.unique(d["t"])
wanted = [0.0, 0.16, 0.32, 0.48, 0.64, 0.80]

fig, axes = plt.subplots(2, 3, figsize=(3.4, 2.5), sharex=True, sharey=True)
for ax, tw in zip(axes.flat, wanted):
    t = times[np.argmin(np.abs(times - tw))]
    s = d[d["t"] == t]
    ax.plot(s["x"], s["y"], "o", ms=1.6, color="#2c5d8a")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect("equal")
    ax.set_title(f"t = {t:.2f} s", fontsize=7, pad=2)
    ax.tick_params(length=2, pad=1, labelsize=6)
    ax.set_xticks([0, 0.25, 0.5]); ax.set_yticks([0, 0.25, 0.5])
for ax in axes[1]:
    ax.set_xlabel("x (m)", labelpad=1)
for ax in axes[:, 0]:
    ax.set_ylabel("y (m)", labelpad=1)
fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.6)
fig.savefig(out + ".pdf", bbox_inches="tight")
