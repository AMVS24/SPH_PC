import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

out = sys.argv[1]
plt.rcParams.update({"font.size": 8, "font.family": "serif"})

r = 0.55
pts = {
    "a": (1.0, 1.2), "b": (1.9, 0.8), "c": (1.5, 2.0),
    "d": (2.6, 3.6), "e": (3.4, 3.2), "f": (3.0, 4.2),
    "g": (6.0, 1.0), "h": (6.8, 1.6), "i": (6.3, 2.2),
    "j": (7.4, 3.6), "k": (8.0, 4.3), "l": (7.0, 4.4),
}
leaves = {"A": "abc", "B": "def", "C": "ghi", "D": "jkl"}
parents = {"L": "AB", "R": "CD"}
query = (2.85, 3.85)

def box_of(names):
    xs = [pts[n][0] for n in names]; ys = [pts[n][1] for n in names]
    return min(xs) - r, min(ys) - r, max(xs) + r, max(ys) + r

def union(b1, b2):
    return min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])

def contains(b, p):
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]

leaf_box = {k: box_of(v) for k, v in leaves.items()}
par_box = {k: union(leaf_box[v[0]], leaf_box[v[1]]) for k, v in parents.items()}
hot = "#c0392b"; cold = "#7f8c8d"; ink = "#222222"

fig, (ax, tx) = plt.subplots(2, 1, figsize=(3.4, 4.4), gridspec_kw={"height_ratios": [1.3, 1]})

for k, b in par_box.items():
    hit = contains(b, query)
    ax.add_patch(Rectangle((b[0] - 0.15, b[1] - 0.15), b[2] - b[0] + 0.3, b[3] - b[1] + 0.3,
                           fill=False, lw=1.6, ec=hot if hit else cold))
    ax.text(b[0] - 0.1, b[3] + 0.22, k, color=hot if hit else cold, fontsize=9, weight="bold")
for k, b in leaf_box.items():
    hit = contains(b, query)
    ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, lw=1.1,
                           ls="--", ec=hot if hit else cold))
    ax.text(b[2] - 0.05, b[1] + 0.05, k, color=hot if hit else cold, fontsize=8, ha="right", va="bottom")
for n, (x, y) in pts.items():
    hit = contains((x - r, y - r, x + r, y + r), query)
    ax.add_patch(Rectangle((x - r, y - r), 2 * r, 2 * r, fill=True, lw=0.4,
                           fc=(0.75, 0.22, 0.17, 0.15) if hit else (0.5, 0.5, 0.5, 0.06),
                           ec=hot if hit else "#b0b0b0"))
    ax.plot(x, y, "o", ms=3.5, color=ink)
ax.add_patch(Circle(query, r, fill=False, ls=":", lw=1.0, ec=ink))
ax.plot(*query, marker="*", ms=10, color=hot, mec=ink, mew=0.5, zorder=5)
ax.annotate("query particle", xy=query, xytext=(3.6, -0.25), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.7, color=ink))
ax.set_xlim(-0.3, 9.0); ax.set_ylim(-0.6, 5.6); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("(a) Particle boxes and the boxes grouping them", fontsize=8)

tree_pos = {"root": (0.5, 0.9), "L": (0.25, 0.6), "R": (0.75, 0.6),
            "A": (0.12, 0.3), "B": (0.38, 0.3), "C": (0.62, 0.3), "D": (0.88, 0.3)}
edges = [("root", "L"), ("root", "R"), ("L", "A"), ("L", "B"), ("R", "C"), ("R", "D")]
visited = {"root"}
for k in ("L", "R"):
    if contains(par_box[k], query):
        visited.add(k)
        for c in parents[k]:
            if contains(leaf_box[c], query):
                visited.add(c)
for a, b in edges:
    (x1, y1), (x2, y2) = tree_pos[a], tree_pos[b]
    on = a in visited and b in visited
    tx.plot([x1, x2], [y1, y2], color=hot if on else cold, lw=1.6 if on else 0.8)
for k, (x, y) in tree_pos.items():
    on = k in visited
    tx.add_patch(Rectangle((x - 0.07, y - 0.055), 0.14, 0.11, fc="white", ec=hot if on else cold, lw=1.2, zorder=3))
    tx.text(x, y, k, ha="center", va="center", color=hot if on else cold, weight="bold", zorder=4)
for k, (x, y) in tree_pos.items():
    if k in leaves:
        tx.text(x, y - 0.12, ",".join(leaves[k]), ha="center", va="top", fontsize=7, color=ink)
tx.text(0.5, -0.17, "red: boxes the query enters; grey: whole branches skipped", ha="center", fontsize=7, color=ink)
tx.set_xlim(0, 1); tx.set_ylim(-0.22, 1.0); tx.axis("off")
tx.set_title("(b) The same boxes as a tree", fontsize=8)

fig.tight_layout()
fig.savefig(out + ".pdf", bbox_inches="tight")

