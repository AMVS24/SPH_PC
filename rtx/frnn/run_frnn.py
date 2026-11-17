"""
ORCS-style scenario sweep: grid neighbour search vs RT-core neighbour search on frozen particle sets.

Scenarios follow Meneses et al. (arXiv:2601.15633): a 3D box of side 103, three initial distributions
(lattice, disordered = uniform random, cluster = Gaussian blobs with sigma 25) and four radius settings
(r = 1, r = 160, r ~ U[1, 160], r ~ LN(1, 2) clipped to [1, 330]). A fifth radius is added per N so that a
disordered scene has ~80 neighbours per particle, matching our SPH solver.

Each configuration runs build/frnn.exe once; inside it the grid and RT searches alternate and are checked
against each other per query. Results go to data/frnn_results.json after every run (resumable).

    python rtx/frnn/run_frnn.py            run (or resume) the sweep
    python rtx/frnn/run_frnn.py --list     print the configurations and stop
    python rtx/frnn/run_frnn.py --summary  print the results table
"""

import argparse
import json
import math
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(HERE, "build", "frnn.exe" if os.name == "nt" else "frnn")
OUT = os.path.join(HERE, "data", "frnn_results.json")

BOX = 103.0
DISTS = ["lattice", "disordered", "cluster"]
NS = [50000, 200000, 1000000]
COOLDOWN_S = 5


def sph_radius(n, target=80.0):
    rho = n/BOX**3
    return (target/(rho*4.0/3.0*math.pi))**(1.0/3.0)


def radii(n):
    return [("r=1", "const:1"),
            ("SPH ~80 nbrs", f"const:{sph_radius(n):.4f}"),
            ("r=160", "const:160"),
            ("r~U[1,160]", "uniform:1:160"),
            ("r~LN(1,2)[1,330]", "lognormal:1:2:1:330")]


def configs():
    for n in NS:
        for d in DISTS:
            for label, spec in radii(n):
                yield n, d, label, spec


def fields(line):
    return dict(t.split("=", 1) for t in line.split() if "=" in t)


def run_one(n, d, spec):
    cmd = [EXE, "--dist", d, "--radius", spec, "--n", str(n), "--box", str(BOX), "--budget-s", "15"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    line = next((l for l in p.stdout.splitlines() if l.startswith("FRNN")), None)
    if line is None:
        return {"error": (p.stderr[-800:] or f"exit {p.returncode}")}
    f = fields(line)
    r = {k: float(v) if k not in ("dist", "radius") else v for k, v in f.items()}
    r["returncode"] = p.returncode
    return r


def speedups(r):
    s = {"search": r["grid_total_ms"]/r["rt_total_ms"]}
    if r["rt_trace_list_ms"] >= 0:
        grid_sph = r["grid_build_ms"] + 2*r["grid_pass_ms"]
        rt_sph = r["rt_sched_ms"] + r["rt_bvh_ms"] + r["rt_trace_list_ms"]
        s["sph_style"] = grid_sph/rt_sph
    return s


def summary(doc):
    print(f"{'N':>8} {'dist':10} {'radius':18} {'nbrs/particle':>14} {'grid ms':>10} {'RT ms':>10} {'RT speedup':>11} {'SPH-style':>10}  check")
    for n, d, label, spec in configs():
        k = f"{n}|{d}|{spec}"
        r = doc.get(k)
        if not r:
            continue
        if "error" in r:
            print(f"{n:>8} {d:10} {label:18} ERROR {r['error'][:80]}")
            continue
        s = speedups(r)
        sph = f"{s['sph_style']:.2f}x" if "sph_style" in s else "no list"
        chk = "ok" if r["valid"] == 1 else f"MISMATCH ({int(r['mismatched_queries'])})"
        print(f"{n:>8} {d:10} {label:18} {r['mean_nbrs']:>14.1f} {r['grid_total_ms']:>10.3f} {r['rt_total_ms']:>10.3f} {s['search']:>10.2f}x {sph:>10}  {chk}")


def main():
    global BOX, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--box", type=float, default=BOX)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--cases", help="semicolon-separated n|dist|radius-label entries to run instead of the full matrix")
    a = ap.parse_args()
    BOX, OUT = a.box, a.out
    doc = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = list(configs())
    if a.cases:
        want = [c.split("|") for c in a.cases.split(";")]
        todo = [c for c in todo if any(int(w[0]) == c[0] and w[1] == c[1] and w[2] == c[2] for w in want)]
    if a.list:
        for c in todo:
            print(*c)
        return
    if a.summary:
        summary(doc)
        return
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for n, d, label, spec in todo:
        k = f"{n}|{d}|{spec}"
        if k in doc and "error" not in doc[k]:
            continue
        t0 = time.time()
        r = run_one(n, d, spec)
        r["wall_s"] = time.time() - t0
        doc[k] = r
        json.dump(doc, open(OUT + ".tmp", "w"), indent=2)
        os.replace(OUT + ".tmp", OUT)
        if "error" in r:
            print(f"{n:>8} {d:10} {label:18} ERROR {r['error'][:200]}", flush=True)
        else:
            print(f"{n:>8} {d:10} {label:18} nbrs={r['mean_nbrs']:.1f} grid={r['grid_total_ms']:.3f}ms rt={r['rt_total_ms']:.3f}ms "
                  f"speedup={speedups(r)['search']:.2f}x valid={int(r['valid'])} ({r['wall_s']:.0f}s)", flush=True)
        time.sleep(COOLDOWN_S)


if __name__ == "__main__":
    main()
