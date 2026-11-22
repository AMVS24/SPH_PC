"""
Multi-resolution investigation: single uniform grid vs multi-level grid vs RT cores (SPH density pass).

  multires  coarse particles everywhere + a central fine region, h smaller by `ratio` (two resolution levels)
  graded    h grows smoothly from the centre outwards by a factor `ratio` (continuous adaptivity)

    python rtx/frnn/run_multires.py            run (or resume) the cases
    python rtx/frnn/run_multires.py --summary  print the results

Results: data/multires_results.json, saved after every case.
"""

import argparse
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(HERE, "build", "frnn.exe" if os.name == "nt" else "frnn")
OUT = os.path.join(HERE, "data", "multires_results.json")

CASES = ([("multires", r, 1000000) for r in (1, 1.5, 2, 2.5, 3, 4, 6, 8)] +
         [("graded", r, 1000000) for r in (2, 4, 8, 16)] +
         [("multires", 4, n) for n in (250000, 4000000)] +
         [("graded", 8, n) for n in (250000, 4000000)])


def fields(line):
    return dict(t.split("=", 1) for t in line.split() if "=" in t)


def run(dist, ratio, n):
    cmd = [EXE, "--dist", dist, "--ratio", str(ratio), "--n", str(n), "--box", "100", "--density", "--mlgrid", "--budget-s", "6"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    line = next((l for l in p.stdout.splitlines() if l.startswith("FRNN")), None)
    if line is None:
        return {"error": p.stderr[-600:] or f"exit {p.returncode}"}
    f = fields(line)
    keep = ["valid", "reps", "rmin", "rmax", "ml_levels", "ml_cells", "worst_rel_density_diff", "ml_worst_rel_diff",
            "grid_build_ms", "grid_pass_ms", "grid_total_ms", "grid_total_sd", "ml_build_ms", "ml_pass_ms", "ml_total_ms",
            "ml_total_sd", "rt_sched_ms", "rt_bvh_ms", "rt_trace_count_ms", "rt_total_ms", "rt_total_sd", "mean_density"]
    return {k: float(f[k]) for k in keep if k in f}


def show(dist, ratio, n, r):
    if "error" in r:
        print(f"{dist:8s} ratio={ratio:<4g} n={n:>8d}  ERROR {r['error'][:150]}", flush=True)
        return
    print(f"{dist:8s} ratio={ratio:<4g} n={n:>8d}  levels={int(r['ml_levels'])}  single grid={r['grid_total_ms']:9.3f}  "
          f"multi-level grid={r['ml_total_ms']:8.3f}  RT={r['rt_total_ms']:7.3f} ms  "
          f"RT vs single={r['grid_total_ms']/r['rt_total_ms']:6.2f}x  RT vs multi-level={r['ml_total_ms']/r['rt_total_ms']:5.2f}x  valid={int(r['valid'])}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    doc = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for dist, ratio, n in CASES:
        k = f"{dist}|{ratio:g}|{n}"
        if a.summary:
            if k in doc:
                show(dist, ratio, n, doc[k])
            continue
        if k in doc and "error" not in doc[k]:
            show(dist, ratio, n, doc[k])
            continue
        doc[k] = run(dist, ratio, n)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(doc, open(OUT + ".tmp", "w"), indent=1)
        os.replace(OUT + ".tmp", OUT)
        show(dist, ratio, n, doc[k])


if __name__ == "__main__":
    main()
