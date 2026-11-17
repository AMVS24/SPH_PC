"""
Grid-CUDA (FP32) vs RT-core neighbour search, same scene, same protocol, run back to back.

Alternates the two executables at every particle count so both see the same thermal and power
state. Writes rtx/bench/data/compare_results.json after every run (resumable) and never overwrites
other result files.

    python rtx/bench/compare_sweep.py
"""

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(DATA, "compare_results.json")

EXES = {
    "grid": os.path.join(ROOT, "gpu", "bench", "build", "bench.exe"),
    "rtx": os.path.join(HERE, "build", "bench.exe"),
}
VARIANT_ARG = {"grid": "gpu_fp32", "rtx": "gpu"}

COUNTS = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000,
          1000000, 2000000, 4000000, 6000000, 8000000, 10000000, 12000000]

WINDOW = 10
MIN_STEPS = 2 * WINDOW
MAX_STEPS = 150
RUN_BUDGET_MS = 20000.0
COOLDOWN_S = 4


def steps_for(est_ms):
    if est_ms is None or est_ms <= 0:
        return MAX_STEPS
    s = int(RUN_BUDGET_MS / est_ms)
    s -= s % WINDOW
    return max(MIN_STEPS, min(MAX_STEPS, s))


def fields(line):
    return dict(tok.split("=", 1) for tok in line.split() if "=" in tok)


def run(which, n, steps):
    cmd = [EXES[which], "--variant", VARIANT_ARG[which], "--n", str(n), "--steps", str(steps)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if p.returncode != 0:
        return {"error": (p.stderr[-800:] or f"exit code {p.returncode}")}
    out = {}
    for line in p.stdout.splitlines():
        if line.startswith("RESULT"):
            f = fields(line)
            out.update(steps=int(f["steps"]), ms=float(f["mean_ms"]), stdev=float(f["stdev_ms"]))
        elif line.startswith("NBR"):
            f = fields(line)
            out.update(cap=int(f["cap"]), overflow=int(f["overflow"]), total=int(f["total"]))
    return out if "ms" in out else {"error": "no RESULT line"}


def main():
    os.makedirs(DATA, exist_ok=True)
    doc = json.load(open(OUT)) if os.path.exists(OUT) else {}
    est = {"grid": None, "rtx": None}
    prev_n = None
    for n in COUNTS:
        for which in ("grid", "rtx"):
            k = f"{which}|{n}"
            if k in doc and "error" not in doc[k]:
                est[which] = doc[k]["ms"]
                continue
            e = None if est[which] is None else est[which]*n/prev_n
            steps = steps_for(e)
            t0 = time.time()
            r = run(which, n, steps)
            doc[k] = r
            json.dump(doc, open(OUT + ".tmp", "w"), indent=2)
            os.replace(OUT + ".tmp", OUT)
            if "error" in r:
                print(f"{which:5s} n={n:>9d}  ERROR {r['error'][:200]}", flush=True)
            else:
                extra = f"  overflow={r['overflow']}" if "overflow" in r else ""
                print(f"{which:5s} n={n:>9d}  steps={r['steps']:>3d}  {r['ms']:>10.3f} +/- {r['stdev']:.3f} ms{extra}  ({time.time() - t0:.0f}s)", flush=True)
                est[which] = r["ms"]
            time.sleep(COOLDOWN_S)
        prev_n = n


if __name__ == "__main__":
    sys.exit(main())
