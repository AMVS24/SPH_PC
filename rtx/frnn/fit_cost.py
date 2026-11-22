import argparse
import json
import math

import numpy as np

PHASES = {
    "grid_sort": ["one", "n"],
    "grid_cellstart": ["one", "cells"],
    "grid_pass": ["one", "n", "cand"],
    "rt_sched": ["one", "n"],
    "rt_bvh": ["one", "n"],
    "rt_trace": ["one", "n", "is"],
}
LOOP = {"grid_sort": "loop_grid_sort_ms", "rt_sched": "loop_rt_sched_ms", "rt_bvh": "loop_rt_bvh_ms"}
GRID = ["grid_sort", "grid_cellstart", "grid_pass"]
RT = ["rt_sched", "rt_bvh", "rt_trace"]


def parse(paths):
    rows = []
    for p in paths:
        for line in open(p):
            if line.startswith("COST"):
                rows.append(dict(t.split("=", 1) for t in line.split()[1:]))
    return rows


def key(r):
    return (r["dist"], r["radius"], r["k"], r["n"], r["box"])


def merge(rows):
    instr = {key(r): r for r in rows if r["build"] == "instr"}
    out = []
    for r in rows:
        if r["build"] != "timing":
            continue
        i = instr.get(key(r))
        if i is None:
            continue
        n = int(r["n"])
        d = {
            "dist": r["dist"], "radius": r["radius"], "k": float(r["k"]), "n": n, "box": float(r["box"]),
            "rmax": float(r["rmax"]), "valid": int(r["valid"]) and int(i["valid"]), "gpu": r["gpu"],
            "one": 1.0, "cells": float(r["cells"]), "cand": float(i["sum_cand"]), "is": float(i["sum_is"]),
            "nbr": float(i["sum_nbr"]), "max_nbr": int(i["max_nbr"]), "max_is": int(i["max_is"]), "max_cand": int(i["max_cand"]),
            "is_w": float(i.get("warp_is", "nan")), "cand_w": float(i.get("warp_cand", "nan")),
        }
        d["n_f"] = float(n)
        for ph in PHASES:
            v = float(r[ph + "_ms"])
            if ph in LOOP and float(r.get(LOOP[ph], -1)) > 0:
                v = float(r[LOOP[ph]])
            d[ph] = v
        out.append(d)
    return out


def feats(d, ph):
    return np.array([d["n_f"] if f == "n" else d[f] for f in PHASES[ph]])


def fit(train):
    coef = {}
    for ph in PHASES:
        X = np.array([feats(d, ph) for d in train])
        y = np.array([d[ph] for d in train])
        w = 1.0/np.maximum(y, 1e-3)
        c, *_ = np.linalg.lstsq(X*w[:, None], y*w, rcond=None)
        coef[ph] = c
    return coef


def pred(coef, d, ph):
    return float(feats(d, ph) @ coef[ph])


def totals(coef, d):
    g = sum(pred(coef, d, p) for p in GRID)
    r = sum(pred(coef, d, p) for p in RT)
    return g, r


def meas(d):
    return sum(d[p] for p in GRID), sum(d[p] for p in RT)


def label(d):
    return f"{d['dist'][:5]:5s} N={d['n']:>7d} {('k=' + format(d['k'], 'g')) if d['radius'] == 'k' else d['radius']:22s}"


def report_rows(coef, rows, title):
    print(f"\n== {title} ==")
    print(f"{'config':40s} {'nbr/q':>8s} {'grid meas':>10s} {'grid pred':>10s} {'err%':>6s} {'RT meas':>10s} {'RT pred':>10s} {'err%':>6s} {'spd meas':>9s} {'spd pred':>9s} win")
    errs_g, errs_r, wins = [], [], 0
    for d in rows:
        gm, rm = meas(d)
        gp, rp = totals(coef, d)
        eg, er = 100*(gp - gm)/gm, 100*(rp - rm)/rm
        errs_g.append(abs(eg)); errs_r.append(abs(er))
        ok = (gm > rm) == (gp > rp)
        wins += ok
        print(f"{label(d):40s} {d['nbr']/d['n']:8.1f} {gm:10.3f} {gp:10.3f} {eg:6.1f} {rm:10.3f} {rp:10.3f} {er:6.1f} {gm/rm:9.3f} {gp/rp:9.3f} {'ok' if ok else 'WRONG'}")
    if rows:
        print(f"mean |err| grid {np.mean(errs_g):.1f}%  RT {np.mean(errs_r):.1f}%  max grid {np.max(errs_g):.1f}%  max RT {np.max(errs_r):.1f}%  winner correct {wins}/{len(rows)}")
    return errs_g, errs_r


def phase_errors(coef, rows, title):
    print(f"\n-- per-phase |err| % ({title}) --")
    for ph in PHASES:
        e = [abs(pred(coef, d, ph) - d[ph])/max(d[ph], 1e-6)*100 for d in rows]
        print(f"{ph:15s} mean {np.mean(e):6.1f}  max {np.max(e):6.1f}")


def crossover(coef, rows, n, box):
    sub = [d for d in rows if d["n"] == n and d["dist"] == "disordered" and d["radius"] == "k"]
    nb = np.array([d["nbr"]/d["n"] for d in sub])
    fi, fc = PHASES["rt_trace"][2], PHASES["grid_pass"][2]
    isq = np.array([d[fi]/d["n"] for d in sub])
    caq = np.array([d[fc]/d["n"] for d in sub])
    a_is = np.polyfit(nb, isq, 1)
    a_ca = np.polyfit(nb, caq, 1)
    rho = n/box**3
    print(f"\n== crossover, disordered N={n} box={box} ==")
    print(f"IS calls/query   = {a_is[1]:.2f} + {a_is[0]:.3f} * nbrs   (geometric ideal 1 + 1.91*(nbrs-1))")
    print(f"grid tests/query = {a_ca[1]:.2f} + {a_ca[0]:.3f} * nbrs   (geometric ideal ~6.45*nbrs for cell = r)")
    ks = np.concatenate([np.linspace(1.0, 10, 91), np.linspace(10, 2000, 1991)])
    diff = []
    for k in ks:
        r = (3*(k - 1)/(4*math.pi*rho))**(1/3) if k > 1 else 0.5*(1/rho)**(1/3)
        g = max(1, min(1024, int(box/r)))
        d = {"one": 1.0, "n_f": float(n), "cells": float(g**3), fc: n*(a_ca[1] + a_ca[0]*k), fi: n*(a_is[1] + a_is[0]*k)}
        gt = sum(float(feats(d, p) @ coef[p]) for p in GRID)
        rt = sum(float(feats(d, p) @ coef[p]) for p in RT)
        diff.append(rt - gt)
    diff = np.array(diff)
    s = np.sign(diff)
    idx = np.where(np.diff(s) != 0)[0]
    if len(idx):
        for i in idx:
            print(f"predicted crossover at ~{ks[i]:.1f} neighbours/query")
    else:
        print(f"no crossover predicted for 1 <= nbrs <= 2000: RT is always {'slower' if diff[0] > 0 else 'faster'}")
    per_nbr_rt = coef["rt_trace"][2]*a_is[0]*1e6
    per_nbr_grid = coef["grid_pass"][2]*a_ca[0]*1e6
    fixed_rt = (coef["rt_sched"][1] + coef["rt_bvh"][1] + coef["rt_trace"][1] + coef["rt_trace"][2]*a_is[1])*1e6
    fixed_grid = (coef["grid_sort"][1] + coef["grid_pass"][1] + coef["grid_pass"][2]*a_ca[1])*1e6
    print(f"per query, per neighbour: RT {per_nbr_rt:.3f} ns  grid {per_nbr_grid:.3f} ns  ratio {per_nbr_rt/per_nbr_grid:.2f}")
    print(f"per query, fixed:         RT {fixed_rt:.3f} ns  grid {fixed_grid:.3f} ns (+ grid cell-start {coef['grid_cellstart'][1]*1e6:.4f} ns per cell)")
    be = (fixed_rt - fixed_grid)/(coef['grid_cellstart'][1]*1e6)
    print(f"at ~1 neighbour, RT breaks even only if the grid has > {be:.1f} cells per particle")
    return a_is, a_ca


def breakdown(coef, rows, k, n):
    d = min((x for x in rows if x["n"] == n and x["radius"] == "k"), key=lambda x: abs(x["k"] - k))
    print(f"\n== RT cost breakdown at k={d['k']:g}, N={n} (measured / model terms, ms) ==")
    c = coef["rt_trace"]
    print(f"rt_sched {d['rt_sched']:.3f}  rt_bvh {d['rt_bvh']:.3f}  rt_trace {d['rt_trace']:.3f}  (trace model: const {c[0]:.3f} + per-query {c[1]*d['n_f']:.3f} + per-IS-call {c[2]*d[PHASES['rt_trace'][2]]:.3f})")
    g = coef["grid_pass"]
    print(f"grid_sort {d['grid_sort']:.3f}  grid_cellstart {d['grid_cellstart']:.3f}  grid_pass {d['grid_pass']:.3f}  (pass model: const {g[0]:.3f} + per-query {g[1]*d['n_f']:.3f} + per-test {g[2]*d[PHASES['grid_pass'][2]]:.3f})")


def ada_check(coef, orcs_rows, ada_json):
    ada = json.load(open(ada_json))
    print("\n== cross-GPU check vs frnn_box1000.json (RTX 5000 Ada): winner and speedup from laptop-fitted model ==")
    print(f"{'config':40s} {'Ada spd meas':>12s} {'model spd':>10s} win")
    ok = tot = 0
    for d in orcs_rows:
        rad = d["radius"]
        k = f"{d['n']}|{d['dist']}|{rad}"
        a = ada.get(k)
        if not a or "grid_total_ms" not in a:
            continue
        gp, rp = totals(coef, d)
        sm = a["grid_total_ms"]/a["rt_total_ms"]
        sp = gp/rp
        good = (sm > 1) == (sp > 1)
        ok += good; tot += 1
        print(f"{label(d):40s} {sm:12.3f} {sp:10.3f} {'ok' if good else 'WRONG'}")
    print(f"winner correct {ok}/{tot}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", nargs="+", required=True)
    ap.add_argument("--orcs", nargs="*", default=[])
    ap.add_argument("--ada", default=None)
    ap.add_argument("--train-k", default="1,16,64,256")
    ap.add_argument("--out", default=None)
    ap.add_argument("--warp", action="store_true")
    a = ap.parse_args()
    if a.warp:
        PHASES["grid_pass"] = ["one", "n", "cand_w"]
        PHASES["rt_trace"] = ["one", "n", "is_w"]

    sweep = [d for d in merge(parse(a.sweep)) if d["valid"] and d["radius"] == "k"]
    tk = {float(x) for x in a.train_k.split(",")}
    train = [d for d in sweep if d["k"] in tk]
    hold = [d for d in sweep if d["k"] not in tk]
    coef = fit(train)
    gpu = sweep[0]["gpu"] if sweep else "?"
    print(f"GPU: {gpu}   train k={sorted(tk)}  train rows={len(train)} held-out rows={len(hold)}")
    print("\n== fitted constants (ms; per-item constants also in ns) ==")
    for ph, c in coef.items():
        terms = "  ".join(f"{f}={v:.4g}" + (f" ({v*1e6:.4f} ns)" if f != "one" else "") for f, v in zip(PHASES[ph], c))
        print(f"{ph:15s} {terms}")
    report_rows(coef, train, "training points")
    report_rows(coef, hold, "held-out points")
    phase_errors(coef, hold, "held-out")
    for n in sorted({d["n"] for d in sweep}):
        crossover(coef, sweep, n, sweep[0]["box"])
    breakdown(coef, sweep, 80, max(d["n"] for d in sweep))
    orcs = [d for d in merge(parse(a.orcs)) if d["radius"] != "k"] if a.orcs else []
    if orcs:
        report_rows(coef, orcs, "ORCS scenarios, same GPU (box 1000), fully out of sample")
        phase_errors(coef, orcs, "ORCS")
        if a.ada:
            ada_check(coef, orcs, a.ada)
    if a.out:
        with open(a.out, "x") as f:
            json.dump({"gpu": gpu, "train_k": sorted(tk), "coef": {k: list(map(float, v)) for k, v in coef.items()}, "features": PHASES}, f, indent=1)


if __name__ == "__main__":
    main()
