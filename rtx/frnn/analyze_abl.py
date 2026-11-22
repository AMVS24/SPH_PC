import sys
import statistics as st
from collections import defaultdict


def parse(path):
    rows = []
    for line in open(path):
        if not line.startswith("ABL "):
            continue
        d = {}
        for t in line.split()[1:]:
            if "=" in t:
                k, v = t.split("=", 1)
                try:
                    d[k] = float(v)
                except ValueError:
                    d[k] = v
        rows.append(d)
    return rows


def key(d):
    return (d["dist"] + (":c%g" % d["core"] if d["dist"] == "profile" else "") + (":b%g" % d["box"] if d["box"] != 103.0 else ""), int(d["n"]), d["k"], d["mode"], d["variant"] + ("+cg" if d.get("compact", 0) else "") + ("+fp64" if d.get("fp64", 0) else "") + ("%d" % d["kcap"] if d.get("kcap", 0) else ""), int(d["div"]))


def main(path):
    rows = parse(path)
    timing = defaultdict(list)
    instr = {}
    for d in rows:
        if d["build"] == "instr":
            instr[key(d)] = d
        else:
            timing[key(d)].append(d)
    print(f"{'dist':16} {'N':>8} {'k':>5} {'mode':7} {'variant':12} {'div':>3} | {'grid ms':>8} {'[min-max]':>15} | {'rt b+t ms':>9} {'[min-max]':>15} {'trace':>7} {'bvh':>6} | {'RT/grid':>7} | {'nbr/q':>6} {'calls/q':>7} {'calls2/q':>8} {'gtest/q':>7} | {'ns/call':>7} {'valid':>5} {'cv':>5}")
    for k in sorted(timing, key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5])):
        L = timing[k]
        g = [r["grid_sort_ms"] + r["grid_cellstart_ms"] + r["grid_pass_ms"] for r in L]
        rt = [r["rt_bvh_ms"] + r["rt_trace_ms"] for r in L]
        tr = st.median([r["rt_trace_ms"] for r in L])
        bv = st.median([r["rt_bvh_ms"] for r in L])
        gm, rm = st.median(g), st.median(rt)
        I = instr.get(k)
        n = k[1]
        if I:
            nbr = (I["sum_rt_nbr"] if I["mode"] == "count" else float("nan"))/n
            calls = I["sum_calls"]/n
            calls2 = I["sum_calls2"]/n
            gt = I["sum_cand"]/n
            nsc = tr*1e6/I["sum_calls"] if I["sum_calls"] > 0 else float("nan")
        else:
            nbr = calls = calls2 = gt = nsc = float("nan")
        valid = int(L[0]["valid"])
        cv = max(r["rt_bt_cv"] for r in L)
        print(f"{k[0]:16} {n:8d} {k[2]:5.0f} {k[3]:7} {k[4]:12} {k[5]:3d} | {gm:8.3f} [{min(g):6.3f}-{max(g):6.3f}] | {rm:9.3f} [{min(rt):6.3f}-{max(rt):6.3f}] {tr:7.3f} {bv:6.3f} | {(rm/gm if gm > 0 else float("nan")):7.2f} | {nbr:6.1f} {calls:7.1f} {calls2:8.1f} {gt:7.1f} | {nsc:7.4f} {valid:5d} {cv:5.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
