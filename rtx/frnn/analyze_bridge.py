import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import analyze_abl as A


def main(path, K=50):
    T = defaultdict(list)
    I = {}
    for d in A.parse(path):
        sc = (d["dist"], d["radius"])
        v = d["variant"] + ("+cg" if d.get("compact", 0) else "") + ("+fp64" if d.get("fp64", 0) else "")
        if d["build"] == "instr":
            I[(sc, v)] = d
        else:
            T[(sc, v)].append(d)
    mn = lambda L, f: min(f(r) for r in L)
    G = lambda r: r["grid_sort_ms"] + r["grid_cellstart_ms"] + r["grid_pass_ms"]
    RT = lambda r: r["rt_bvh_ms"] + r["rt_trace_ms"]
    scenes = sorted({k[0] for k in T})
    for sc in scenes:
        n = int(T[(sc, "base")][0]["n"])
        nbr = I[(sc, "base")]["sum_rt_nbr"]/n
        kc = I[(sc, "kcap")]["sum_rt_nbr"]/n
        g32 = min(mn(T[(sc, "base")], G), mn(T[(sc, "base+cg")], G))
        g64 = mn(T[(sc, "base+fp64")], G)
        rt = mn(T[(sc, "base")], RT)
        rtr = mn(T[(sc, "refit")], RT)
        rtk = mn(T[(sc, "kcap")], RT)
        x32 = T[(sc, "base")][0]
        x64 = T[(sc, "base+fp64")][0]
        full_bytes = I[(sc, "base")]["sum_grid_nbr"]*4 + n*8
        bw = full_bytes/(x32["xfer_d2h_ms"]*1e-3)
        d2h_full = x32["xfer_d2h_ms"]
        h2d64 = x64["xfer_h2d_ms"]
        d2h_k = n*K*4/bw*1e3
        rtnn = rtk + d2h_k
        cuns = g64 + h2d64 + d2h_full
        print(f"== {sc[0]} {sc[1]}  N={n}  true nbrs/q={nbr:.1f}  returned with K={K}: {kc:.1f} ({100*kc/nbr:.0f}%)  D2H bw={bw/1e9:.1f} GB/s")
        print(f"   fair (complete, FP32, GPU-resident):   grid {g32:7.2f} ms   RT {rt:7.2f} ms   -> RT/grid {rt/g32:5.2f}x")
        print(f"   RT best case (refit BVH):              grid {g32:7.2f} ms   RT {rtr:7.2f} ms   -> RT/grid {rtr/g32:5.2f}x")
        steps = [
            ("+ RT stops at K=50 (RTNN range search)", rtk, g32),
            ("+ grid in FP64 (cuNSearch default)", rtk, g64),
            ("+ grid re-uploads positions (H2D)", rtk, g64 + h2d64),
            ("+ grid copies full lists to host", rtk, g64 + h2d64 + d2h_full),
            ("+ RT copies its K-capped list to host", rtnn, cuns),
        ]
        for name, a, b in steps:
            print(f"   {name:40} grid-side {b:7.2f} ms  RT-side {a:7.2f} ms  -> apparent RT speedup {b/a:5.2f}x")


if __name__ == "__main__":
    main(sys.argv[1])
