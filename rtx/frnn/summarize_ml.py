import sys


def fields(line):
    return dict(t.split("=", 1) for t in line.split() if "=" in t)


def f(x):
    return float(x)


def main(path):
    rows = [fields(l) for l in open(path, encoding="utf-8", errors="replace") if l.startswith("FRNNML")]
    for l in open(path, encoding="utf-8", errors="replace"):
        if l.startswith("INFO"):
            print(l.rstrip())
    hdr = f"{'mode':8} {'scene':16} {'N':>8} {'hrange':>7} {'nbrs':>7} {'valid':>5} | {'single b/p ms':>17} {'ML b/p ms':>15} {'RT s+bvh/trace ms':>18} | {'single':>9} {'ML':>8} {'RT':>8} | {'RT/single':>9} {'RT vs ML':>8}"
    print(hdr)
    print("-"*len(hdr))
    for r in rows:
        scene = f"multires r={f(r['ratio']):g}" if r["dist"] == "multires" else f"profile a={f(r['alpha']):g}"
        st, mt, rt = f(r["single_total_ms"]), f(r["ml_total_ms"]), f(r["rt_total_ms"])
        sbp = f"{f(r['single_build_ms']):.2f}/{f(r['single_pass_ms']):.2f}" if st > 0 else "skipped"
        mbp = f"{f(r['ml_build_ms']):.2f}/{f(r['ml_pass_ms']):.2f}"
        rbp = f"{f(r['rt_sched_ms']) + f(r['rt_bvh_ms']):.2f}/{f(r['rt_trace_ms']):.2f}"
        sp_s = f"{st/rt:.2f}x" if st > 0 else "-"
        sp_m = f"{mt/rt:.2f}x"
        nb = r["mean_nbrs"] if r["mode"] == "count" else "-"
        print(f"{r['mode']:8} {scene:16} {r['n']:>8} {f(r['hrange']):7.2f} {nb:>7} {r['valid']:>5} | {sbp:>17} {mbp:>15} {rbp:>18} | "
              f"{(f'{st:.2f}' if st > 0 else '-'):>9} {mt:8.2f} {rt:8.2f} | {sp_s:>9} {sp_m:>8}")


if __name__ == "__main__":
    main(sys.argv[1])
