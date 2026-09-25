"""Validation checks on solver output. Solver-agnostic: only reads files.

Snapshot format (one CSV for the whole run):  <run>.csv
    frame,t,id,x,y,vx,vy,rho,p
One row per particle per snapshot. `frame` is the snapshot index (0,1,2,...), `id` is the
particle index 0..n-1 matching the row order of <scene>.init.csv. Rows may appear in any order.

Step log (one per run, next to the snapshots):  <run>.steps.csv
    step,t,dt,h,c0,amax,vmax
One row per time step the solver took. t = simulation time at the START of the step;
h = smoothing length; c0 = numerical speed of sound (for p = k(rho-rho0) use c0 = sqrt(k));
amax = max |acceleration| over particles that step; vmax = max |velocity| that step.

Usage:
    python run_checks.py hydrostatic scenes/hydrostatic.json out/hydrostatic.csv
    python run_checks.py dambreak    scenes/dambreak.json    out/dambreak.csv \
        --ref reference/koshizuka_oka_front.csv
    python run_checks.py momentum    scenes/momentum.json    out/momentum.csv

Options:
    --ref FILE         reference front curve, CSV with columns T,Z (required for dambreak)
    --allow-no-ref     dambreak only: skip the reference comparison instead of failing
    --steps FILE       step log (default: <output without .csv>.steps.csv)
    --allow-no-steps   skip the time-step/CFL checks instead of failing when the log is missing
    --cfl C            CFL coefficient limit for both dt criteria (default 0.4)
    --mom-tol X        momentum-drift tolerance, relative (default 1e-4; tighten to ~1e-10
                       once the solver writes full-precision output)
    --init FILE        initial-condition CSV (default: <params without .json>.init.csv)

Reference CSV: normalised time T, normalised front position Z. Fill it from the paper's
digitised data -- do not invent values.

Exit code 0 = every fatal check passed; 1 = any fatal check failed or the file was malformed.
[INFO] lines are advisory and never affect the exit code.
"""
import argparse, json, os, sys
import numpy as np

COLUMNS = ["frame", "t", "id", "x", "y", "vx", "vy", "rho", "p"]
STEP_COLUMNS = ["step", "t", "dt", "h", "c0", "amax", "vmax"]
WALL_MARGIN = 4   # momentum scene: fluid must stay > WALL_MARGIN*dx from every wall
FRONT_LAYER = 3   # dam-break front is tracked in the bottom FRONT_LAYER*dx of the fluid
LEFT_STRIP = 2   # column height is read from particles within LEFT_STRIP*dx of the left wall


class BadFile(Exception):
    pass


class Report:
    def __init__(self):
        self.fails = 0
        self.total = 0

    def check(self, name, ok, detail="", fatal=True):
        ok = bool(ok)
        tag = "PASS" if ok else ("FAIL" if fatal else "INFO")
        print(f"[{tag}] {name}  {detail}".rstrip())
        self.total += 1
        self.fails += (not ok) and fatal


def _read_csv(path, what):
    try:
        d = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    except (OSError, ValueError) as e:
        raise BadFile(f"cannot read {what} {path}: {e}")
    if d.dtype.names is None or d.size == 0:
        raise BadFile(f"{what} {path} is empty or has no header row")
    return d


def load(path):
    """Return (times, snapshots): snapshots[k] is a structured array sorted by id."""
    d = _read_csv(path, "solver output")
    missing = [c for c in COLUMNS if c not in d.dtype.names]
    if missing:
        raise BadFile(f"solver output is missing columns {missing}; expected header {','.join(COLUMNS)}")
    d = d[np.argsort(d["frame"], kind="stable")]
    _, first = np.unique(d["frame"], return_index=True)
    ts, snaps = [], []
    for g in np.split(d, first[1:]):
        if np.ptp(g["t"]) > 1e-9:
            raise BadFile(f"frame {int(g['frame'][0])} contains rows with different t")
        ts.append(g["t"][0])
        snaps.append(g[np.argsort(g["id"], kind="stable")])
    return np.array(ts), snaps


def load_ref(path):
    d = _read_csv(path, "reference")
    if "T" not in d.dtype.names or "Z" not in d.dtype.names:
        raise BadFile(f"reference {path} must have columns T,Z")
    T, Z = d["T"], d["Z"]
    if len(T) < 3 or not (np.isfinite(T).all() and np.isfinite(Z).all()) or (np.diff(T) <= 0).any():
        raise BadFile(f"reference {path} needs >=3 finite rows with strictly increasing T")
    return T, Z


def load_steps(path):
    d = _read_csv(path, "steps log")
    missing = [c for c in STEP_COLUMNS if c not in d.dtype.names]
    if missing:
        raise BadFile(f"steps log is missing columns {missing}; expected header {','.join(STEP_COLUMNS)}")
    return d[np.argsort(d["step"], kind="stable")]


def integrity(r, P, ts, snaps, init):
    """Checks common to every scene. Returns False if the data is too broken for physics checks."""
    n, dx, g = P["n"], P["dx"], P["g"]
    x0, y0, x1, y1 = P["domain"]

    counts_ok = all(len(s) == n for s in snaps)
    r.check("particle count constant", counts_ok, f"n={n}")
    ids_ok = counts_ok and all(np.array_equal(s["id"], np.arange(n)) for s in snaps)
    r.check("ids are exactly 0..n-1 in every frame", ids_ok)
    finite = all(np.isfinite(np.column_stack([s[c] for c in COLUMNS[1:]])).all() for s in snaps)
    r.check("no NaN/inf", finite)
    if not (counts_ok and ids_ok and finite):
        return False

    tol = 1e-6
    inside = all(((s["x"] >= x0 - tol) & (s["x"] <= x1 + tol) &
                  (s["y"] >= y0 - tol) & (s["y"] <= y1 + tol)).all() for s in snaps)
    r.check("no wall penetration", inside)

    # Without these, a solver that stopped early (or only wrote t=0) would pass every physics check.
    dt = P["snapshot_dt"]
    r.check("times strictly increasing", (np.diff(ts) > 0).all())
    r.check("first frame at t=0", abs(ts[0]) <= dt, f"t_first={ts[0]:.4g}")
    r.check("reached t_end", ts[-1] >= P["t_end"] - dt, f"t_last={ts[-1]:.4g} t_end={P['t_end']:.4g}")

    if init is None:
        return True
    ok_shape = init.ndim == 2 and init.shape[0] == n and init.shape[1] in (2, 4)
    r.check("init.csv has n rows", ok_shape, f"shape={init.shape}")
    if ok_shape:
        s0 = snaps[0]
        dpos = max(np.abs(s0["x"] - init[:, 0]).max(), np.abs(s0["y"] - init[:, 1]).max())
        r.check("first frame matches init.csv positions", dpos <= 1e-3 * dx,
                f"max |dpos|={dpos:.3g} (tol {1e-3*dx:.3g})")
        v_init = init[:, 2:4] if init.shape[1] == 4 else np.zeros((n, 2))   # 2-column init = at rest
        dv = max(np.abs(s0["vx"] - v_init[:, 0]).max(), np.abs(s0["vy"] - v_init[:, 1]).max())
        vtol = 1e-3 * max(np.hypot(v_init[:, 0], v_init[:, 1]).max(), np.sqrt(9.81 * dx))
        r.check("first frame matches init.csv velocities", dv <= vtol, f"max |dv|={dv:.3g} (tol {vtol:.3g})")
    return True


def step_checks(r, P, d, snaps, cfl):
    """Time-step validity. Needs the solver's own per-step log (dt, h, c0, amax, vmax)."""
    t, dt, h, c0, amax, vmax = (d[c] for c in STEP_COLUMNS[1:])
    ok = (np.isfinite(np.column_stack([d[c] for c in STEP_COLUMNS])).all()
          and (dt > 0).all() and (h > 0).all() and (c0 > 0).all())
    r.check("steps log finite with dt,h,c0 > 0", ok)
    if not ok:
        return
    r.check("step indices contiguous 0..N-1", np.array_equal(d["step"], np.arange(len(d))), f"N={len(d)}")
    if len(d) > 1:
        gap = np.abs(t[1:] - t[:-1] - dt[:-1]) / dt[:-1]
        r.check("time advances by dt each step", (gap <= 1e-2).all(), f"max |dt_gap|/dt={gap.max():.2g}")
    r.check("steps log covers [0, t_end]",
            t[0] <= 1e-2 * dt[0] and t[-1] + dt[-1] >= P["t_end"] * (1 - 1e-3),
            f"t_first={t[0]:.4g} t_last+dt={t[-1] + dt[-1]:.4g} t_end={P['t_end']:.4g}")

    # Cross-check: a solver that under-reports vmax could pass CFL trivially.
    vs = max(np.hypot(s["vx"], s["vy"]).max() for s in snaps)
    r.check("logged vmax consistent with snapshots", vs <= 1.05 * vmax.max() + 1e-12,
            f"snapshot vmax={vs:.4g} logged max={vmax.max():.4g}")

    lim = cfl * (1 + 1e-6)
    ac = dt / (h / (c0 + vmax))                                            # acoustic/advective
    fr = np.where(amax > 0, dt / np.sqrt(h / np.maximum(amax, 1e-300)), 0)  # force
    r.check("acoustic CFL: dt <= C*h/(c0+vmax)", ac.max() <= lim, f"max ratio={ac.max():.3g} (C={cfl})")
    r.check("force CFL: dt <= C*sqrt(h/amax)", fr.max() <= lim, f"max ratio={fr.max():.3g} (C={cfl})")

    mach = (vmax / c0).max()
    r.check("Mach vmax/c0 <= 0.1 (density error ~ Mach^2 <= 1%)", mach <= 0.1, f"max={mach:.3g}", fatal=False)
    eff = np.median(np.maximum(ac, fr))
    r.check("dt not overly conservative (median ratio >= 0.05)", eff >= 0.05,
            f"median ratio={eff:.3g}; dt min/median/max={dt.min():.3g}/{np.median(dt):.3g}/{dt.max():.3g}",
            fatal=False)


def momentum(P, ts, snaps, r, tol):
    """Linear momentum of a free fluid must be conserved: internal pair forces cancel."""
    m, dx = P["mass"], P["dx"]
    x0, y0, x1, y1 = P["domain"]
    r.check("scene has gravity off", P["g"] == 0, f"g={P['g']}")

    mg = WALL_MARGIN * dx
    clear = all(((s["x"] > x0 + mg) & (s["x"] < x1 - mg) & (s["y"] > y0 + mg) & (s["y"] < y1 - mg)).all()
                for s in snaps)
    r.check("fluid never reaches a wall", clear,
            "" if clear else "walls exchange momentum, so a wall hit makes this test invalid (or the fluid blew up)")

    Pm = m * np.array([[s["vx"].sum(), s["vy"].sum()] for s in snaps])
    s0 = snaps[0]
    scale = m * np.hypot(s0["vx"], s0["vy"]).sum()        # sum |m v_i|: upper bound on |P|
    drift = np.hypot(*(Pm - Pm[0]).T) / scale
    r.check("total momentum conserved", drift.max() <= tol,
            f"max |dP|/sum|mv|={drift.max():.3e} (tol {tol:.0e}) P0=({Pm[0,0]:.4g},{Pm[0,1]:.4g})")

    # A solver that never evolves the flow would trivially conserve momentum.
    v0 = np.hypot(s0["vx"], s0["vy"]).max()
    dv = np.hypot(snaps[-1]["vx"] - s0["vx"], snaps[-1]["vy"] - s0["vy"]).max()
    r.check("flow actually evolved (test is non-vacuous)", dv >= 0.05 * v0,
            f"max |v(t_end)-v(0)|={dv:.3g} vs v0max={v0:.3g}")


def energy(P, s):
    m, g = P["mass"], P["g"]
    return np.sum(0.5 * m * (s["vx"] ** 2 + s["vy"] ** 2) + m * g * s["y"])


def hydrostatic(P, ts, snaps, r, dens_tol=0.02, vel_tol=0.02, slope_tol=0.10):
    g, rho0, Hf, dx = P["g"], P["rho0"], P["fill_height"], P["dx"]
    W = P["domain"][2]
    last = snaps[-1]
    vmax = np.max(np.hypot(last["vx"], last["vy"]))
    ref_v = np.sqrt(g * Hf)
    r.check("fluid at rest at t_end", vmax < vel_tol * ref_v, f"vmax={vmax:.3g} limit={vel_tol*ref_v:.3g}")

    # interior = away from free surface, floor and side walls (kernel truncation biases all of them)
    inner = ((last["y"] > 3 * dx) & (last["y"] < Hf - 3 * dx) &
             (last["x"] > 3 * dx) & (last["x"] < W - 3 * dx))
    if inner.sum() < 10:
        r.check("enough interior particles", False, f"only {inner.sum()}; use a smaller --dx")
    else:
        dev = np.abs(last["rho"][inner] / rho0 - 1)
        r.check("interior density ~ rho0", dev.mean() < dens_tol,
                f"mean dev={dev.mean():.3%} (tol {dens_tol:.0%}; depends on EOS stiffness)")

        # Only meaningful if the solver's EOS gives physical pressure. p=k(rho-rho0) does not
        # unless k is tuned, so this is advisory.
        depth = Hf - last["y"][inner]
        slope = np.polyfit(depth, last["p"][inner], 1)[0]
        r.check("pressure gradient ~ rho0*g", abs(slope / (rho0 * g) - 1) < slope_tol,
                f"slope/rho0g={slope/(rho0*g):.3f}", fatal=False)

    top = np.percentile(last["y"], 99)
    r.check("free-surface height drift < 2dx", abs(top - Hf) < 2 * dx, f"top={top:.4f} vs {Hf:.4f}")

    ke = [0.5 * P["mass"] * np.sum(s["vx"] ** 2 + s["vy"] ** 2) for s in snaps]
    r.check("KE decays (late <= 10% of peak)", np.mean(ke[-5:]) <= 0.1 * max(ke), f"peak={max(ke):.3g}")


def front_and_height(P, s):
    dx = P["dx"]
    bottom = s["y"] < FRONT_LAYER * dx
    # +dx/2: particle centre -> fluid edge, so Z(0) = 1 exactly
    front = s["x"][bottom].max() + dx / 2 if bottom.any() else np.nan
    left = s["x"] < LEFT_STRIP * dx
    height = s["y"][left].max() + dx / 2 if left.any() else np.nan
    return front, height


def dambreak(P, ts, snaps, r, ref=None, rmse_tol=0.10, e_tol=0.02):
    g, L, dx = P["g"], P["column_width"], P["dx"]
    m = np.array([front_and_height(P, s) for s in snaps])
    front, height = m[:, 0], m[:, 1]

    # Normalisation differs between papers -- verify against the one your reference uses.
    # Martin & Moyce: T = t*sqrt(g/a).  Koshizuka & Oka: T = t*sqrt(2g/L).
    T = ts * np.sqrt(2 * g / L)
    Z = front / L

    r.check("front tracked in every frame", np.isfinite(front).all())
    hit = front > P["domain"][2] - 2 * dx
    upto = int(np.argmax(hit)) if hit.any() else len(front)
    seg = Z[:upto]
    if len(seg) < 3 or not np.isfinite(seg).all():
        r.check("front monotone before wall impact", False, f"only {len(seg)} usable pre-impact frames")
    else:
        dz = np.diff(seg)
        r.check("front monotone before wall impact", (dz > -0.02).all(), f"min dZ={dz.min():.3g}")

    E = np.array([energy(P, s) for s in snaps])
    r.check("energy does not grow", E.max() <= E[0] * (1 + e_tol),
            f"E0={E[0]:.4g} Emax={E.max():.4g}  (KE+PE only; elastic EOS energy not counted)")
    h_mid = height[len(height) // 3]
    r.check("column height decreases", np.isfinite(h_mid) and h_mid < height[0],
            f"h0={height[0]:.3f} h(t/3)={h_mid:.3f}")

    if ref is not None:
        Tr, Zr = ref
        inside = (Tr >= T[0]) & (Tr <= T[-1])   # never extrapolate the simulation
        if inside.sum() < 3:
            r.check("front position vs reference", False,
                    f"only {inside.sum()}/{len(Tr)} reference points fall in simulated T range "
                    f"[{T[0]:.2f},{T[-1]:.2f}]")
        else:
            Zi = np.interp(Tr[inside], T, Z)
            rmse = np.sqrt(np.mean((Zi - Zr[inside]) ** 2)) / Zr[inside].max()
            r.check("front position vs reference", rmse < rmse_tol,
                    f"nRMSE={rmse:.3%} (tol {rmse_tol:.0%}) over {inside.sum()}/{len(Tr)} ref points")
    return T, Z, height


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", choices=["hydrostatic", "dambreak", "momentum"])
    ap.add_argument("params")
    ap.add_argument("output")
    ap.add_argument("--ref")
    ap.add_argument("--allow-no-ref", action="store_true")
    ap.add_argument("--init")
    ap.add_argument("--steps")
    ap.add_argument("--allow-no-steps", action="store_true")
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--mom-tol", type=float, default=1e-4)
    a = ap.parse_args()

    P = json.load(open(a.params))
    r = Report()

    try:
        ts, snaps = load(a.output)
    except BadFile as e:
        r.check("solver output well-formed", False, str(e))
        sys.exit(1)

    init_path = a.init or os.path.splitext(a.params)[0] + ".init.csv"
    try:
        init = np.loadtxt(init_path, delimiter=",", skiprows=1)
    except OSError:
        r.check("init.csv found", False, init_path)
        init = None

    if not integrity(r, P, ts, snaps, init):
        print(f"\n{r.fails}/{r.total} checks failed; data too malformed for physics checks")
        sys.exit(1)

    steps_path = a.steps or os.path.splitext(a.output)[0] + ".steps.csv"
    if not os.path.exists(steps_path):
        if a.allow_no_steps:
            print("[SKIP] time-step/CFL checks  (--allow-no-steps)")
        else:
            r.check("steps log provided", False,
                    f"{steps_path} not found; solver must log per-step dt (or pass --allow-no-steps)")
    else:
        try:
            step_checks(r, P, load_steps(steps_path), snaps, a.cfl)
        except BadFile as e:
            r.check("steps log readable", False, str(e))

    if a.scene == "hydrostatic":
        hydrostatic(P, ts, snaps, r)
    elif a.scene == "momentum":
        momentum(P, ts, snaps, r, a.mom_tol)
    else:
        ref = None
        if a.ref:
            try:
                ref = load_ref(a.ref)
            except BadFile as e:
                r.check("reference file readable", False, str(e))
        elif a.allow_no_ref:
            print("[SKIP] front position vs reference  (--allow-no-ref)")
        else:
            r.check("reference provided", False,
                    "quantitative validation needs --ref (or pass --allow-no-ref to skip)")
        T, Z, h = dambreak(P, ts, snaps, r, ref)
        np.savetxt(a.output.replace(".csv", ".front.csv"), np.column_stack([T, Z, h]),
                   delimiter=",", header="T,Z,h", comments="")

    print(f"\n{r.fails}/{r.total} checks failed")
    sys.exit(1 if r.fails else 0)
