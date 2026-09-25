"""Initial conditions for validation scenes.

Writes  <out>/<scene>.init.csv  (x,y,vx,vy per particle)  and  <out>/<scene>.json  (parameters).
The solver under test must read both, run to params["t_end"], and write snapshots
in the format documented in run_checks.py.

The requested --dx is adjusted so the particle count across each dimension is an integer:
the lattice then fills the nominal geometry *exactly* (no silent +/- dx error in the column
width, which the dam-break normalisation Z = x/L depends on). The adjusted spacing is stored
as params["dx"]; the request is kept as params["dx_requested"]. Always use params["dx"].

Scenes: hydrostatic, dambreak, momentum. The momentum scene is two unequal blobs colliding in
free space with gravity OFF (params["g"] == 0) -- it tests pair-force symmetry, so the solver must
not apply gravity and the fluid must never reach a wall.

Usage: python scenes.py hydrostatic|dambreak|momentum --dx 0.005 --out scenes/
"""
import argparse, json, os
import numpy as np

G = 9.81
RHO0 = 1000.0


def _grid(nx, ny, dx):
    # particle centres at half-integer multiples of dx, so nothing sits exactly on a wall
    xs = (np.arange(nx) + 0.5) * dx
    ys = (np.arange(ny) + 0.5) * dx
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack([X.ravel(), Y.ravel()])


def _blob(x0, y0, w, h, dx):
    nx, ny = max(1, round(w / dx)), max(1, round(h / dx))
    return _grid(nx, ny, dx) + np.array([x0, y0])


def hydrostatic(dx_req):
    W, H_fill, H_tank = 0.5, 0.3, 0.6
    nx = max(1, round(W / dx_req))
    dx = W / nx
    ny = max(1, round(H_fill / dx))
    pts = _grid(nx, ny, dx)
    return pts, np.zeros_like(pts), dict(scene="hydrostatic", dx=dx, domain=[0, 0, W, H_tank],
                                   fill_height=ny * dx, t_end=2.0, snapshot_dt=0.02)


def dambreak(dx_req):
    # Koshizuka & Oka (1996) / Martin & Moyce geometry: column L x 2L, tank 4L wide
    L = 0.146
    nx = max(1, round(L / dx_req))
    dx = L / nx
    pts = _grid(nx, 2 * nx, dx)
    return pts, np.zeros_like(pts), dict(scene="dambreak", dx=dx, domain=[0, 0, 4 * L, 4 * L],
                                       column_width=L, column_height=2 * L,
                                       t_end=0.8, snapshot_dt=0.005)


def momentum(dx):
    # Two unequal blobs on an off-axis collision course, no gravity, far from the walls.
    # Total momentum is nonzero (A dominates) and, absent walls/gravity, must be conserved.
    A = _blob(0.15, 0.40, 0.20, 0.20, dx)
    B = _blob(0.55, 0.45, 0.10, 0.10, dx)
    vel = np.vstack([np.tile([0.5, 0.0], (len(A), 1)), np.tile([-0.2, 0.1], (len(B), 1))])
    return np.vstack([A, B]), vel, dict(scene="momentum", dx=dx, g=0.0, domain=[0, 0, 1, 1],
                                        t_end=0.6, snapshot_dt=0.01)


SCENES = dict(hydrostatic=hydrostatic, dambreak=dambreak, momentum=momentum)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", choices=SCENES)
    ap.add_argument("--dx", type=float, default=0.005)
    ap.add_argument("--out", default="scenes")
    a = ap.parse_args()
    pts, vel, params = SCENES[a.scene](a.dx)
    dx = params["dx"]
    params.setdefault("g", G)
    params.update(dx_requested=a.dx, rho0=RHO0, n=len(pts), mass=RHO0 * dx ** 2)  # 2D: areal density
    os.makedirs(a.out, exist_ok=True)
    np.savetxt(f"{a.out}/{a.scene}.init.csv", np.column_stack([pts, vel]), delimiter=",",
               header="x,y,vx,vy", comments="")
    json.dump(params, open(f"{a.out}/{a.scene}.json", "w"), indent=2)
    note = f" (adjusted from {a.dx})" if abs(dx - a.dx) > 1e-12 else ""
    print(f"{a.scene}: {len(pts)} particles, dx={dx:.6g}{note}")
