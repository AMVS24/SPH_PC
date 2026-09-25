# SPH validation harness

Solver-agnostic checks for three cases: **hydrostatic tank**, **2D dam break**, and a
**free-fluid momentum test**, plus **time-step (CFL) checks** on every run.
The harness never runs your solver. It generates the initial conditions, you run your solver
on them, and it then checks the solver's output files. The same checks therefore apply to the
sequential, OpenMP and CUDA versions.

```
scenes.py ──► scenes/<scene>.init.csv + <scene>.json ──► [your solver] ──► out/<scene>.csv        ──► run_checks.py
                                                                          └► out/<scene>.steps.csv ──┘
```

## Files

| File | Role |
|---|---|
| `scenes.py` | Generates initial particle positions and scene parameters |
| `run_checks.py` | Loads solver output, runs checks, prints PASS/FAIL/INFO, exits 1 on any fatal failure |
| `reference/` | Digitised experimental curves (`T,Z` CSV). Empty — fill from the papers |

Requires Python 3 and NumPy.

## Usage

```bash
# 1. generate a scene
python3 scenes.py hydrostatic --dx 0.005 --out scenes
python3 scenes.py dambreak    --dx 0.005 --out scenes
python3 scenes.py momentum    --dx 0.01  --out scenes

# 2. run YOUR solver on scenes/<scene>.init.csv + scenes/<scene>.json,
#    writing out/<scene>.csv AND out/<scene>.steps.csv in the formats below

# 3. check
python3 run_checks.py hydrostatic scenes/hydrostatic.json out/hydrostatic.csv
python3 run_checks.py dambreak    scenes/dambreak.json    out/dambreak.csv \
        --ref reference/koshizuka_oka_front.csv
python3 run_checks.py momentum    scenes/momentum.json    out/momentum.csv
```

| Flag | Effect |
|---|---|
| `--ref FILE` | Reference front curve (`T,Z`). **Required for dam break.** |
| `--allow-no-ref` | Dam break only: skip the reference comparison (prints SKIP) instead of failing. For development only, not for reported results. |
| `--steps FILE` | Step log. Default: `<output without .csv>.steps.csv`. |
| `--allow-no-steps` | Skip the time-step/CFL checks (prints SKIP) instead of failing when the log is missing. For development only. |
| `--cfl C` | CFL coefficient limit for both dt criteria. Default 0.4. |
| `--mom-tol X` | Momentum-drift tolerance (relative). Default 1e-4. Tighten to about 1e-10 once the solver writes full-precision output. |
| `--init FILE` | Initial-condition CSV. Default: the params path with `.json` replaced by `.init.csv`. |

Exit code: 0 if every fatal check passed, 1 otherwise, including a malformed output file.
`[INFO]` lines are advisory and never affect the exit code.
The dam-break run also writes `out/dambreak.front.csv` (columns `T,Z,h`) for plotting.

## Inputs your solver must read

**Use `params["dx"]`, not the `--dx` you requested.** `scenes.py` rounds `dx` so that the
lattice fills the nominal geometry exactly (`L/dx` is an integer). The original request is kept
as `dx_requested`.

`scenes/<scene>.init.csv`: header `x,y,vx,vy`, one row per particle. The row order defines
particle `id` (row 0 is id 0). Velocities are zero for hydrostatic and dam break.

`scenes/<scene>.json` fields:

| Field | Meaning |
|---|---|
| `n` | particle count |
| `dx` | actual initial particle spacing (m) |
| `mass` | per-particle mass = `rho0 * dx²` (2D, so areal density) |
| `rho0` | rest density (1000) |
| `g` | gravity magnitude (9.81), acting in −y. **0 for the momentum scene: apply no gravity.** |
| `domain` | `[xmin, ymin, xmax, ymax]` tank walls |
| `t_end` | simulation end time (s) |
| `snapshot_dt` | snapshot interval (s) |
| `fill_height` | hydrostatic only: water depth |
| `column_width`, `column_height` | dam break only: initial column size |

Hydrostatic and dam break start at rest. The momentum scene starts with two blobs moving
toward each other (see its `init.csv`).

## Output your solver must write

One CSV for the whole run, header required, one row per particle per snapshot:

```
frame,t,id,x,y,vx,vy,rho,p
```

- `frame`: snapshot index 0, 1, 2, ... All rows of a snapshot share the same `frame` and `t`.
- `id`: particle index 0..n-1, same as the row order of `init.csv`.
- Frame 0 must be the initial state at t≈0, and the last frame must reach `t_end`
  (within one `snapshot_dt`).
- Rows may be in any order.

### Step log

A second CSV, one row per time step the solver actually took:

```
step,t,dt,h,c0,amax,vmax
```

| Column | Meaning |
|---|---|
| `step` | 0, 1, 2, ... contiguous |
| `t` | simulation time at the **start** of the step |
| `dt` | the step size used |
| `h` | smoothing length |
| `c0` | numerical speed of sound. For `P = k(ρ − ρ₀)` use `c0 = √k`. |
| `amax` | max acceleration magnitude over all particles that step |
| `vmax` | max speed over all particles that step |

The solver already computes all of these to pick `dt`. Logging them costs almost nothing.

## Checks

**Every scene, structural (fatal):**

- The particle count is constant and the ids are exactly 0..n-1 in every frame.
- There are no NaN or inf values.
- No particle goes outside `domain`.
- Times strictly increase, the first frame is at t=0, and the last frame reaches `t_end`.
- Frame 0 matches `init.csv` positions (tolerance 1e-3·dx) and velocities.

If a structural check fails, the physics checks are skipped.

**Every scene, time step (needs the step log):**

- The log is finite, `dt`, `h` and `c0` are positive, and step indices are contiguous.
- Time advances by `dt` each step (1% tolerance), and the log covers `[0, t_end]`.
- The logged `vmax` is consistent with the snapshots. It cannot be below the fastest snapshot
  speed by more than 5%, so a solver cannot pass CFL by under-reporting velocity.
- **Acoustic CFL:** `dt ≤ C·h/(c0 + vmax)` at every step (C = 0.4).
- **Force CFL:** `dt ≤ C·√(h/amax)` at every step (C = 0.4).
- **INFO only:** Mach `vmax/c0 ≤ 0.1`, since density error scales as Mach², so 0.1 keeps it near 1%.
- **INFO only:** `dt` is not overly conservative (median ratio to the limit ≥ 0.05).

**Hydrostatic:**

- `vmax` at `t_end` is under 2% of √(g·H).
- The interior mean density is within 2% of `rho0`. The interior excludes 3·dx from the free surface, floor and side walls.
- The top surface (99th percentile of `y`) is within 2·dx of `fill_height`.
- Late-time KE is at most 10% of peak KE.
- **INFO only:** the pressure-vs-depth slope is within 10% of `rho0·g`.

**Dam break:**

- The surge front is tracked in every frame. It is the bottom layer, `y < 3·dx`, plus dx/2.
- The front advances monotonically until it reaches the far wall.
- KE+PE never rises more than 2% above its initial value.
- The column height at the left wall drops.
- The normalised front curve matches the reference (nRMSE < 10%). Only reference points inside the simulated time range are compared, and fewer than 3 such points is a FAIL.

**Momentum (free-fluid scene):** two unequal blobs collide off-axis with gravity off.

- The scene has `g == 0`.
- The fluid stays more than 4·dx from every wall. Walls exchange momentum, so a wall hit
  invalidates the test.
- Total momentum `m·Σvᵢ` stays constant: the max relative drift `|ΔP| / Σ|m·vᵢ|` is under `--mom-tol`.
- The flow actually evolved (max velocity change ≥ 5% of the initial max), so a frozen solver
  can't pass by not doing anything.

## Caveats

- **Density and free-surface checks assume a stiff EOS.** With `P = k(ρ − ρ₀)`, density rises with depth by `ρ₀gh/k` and the free surface sags. A correct solver with a soft `k` can fail the density and height checks. That is a finding about `k`, not necessarily a bug.
- **Kernel dimensionality.** `mass = rho0·dx²` assumes 2D-normalised kernels. If the solver keeps 3D constants (poly6 `315/(64πh⁹)`) in 2D, density will not sum to `rho0`. Check this first if hydrostatic density fails.
- **Time normalisation.** It is coded as `T = t·√(2g/L)`, `Z = x_front/L`. This was written from memory of Koshizuka & Oka and is **not verified against the paper**. Martin & Moyce use `T = t·√(g/a)`. Match the normalisation to your reference data.
- **Tolerances are uncalibrated guesses** (2%, 10%, 0.02). Calibrate them against a trusted solver.
- **Energy check** counts KE+PE only, not elastic EOS energy.
- **CFL coefficients.** 0.4 is a generous upper limit. Monaghan-style solvers usually use 0.25, so
  a correct solver should pass with room to spare. `h` and `c0` come from the solver's own log, so
  the harness checks the solver's `dt` against the solver's own stated constants. It does not
  check that `c0` is physically reasonable.
- **Momentum test requires pair-symmetric forces.** Pressure and viscosity must be symmetric
  pair forces. The Müller pressure force is a force *density*, so the acceleration must be
  `f_i/ρ_i`. Dividing by mass instead breaks conservation when densities differ. Global damping,
  artificial drag or a non-symmetric viscosity will also fail, which is the intended result.
- **Momentum tolerance vs precision.** Real double-precision conservation is about 1e-15.
  The 1e-4 default only allows for CSV rounding (`%g` gives about 6 digits). Parallel
  reductions change summation order but stay near 1e-12. A GPU in fp32 needs about 1e-5.
- **Reference data.** `reference/` is deliberately empty. Digitise it from the paper.
- **Units.** Scenes are in SI. If your solver uses arbitrary units, scale `g`, `L` and `dx` consistently.
- **CSV does not scale.** It is meant for validation at modest N (about 10⁴ particles). Do not write snapshots inside timed benchmark regions.

## Tested

The checker was exercised only on synthetic data: valid frozen hydrostatic and synthetic
dam-break runs, and runs with specific bugs injected (truncated output, wrong t=0 state, old
format, missing/bad/out-of-range reference, non-pair drag that breaks momentum, a frozen solver,
`dt` too large, under-reported `vmax`, a step log that stops early or drops steps, a missing
step log, and a high Mach number). It has **not** been run against a real solver.
