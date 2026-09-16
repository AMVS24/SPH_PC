# Future Milestones — open items and decisions

Running list of things flagged during M1 that need resolving before or during
later milestones. Nothing here is done yet.

Course context: CS F422 Parallel Computing, research-based project, 25% of grade,
IEEE 2-column report (6–10 pages), public code repo required by M4.

| Milestone | Due | Deliverable |
|---|---|---|
| M1 | 07/09/2026 | Literature review + sequential baseline, target problem metrics |
| M2 | 30/09/2026 | First parallel implementation (OpenMP / naive CUDA), scaling baseline |
| M3 | 30/10/2026 | Advanced optimisation + distributed scaling (tiling, comm hiding, MPI) |
| M4 | 20/11/2026 | Profiling (Nsight/VTune/TAU), benchmarking, final paper + code repo |

---

## 1. The C baseline (deferred from M1, decide before M2)

**Status: deliberately deferred. Revisit before M2 starts.**

The speedup denominator is currently "naive Python". A 1000× speedup over
interpreted CPython measures CPython, not parallelism, and it is the first thing
a reviewer will attack. The course brief says "reference C/Python single-threaded
model", and the sample paper (Kuznik et al. 2010) baselines against an optimised
single-threaded CPU implementation.

Why it matters beyond honesty:

- M2 (OpenMP) becomes nearly free — `#pragma omp parallel for` on the same loops
  as the C baseline, so speedup numbers are apples-to-apples by construction.
- Sets up the CUDA port in M2/M3 without a language change mid-project.
- The Python stack then becomes the *prior exploration / motivation* section
  rather than the baseline. This is a genuinely better narrative: it is how the
  neighbour-search bottleneck was found in the first place.

Decision needed: port the solver to single-threaded C before M2, and re-baseline.

## 2. Physics divergence across implementations (blocking for M2 speedups)

The three existing implementations do **not** solve the same equations, so the
scaling comparison in the old report is not apples-to-apples.

| | gravity | viscosity | pressure force | integrator |
|---|---|---|---|---|
| `sequential_py/naive_py.py` | `(0, −40)` ✅ | Monaghan Π_ij ✅ | `Σ m·p̄/ρⱼ·∇W` | forward Euler (`p` before `v`) |
| `Vectorised/SPH_vectorised.py` | `(0, −40)` | Monaghan Π_ij | symmetric `p/ρ²` form | semi-implicit Euler |
| `Compute Shader/SPH_Taichi.py` | **none** | **none** | **`Σ m·∇W` — no pressure, no ρ** | semi-implicit Euler |

Gravity and viscosity have now been added to `naive_py.py` (ported from the
vectorised version). Still outstanding:

- **Taichi force kernel is not SPH.** It runs `update_densities()`, stores the
  result, and never uses it — the force is a bare kernel-weighted repulsion
  ([SPH_Taichi.py:197](Compute%20Shader/SPH_Taichi.py#L197)). It has no gravity
  and no viscosity either. Some of the reported "5× from 100→10k" is the GPU
  version simply doing less arithmetic per pair.
- **Pressure form differs** between naive (`avg_pressure / ρⱼ`) and vectorised
  (symmetric `pᵢ/ρᵢ² + pⱼ/ρⱼ²`). Both are repulsive and neither is wrong, but
  they are different discretisations.
- **Integrator differs.** Naive advances position with the stale velocity
  (forward Euler); the other two do velocity-first (semi-implicit). Trajectories
  diverge even where forces agree. The old report claims LeapFrog — none of the
  three implement it.

**Action:** write one documented physics spec, make every implementation conform,
and verify by dumping particle state at frame N and diffing across
implementations. Without this, M2/M3 speedup claims are unfalsifiable.

## 3. Give the benchmark a purpose: switch to a canonical case

Right now the scene is particles in a box with random initial positions — there
is nothing to validate against. M4 requires quantitative validation against a
canonical reference, not a qualitative "it looks like water".

**Recommended: 2D dam-break.** It is the standard SPH validation case, it is what
DualSPHysics validates on (Crespo et al., PLoS ONE 2011), and it fixes four
things at once:

- deterministic initial condition → no RNG, no run-to-run benchmark noise, and
  the settle-detection in `baseline_testing.py` becomes unnecessary
- gravity becomes physically necessary rather than an optional add-on
- "steady state" gets a real definition
- gives the validation metric M4 needs: **surge-front position vs normalised
  time**, comparable against Koshizuka & Oka experimental data

Alternative if a steady-state rather than transient case is preferred:
lid-driven cavity.

## 4. Metrics (partly done)

- ✅ ms per timestep and particle-updates/sec are now the primary metrics in
  `baseline_testing.py`; render time is recorded separately and excluded.
- ✅ RNG seeded (`random.seed(42)` in `naive_py.py`).
- ⬜ Add **speedup** and **parallel efficiency** columns once M2 lands.
- ⬜ Decide whether FPS stays in the paper at all — it is a graphics metric, not
  a solver metric. Probably keep it only for the real-time-feasibility argument.

## 5. Sweep range

Current default sweep is 10→2000 (24 points: 10–100 by 10, 200–1000 by 100,
1200–2000 by 200). The interesting regime — and the crossover where spatial
hashing overtakes brute force — is 10²–10⁴. The naive solver is O(N²) so the
tail gets expensive fast; the harness has wall-clock budgets to keep it
tractable and flags any N that never reached steady state. Extend the range
once the C baseline exists and the tail is affordable.

**Measured 2026-09-07:** only N ≤ 100 reach steady state. Everything above is
warmup-budget-truncated — N=2000 got 2 warmup frames and 10 measured frames.
Mitigating evidence that this does not distort the timing: the fitted constant
k = t/N² is flat at 7.7–9.3e-03 ms across the whole unsettled range, i.e. the
unsettled points sit on the same O(N²) curve as the settled ones. Supports
reporting the tail as a cost model rather than as fluid steady state.

Separate observation worth chasing: k rises from ~6.5e-03 at N ≤ 100 to
~8.5e-03 at N ≥ 300, a ~30% degradation. Not a settling artifact (it goes the
wrong way for fixed-overhead effects). Most likely cache behaviour as the
particle working set outgrows L2. Relevant to the M2/M3 memory-tiling argument.

## 6. Repo hygiene (needed by M4)

- ⬜ `git init` — the project is not currently under version control, and M4
  requires a public code repository.
- ⬜ Pin the environment (`environment.yml` exists; add exact versions before the
  final benchmark run so results are reproducible).
- ⬜ Record hardware for every benchmark table (the old report's numbers are from
  an i7 + GTX 1650 Ti laptop; the new work targets an RTX 5060 Ti / 5070 Ti).

## 7. Research angle — unresolved

Verify-then-propose. Two proposals have already dissolved on contact with prior
work, so nothing goes in the report without a literature check first.

**Dead — do not re-propose.** RT-core accelerated neighbour search. The space is
essentially closed:

- Zhu, *RTNN* (PPoPP 2022, arXiv:2201.01366) — neighbour search as ray tracing,
  query scheduling/partitioning, 2.2×–65× over existing GPU libraries
- Zhao et al. (IJNME 124:696, 2023) — general RT-based neighbour search vs
  cell-based, explicitly covering SPH/DEM/MD/peridynamics
- Meneses et al. (Future Gen. Computer Systems 183:108555, Oct 2026;
  arXiv:2601.15633) — BVH update/rebuild ratio optimiser, neighbour-list-free RT
  variants, periodic BCs, energy analysis, cross-GPU scaling, non-uniform radii
- Mochi (proxy spheres for non-uniform radii); public StarsX/RayTracedSPH and an
  AMD HIP RT SPH writeup

**Threads still worth a literature search (unverified):**

1. Multi-GPU / distributed RT-core neighbour search — all of the above is
   single-GPU. Would also satisfy the M3 MPI requirement.
2. Concurrent CPU/GPU execution — the "Future Work" section of the old report:
   overlap host-side hash-key computation and neighbour-index preparation for
   frame *n+1* with device-side density/force evaluation for frame *n*.
3. Whether the **neighbour-construction-dominates-force-evaluation** tension is
   already characterised in recent GPU SPH work. This is the seed observation
   from the old profiling data and the strongest paper opening so far:

   | particles | spatial hashing | total | share |
   |---|---|---|---|
   | 100 | 13.71 ms | 14.98 ms | 92% |
   | 1k | 24.62 ms | 25.99 ms | 95% |
   | 10k | 58.67 ms | 62.99 ms | 93% |
   | 100k | 107.31 ms | 206.30 ms | 52% |

   The classical GPU SPH literature (Crespo et al. 2011) implies force
   evaluation dominates. This data says neighbour construction does. Caveat
   before leaning on it: the Taichi force kernel is doing less work than a real
   SPH force kernel (see §2), so the ratio is likely overstated. **Re-profile
   after the physics spec is unified.**

4. Also open: self-consistent adaptive smoothing length (hᵢ solved iteratively
   against a target neighbour count, so the radius changes *within* a timestep,
   as opposed to a static non-uniform radius distribution). Narrow and contested.

**Constraints any angle must satisfy:** (a) distributed/MPI component for M3,
(b) learnable from zero CUDA in a semester, (c) builds on the existing solver and
profiling data rather than discarding it, (d) quantitative validation against a
canonical reference case.

**Target venues:** IPDPS/SC-style workshop, HiPC Student Research Symposium, or
Computer Physics Communications. Scale of contribution: roughly Kuznik 2010 —
single-GPU, one clean secondary claim beyond raw speedup, validated against a
canonical reference solution.

## 8. Open logistics

- Multi-node / cluster access **not yet confirmed** — this is the main risk to
  the M3 distributed requirement. Needs an answer early.
- Group size: max 3 (per the user's brief) vs max 4 (per the intro slides) —
  confirm with the instructor.
