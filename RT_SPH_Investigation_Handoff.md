# RT cores for SPH neighbour search: investigation handoff

Purpose of this file: give a paper-writing agent everything that was measured, inferred and concluded, with the exact numbers, where the data lives, and which claims are confirmed versus inferred. All work was done on 2026-11-20.

---

## 1. One-paragraph summary

At SPH neighbour counts (about 50-100 neighbours per particle, complete neighbour sets, single precision), RT-core fixed-radius neighbour search via OptiX is 3-4x slower than a textbook sort-based uniform GPU grid. This held on two GPU generations (Ada, Blackwell), for count, SPH density and SPH pressure+viscosity force workloads, with or without BVH refit. It also held with every RT optimisation from the literature applied. For adaptive-resolution SPH (h tied to local density), a multi-level grid beats RT by 1.3-2.0x, and RT falls further behind as the h range grows. The reason is a per-candidate cost floor. Every candidate neighbour costs one programmable shader invocation, which we measured at about 8x the cost of a grid distance test. The BVH's tighter candidate set (2.2 calls per neighbour versus the grid's 6.4 tests) does not make up for that.

Published RT wins (RTNN, ORCS, DEM work) were reproduced with the same code. They arise from comparison conditions SPH does not share:

- **Double-precision grid baselines.** This is the dominant factor. RT hardware is single-precision only, and FP64 runs at 1/64 rate on GeForce and RTX-workstation parts.
- **Host-transfer asymmetries** in library baselines.
- **K-capped (truncated) RT searches.**
- **Grids sized by the largest radius,** or dense cell arrays over mostly empty domains.
- **Very sparse queries** of about 1-4 neighbours.

---

## 2. Context

- Project: GPU SPH solver (CUDA). Previous milestones built a sequential C baseline, OpenMP, a CUDA uniform grid (FP32 and FP64), and an OptiX RT-core SPH solver.
- Prior state (before this investigation):
  - The OptiX SPH solver was about 2x slower than the CUDA grid solver for a full step at 1M particles: 17.29 ms vs 8.53 ms on the laptop GPU (`SPH_PC/rtx/bench/data/compare_results.json`).
  - A neighbour-search-only benchmark on ORCS scenarios (box 1000, RTX 5000 Ada) had RT winning at r = 1 and with log-normal radii, and losing about 3.5-4x at SPH-like radius (`SPH_PC/rtx/frnn/data/frnn_box1000.json`, figure `our_speedup.png`).
  - A multi-resolution proof of concept claimed RT up to 31-40x faster at h ratio 8 (`SPH_PC/rtx/frnn/data/multires_poc.txt`). This investigation shows that baseline was unfair (Section 6).

## 3. Hardware and software

| Item | Laptop | Workstation (remote, shared) |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop (Blackwell) | NVIDIA RTX 5000 Ada Generation |
| SMs / compute capability | 46 / 12.0 | 100 / 8.9 |
| Memory | 12 GB, 192-bit | 32 GB, 256-bit |
| Driver | 621.24 (Windows 11) | 595.84 (Ubuntu 24.04) |
| CUDA | 12.8 | 13.2 for our code (12.8 also installed) |
| OptiX headers | 9.1 (`OPTIX_VERSION 90100`) | same |
| Power limit / clocks | AC power; clocks 2.66-2.69 GHz during runs; max 81 C | 250 W; clocks locked at 2550 MHz |
| Notes | Exclusive use | Shared with other users' jobs (about 15 GB resident, about 10-12% SM use intermittently). This causes occasional outliers, so the minimum over launches is reported. |

For cross-paper comparison: ORCS used an RTX PRO 6000 Blackwell Server Edition (188 SMs).

An early laptop sweep ran on battery with the GPU capped at 33 W. Those files (`cost_local_5070tiL_sweep_20261120.txt`, `cost_local_5070tiL_orcs_20261120.txt`, `cost_local_5070tiL_sweep2_20261120.txt`) are kept but excluded.

## 4. Methods

### 4.1 Neighbour-search definitions
- Fixed-radius search in 3D, FP32 positions. A neighbour is `|x_i - x_j| < r_j` (scatter form; for constant radius this is the usual symmetric FRNN). The particle itself is counted.
- Workload modes:
  - **count:** number of neighbours.
  - **density:** cubic-spline `sum W(r_ij, h_j)` with `h = r/2`.
  - **force:** SPH symmetric pressure term `(p_i/rho_i^2 + p_j/rho_j^2) grad W` plus Monaghan-style artificial viscosity, reading neighbour velocity, density and pressure.
  - **LJ:** Lennard-Jones force per pair, used only for the ORCS comparison.

### 4.2 Grid baseline
- Sort-based uniform grid (Green 2010 style). A CUB radix sort of cell ids, then a binary-search cell-start array; cell size = r_max; 27-cell stencil.
- Particle data is Morton-ordered once at start; the RT path also uses Morton order.
- Variants:
  - cells of r/2 (125-cell stencil);
  - **compact grid:** no dense cell array; binary search over sorted keys per stencil row;
  - **FP64 distance arithmetic;**
  - **LJ in FP32 or FP64;**
  - **multi-level grid** (`frnn_ml.cu`): one grid per power-of-two h level, each bounded to its level's bounding box, positions re-sorted per build. Its build includes one small device-to-host copy, so it is not optimised.

### 4.3 RT implementation (OptiX 9.1)
- **Base design:** follows RTNN and ORCS.
  - Custom AABB primitive of half-width r around each particle.
  - Zero-length ray from each query (tmin 0, tmax 1e-16, direction +x).
  - Intersection program does the distance test and accumulates in payload registers.
  - Any-hit and closest-hit disabled.
  - BVH `PREFER_FAST_BUILD`, rebuilt every call.
  - Queries in Morton order.
- **RT variants (ablations):**

| Variant | Change |
|---|---|
| `base` | as above |
| `refit` | BVH built with `ALLOW_UPDATE`, then refit (`OPERATION_UPDATE`) on each later call. Frozen particles, so this is the best case for refit quality. |
| `fasttrace` | `PREFER_FAST_TRACE` BVH |
| `report` | intersection program reports a hit, any-hit accumulates (the path our SPH solver uses) |
| `unsorted` | queries in random order (tests RTNN's coherence finding) |
| `empty` | intersection program increments a counter only: no memory reads, no math (invocation and traversal floor) |
| `tiny` | AABBs shrunk to 1e-4 r (per-ray traversal floor) |
| `aabbseg` | custom AABBs, but the query is a segment of length 2 r_max ending at the query point |
| `spherefront` | Blackwell hardware sphere primitive (`OPTIX_BUILD_INPUT_TYPE_SPHERES`, built-in intersection); same segment as `aabbseg`; any-hit tests containment |
| `kcap` | RTNN-style early termination: ray terminates after K neighbours (K = 50) |

- RT timing = BVH build (or refit) + trace. The Morton reorder ("sched") is excluded from RT time, which favours RT. Grid timing = sort + cell-start + pass.

### 4.4 Correctness and timing protocol
- Every timed configuration is checked per query against the grid.
  - Count: at most 1 mismatched count, and at most n/1000 queries mismatched.
  - Density and force: relative error ≤ 1e-3 for 99.9% of particles.
  - Observed: worst density error about 1e-6, worst force error about 1e-5; count mismatches 0-2 (float rounding at the radius).
  - `empty`, `tiny` and `kcap` are deliberately not exact; they report counts only.
- An instrumented build counts intersection and any-hit invocations and grid candidate tests.
- Timing: CUDA events, 1 s warm-up, then 5 reps per process, 3 independent process launches per config. Tables report the **minimum over launches** (interference only adds time). Laptop CV < 3%. Clocks and power logged at 1 Hz.
- N = 1M unless stated. Scenes:
  - disordered (uniform random in a cube of side 103);
  - lattice;
  - a shared point file (`pts_disordered_1M.txt`, the same file for every code);
  - **profile:** centrally condensed power-law density, `n ∝ (r²+c²)^(-α/2)`, with h chosen per particle to hit about 80 neighbours, so the h range is set by the core radius;
  - **surface:** LiDAR-like points on 8 planar discs and 4 spherical shells;
  - **random radii:** uniform or log-normal radii independent of position (ORCS-style and DEM-style).
- Neighbour count is set via radius: k = 80 at 1M in box 103 means r = 2.754.

## 5. Claim 1: uniform h at SPH density, RT is 3-4x slower

### 5.1 Ablation matrix at k = 80, disordered, 1M
Grid = full grid time in ms; RT = build + trace in ms; ratio = RT / grid. Minimum over 3 launches.

| Mode, variant | Laptop: grid / RT / trace / ratio / trace-only ratio | Ada: grid / RT / trace / ratio / trace-only ratio |
|---|---|---|
| count base | 3.26 / 11.85 / 9.85 / 3.64 / 3.02 | 1.71 / 6.17 / 4.85 / 3.60 / 2.83 |
| count refit | 3.26 / 10.24 / 9.96 / 3.14 / 3.06 | 1.75 / 5.18 / 4.96 / 2.96 / 2.83 |
| count fasttrace | 3.28 / 11.67 / 9.23 / 3.56 / 2.82 | 1.74 / 6.17 / 4.54 / 3.55 / 2.62 |
| count report (any-hit path) | 3.27 / 14.47 / 12.46 / 4.42 / 3.81 | 1.73 / 7.37 / 6.05 / 4.27 / 3.51 |
| count unsorted queries | 3.37 / 17.74 / 15.68 / 5.27 / 4.65 | 1.77 / 11.31 / 9.96 / 6.40 / 5.64 |
| count empty intersection program | 3.26 / 8.80 / 6.78 / 2.70 / 2.08 | 1.75 / 4.67 / 3.35 / 2.67 / 1.91 |
| count aabbseg | 3.31 / 21.64 / 19.60 / 6.54 / 5.93 | 1.72 / 11.03 / 9.71 / 6.40 / 5.63 |
| count spherefront (hardware spheres) | 3.23 / 25.71 / 22.96 / 7.95 / 7.10 | 1.74 / 17.92 / 16.53 / 10.27 / 9.48 |
| density base | 3.38 / 12.74 / 10.73 / 3.77 / 3.17 | 1.87 / 6.70 / 5.36 / 3.58 / 2.87 |
| density spherefront | 3.39 / 25.79 / 23.02 / 7.60 / 6.79 | 1.87 / 18.65 / 17.26 / 9.99 / 9.25 |
| force base | 5.68 / 17.63 / 15.63 / 3.11 / 2.75 | 3.08 / 9.06 / 7.70 / 2.94 / 2.50 |
| force refit | 5.72 / 16.25 / 15.97 / 2.84 / 2.79 | 3.11 / 8.00 / 7.77 / 2.57 / 2.50 |
| force report | 5.71 / 21.22 / 19.20 / 3.71 / 3.36 | 3.09 / 10.91 / 9.56 / 3.53 / 3.09 |
| force unsorted | 5.99 / 33.35 / 31.26 / 5.57 / 5.22 | 3.13 / 15.71 / 14.34 / 5.02 / 4.58 |
| force spherefront | 5.70 / 26.58 / 23.82 / 4.67 / 4.18 | 3.09 / 22.01 / 20.60 / 7.13 / 6.68 |
| lattice count base | 2.94 / 9.49 / 7.70 / 3.23 / 2.62 | 1.49 / 5.06 / 3.86 / 3.39 / 2.59 |
| lattice force base | 5.32 / 14.70 / 12.89 / 2.76 / 2.43 | 2.71 / 7.41 / 6.21 / 2.74 / 2.29 |

- **Grid with cells of r/2:** laptop count 4.26 ms vs 3.36 ms for cells of r; Ada 2.60 vs 1.74. The baseline grid uses its best configuration.
- **BVH build vs refit:** build about 2.0 ms (laptop) / 1.35 ms (Ada); refit about 0.28 / 0.23 ms.
- **Instrumented counts at k = 80** (identical on both GPUs):

| Quantity | Per query |
|---|---|
| True neighbours | 78.6 |
| RT intersection calls | 180.8 (2.30 per neighbour; geometric ideal for a cube around a sphere is 1.91) |
| `fasttrace` calls | 163.0 |
| `aabbseg` calls | 341.5 |
| `spherefront` any-hit calls | 116.6 (of which 38.0 failed containment) |
| Grid distance tests | 505.7 (6.43 per neighbour) |
| Grid tests, cells r/2 | 294.7 |

- **Amortised cost per candidate:**

| Quantity | Laptop | Ada |
|---|---|---|
| Intersection call (base) | about 0.054 ns | about 0.027 ns |
| Intersection call (empty) | 0.0375 ns | about 0.0185 ns |
| Grid distance test | about 0.0066 ns | about 0.0034 ns |
| Call / test cost ratio | about 8.2 (empty: about 5.6) | about 8.0 (empty: about 5.5) |

  Hardware-sphere any-hit: 0.20 ns (laptop), 0.14 ns (Ada).

### 5.2 Neighbour-count sweep (count mode, disordered, 1M)
RT / grid time ratio, laptop / Ada:

| k | 1 | 4 | 16 | 32 | 80 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|---|
| base (rebuild) | 3.19 / 3.46 | 3.80 / 4.10 | 3.36 / 3.76 | 3.43 / 3.76 | 3.64 / 3.60 | 3.64 / 3.31 | 4.15 / 3.80 | 5.16 / 5.12 |
| trace only | 0.65 / 0.56 | 1.25 / 1.12 | 1.87 / 1.82 | 2.34 / 2.33 | 3.02 / 2.83 | 3.22 / 2.82 | 3.88 / 3.48 | 4.99 / 4.89 |
| refit | 1.02 / 1.16 | 1.65 / 1.68 | 2.11 / 2.17 | - | 3.14 / 2.96 | - | 3.94 / 3.57 | - |

- No crossover at any k. RT's trace alone beats the grid only at k ≈ 1 (trace-only crossover k ≈ 2-3). With refit, RT reaches parity only at k ≈ 1.
- Density mode is the same pattern (3.0-5.3x).
- Hardware spheres: 4.1-13x slower than the grid on the laptop, rising with k; Ada up to 23x at k = 512.

### 5.3 Cost model (direction 2)
- Model: `T_rt = c_sched N + c_bvh N + c_ray N + c_is N_is` and `T_grid = c_sort N + c_cells cells + c_q N + c_test N_tests`. Fitted on k = 1, 16, 64, 256 at N = 200k and 1M; held out k = 4, 32, 80, 128, 512.
- Laptop fit:
  - Constants: BVH 1.93 ns/particle; RT 0.35 ns/query + 0.055 ns per intersection call; grid 0.56 ns/query + 0.0046 ns per test.
  - Held-out error: RT 3.5% mean (15% max); grid 12.8% mean (29.8% max). The grid's missing term is that cost per test varies with cell occupancy.
- Ada fit:
  - Constants: BVH 1.22 ns/particle; RT 0.148 ns/query + 0.0266 ns per intersection call; grid 0.317 ns/query + 0.0022 ns per test.
  - Calls per query = 4.24 + 2.2 k; tests per query = 10.2 + 6.3 k.
  - Held-out error: RT 6.9% and grid 6.6% mean. One RT point (k = 80, 1M) was contaminated by contention (58.6%).
- Winner predicted correctly at 18/18 sweep points.
- **No crossover predicted for 1 ≤ k ≤ 2000.** At k ≈ 1, RT breaks even only if the grid has more than about 60-84 cells per particle (a dense-array pathology).
- On 38 ORCS-style scenarios (box 1000, out of sample), the winner was predicted correctly in 37/38 on both GPUs. The miss was cluster with log-normal radii at 1M. Magnitude errors there are about 20-30% mean (up to 200% for dense clusters). Two terms are missing: cost per intersection call depends on BVH quality (heterogeneous boxes), and grid cost per query depends on the size of the cell array.
- Data: `cost_*` files; fitter `fit_cost.py`; constants `cost_model_5070tiL_20261120.json`.

### 5.4 Best case for RT (an estimate, not a measurement)
Assume a perfect BVH (one invocation per true neighbour, 78.6 instead of 180.8 calls) and a zero-work intersection program. Scaling the measured empty-program trace gives:

- laptop: 6.78 × 78.6/180.8 ≈ 2.95 ms vs grid 3.26 ms;
- Ada: 3.35 × 0.435 ≈ 1.46 ms vs grid 1.71 ms.

Adding refit (about 0.23-0.28 ms) gives **about parity**. This assumes cost per call stays constant, so present it as an estimate.

## 6. Claim 2: adaptive h, the multi-level grid beats RT

### 6.1 Fair baseline vs the old proof of concept (Ada, N = 1M; times in ms)

| Scene | Neighbours | Single grid (cells r_max) | Multi-level grid | RT (sort + BVH + trace) | RT / multi-level |
|---|---|---|---|---|---|
| two-level, ratio 1, count | 79 | 1.82 | 4.04 | 6.95 | 1.72x slower |
| ratio 2 | 81 | 6.79 | 4.79 | 9.50 | 1.98 |
| ratio 4 | 87 | 38.56 | 4.58 | 8.66 | 1.89 |
| ratio 8 | 96 | 290.32 | 4.49 | 8.56 | 1.91 |
| ratio 16 | 108 | 2635.15 | 4.40 | 8.67 | 1.97 |
| profile α = 1 (h range 4.6) | 80 | 2.64 | 4.59 | 6.94 | 1.51 |
| profile α = 2 (h range 21.5) | 80 | 26.72 | 5.32 | 7.20 | 1.35 |
| profile α = 3 (h range 100) | 81 | 3071.50 | 5.39 | 7.30 | 1.35 |
| density, ratio 8 | - | 359.06 | 5.48 | 10.31 | 1.88 |
| density, ratio 16 | - | 7426.80 | 4.84 | 8.95 | 1.85 |
| density, ratio 8, N = 4M | - | 1734.30 | 19.13 | 34.44 | 1.80 |
| density, profile α = 2, N = 4M | - | 212.21 | 22.61 | 31.14 | 1.38 |

- All runs valid.
- RT beats only the single grid sized for the largest radius (up to 830x). That was the old proof-of-concept baseline.
- Data: `SPH_PC/rtx/frnn/data/multires_fair_remote_20261120_193748.txt`.

### 6.2 Extreme h ranges (Ada, profile α = 3, N = 1M, about 81 neighbours, 3 launches)

| h range | Levels | Multi-level grid ms (build + pass) | RT ms (sort + BVH + trace) | RT / multi-level |
|---|---|---|---|---|
| 100 | 7 | 5.39 (3.04 + 2.35) | 7.28 (1.50 + 5.78) | 1.35 |
| 333 | 9 | 5.16 | 7.58 | 1.47 |
| 1,000 | 10 | 5.08 | 8.08 | 1.59 |
| 3,332 | 12 | 4.93 | 8.65 | 1.76 |
| 9,999 | 14 | 4.95 (2.25 + 2.71) | 9.12 (1.59 + 7.53) | 1.84 |

- Density mode: 1.27 → 1.69.
- RT falls further behind as the h range grows.
- With an ideal refit (about 0.25 ms instead of a build), RT would still be about 1.15x slower at h range 100 and about 1.6x at 10^4 (estimate).
- Data: `SPH_PC/rtx/frnn/data/multires_stress_remote_ada_20261120.txt`.

### 6.3 Mechanism (harness `profile` scene, RT only, about 81 neighbours)

| Core (h range) | Laptop: calls/query, trace ms, ns/call, empty-program trace | Ada: calls/query, trace ms, empty-program trace |
|---|---|---|
| 0.01 (100) | 188.6, 10.56, 0.056, 7.08 | 188.3, 5.29, 3.41 |
| 0.001 (1,000) | 189.5, 11.95, 0.063, 7.69 | 189.6, 6.19, 3.75 |
| 0.0001 (10^4) | 190.5, 13.74, 0.072, 8.50 | 190.6, 7.39, 4.17 |

- Candidate calls stay flat; traversal cost per candidate rises, even with an empty intersection program.
- Interpretation: boxes of very different sizes in one BVH reduce traversal efficiency.
- **Hardware spheres** in the same scenes need a segment of 2 x the *global* r_max: 980 / 1,489 / 1,926 calls per query; trace 130 / 199 / 255 ms (laptop), 79 / 121 / 166 ms (Ada). Structurally unusable for multi-resolution.

## 7. Claim 3: Blackwell hardware spheres do not help

- **Empirical semantics** (OptiX 9.1, laptop, Blackwell):
  - A ray starting inside a built-in sphere produces no hit.
  - Zero-length rays: 10,345,122 any-hit calls, none passing a containment test, none flagged back-face (variants `sphere` and `sphereany`). This is consistent with the OptiX 9.1 guide's front-face statement and with Mochi's remark that "OptiX considers spheres hollow".
- **Correct formulation:** the ray starts 2 r_max before the query point and ends at the query; any-hit tests containment. It is valid (1 rounding mismatch in 7.6M).
- **Decomposition at k = 80 (laptop):**

| Variant | Calls per query | Trace |
|---|---|---|
| Point query, custom boxes | 181 | 9.85 ms |
| Segment, custom boxes | 342 | 19.60 ms |
| Segment, hardware spheres | 117 | 22.96 ms |

  At equal ray length, hardware spheres cut invocations 2.9x but the trace is 16% slower. The cost per any-hit invocation is about 3.4x an intersection call.
- **Conclusion:** the hollow-sphere semantics force a segment query that costs more than the software intersection it replaces.
- To our knowledge this is the first evaluation of hardware spheres for neighbour search. Mochi and ORCS used custom AABBs.

## 8. Claim 4: where RT's published wins come from

### 8.1 Code-quality control: our RT vs ORCS's RT; our grid vs ORCS's GPU-CELL
Per-SM cost = ms x SM count (ORCS: 188 SMs; our laptop: 46). ORCS times are full Lennard-Jones steps from their Table 2 (wall boundaries, transcribed in `SPH_PC/rtx/frnn/make_speedup_figures.py`). Our grid runs LJ forces; our RT is the `base` count search.

| ORCS scenario | ORCS GPU-CELL | Our grid + LJ, FP32 | Our grid + LJ, FP64 | ORCS / our FP64 | Our RT | ORCS RT-REF |
|---|---|---|---|---|---|---|
| disordered, r = 1, 1M | 656 | 36.3 | 587 | 1.12 | 139 | 124 |
| cluster, r = 1, 1M | 1,282 | 41.3 | 750 | 1.71 | 152 | 897 |
| lattice, r = 1, 1M (self-only neighbours) | 558 | 25.2 | 81 | 6.86 | 111 | 113 |
| disordered, log-normal, 50k | 24,628 | 473 | 9,169 | 2.69 | 701 | 1,641 |
| lattice, log-normal, 50k | 24,440 | 473 | 9,152 | 2.67 | 678 | 1,993 |
| cluster, log-normal, 50k | 43,052 | 474 | 9,915 | 4.34 | 961 | 27,260 |

- Our RT is as efficient per SM as ORCS's own RT implementation or better (0.9-28x).
- An FP64 textbook cell list with LJ reproduces ORCS's GPU-CELL timings to within 1.1-1.7x (r = 1) and 2.7-4.3x (log-normal, all-pairs). The same cell list in FP32 is 18-91x faster.
- The penalty grows with arithmetic intensity (18-31x at r = 1, 52-91x when all pairs are scanned), approaching the 64:1 FP32:FP64 rate ratio. That is the FP64 signature.
- **Caveat: ORCS does not state the precision of GPU-CELL.** This is a quantitative inference, not a confirmation.
- Data: `SPH_PC/rtx/frnn/data/orcs_lj_local_5070tiL_20261120.txt`.

### 8.2 Confounds found by reading third-party code
These are facts from the cloned source code, not run:

- **RTNN** (github.com/horizon-research/rtnn, `src/optixNSearch/`):
  - Range search writes neighbours up to `params.limit` (K). When the K-th is found it calls `optixReportIntersection`, and the any-hit program `__anyhit__terminateRay` terminates the ray (`geometry.cu`, `write_res_radius`). Default K = 50 (`util.cpp`).
  - Range results are therefore truncated whenever a query has more than K neighbours.
  - `uploadData()` runs before the "total search time" timer starts (`main.cpp`).
  - The device-to-host result copy is inside the timed search (`search.cpp`, "result copy D2H").
- **cuNSearch** (github.com/InteractiveComputerGraphics/cuNSearch; RTNN's main grid baseline, from SPlisHSPlasH):
  - Defaults to FP64 (`CUNSEARCH_USE_DOUBLE_PRECISION ON` in `CMakeLists.txt`).
  - Every `find_neighbors()` re-copies particle data (`copyParticleData` in `cuNSearch.cu`).
  - It copies all neighbour lists, counts and offsets to the host (`cuNSearchDeviceData.cu` lines 288-290).
- Which cuNSearch precision RTNN used in its paper is **not verified**.

### 8.3 Bridge experiment: adding each confound to the fair comparison
All codes read the same file `pts_disordered_1M.txt` (1M uniform random points, box 103) or the surface scene. Apparent RT speedup = grid-side time / RT-side time.

**Laptop (5070 Ti Laptop):**

| Step | file r = 2.754 (78.7 neighbours) | surface r = 1 (90.8) | surface r = 0.5 (22.2) | file r = 1.0 (4.8) |
|---|---|---|---|---|
| Fair: complete, FP32, GPU-resident (grid / RT ms) | 3.20 / 11.92 → **0.27x** | 1.96 / 9.11 → 0.22x | 1.06 / 4.22 → 0.25x | 1.03 / 3.53 → 0.29x |
| RT best case with refit | 0.31x | 0.27x | 0.43x | 0.79x |
| + RT stops at K = 50 | 0.33x | 0.28x | 0.24x | 0.33x |
| + grid in FP64 | **4.22x** | **5.80x** | 3.07x | 2.08x |
| + grid re-uploads positions | 4.47x | 6.14x | 3.52x | 2.77x |
| + grid copies full lists to host | **7.93x** | **11.44x** | 5.72x | 3.67x |
| + RT copies its K-capped list (pageable; bound) | 2.52x | 2.96x | 1.03x | 0.48x |

- Fraction of neighbours returned with K = 50: 63% (r = 2.754), 51% (surface r = 1), 99% (r = 0.5), 100% (r = 1.0).
- FP64 grid vs FP32 grid on the same kernel: 12.9x slower at k = 79; 20.7x on surface r = 1; 6.3x at k = 4.8.
- Measured pageable device-to-host bandwidth: about 9.5-10 GB/s.

**Ada (RTX 5000):**

| Step | file r = 2.754 | surface r = 1 | surface r = 0.5 | file r = 1.0 |
|---|---|---|---|---|
| Fair (grid / RT ms) | 1.68 / 6.18 → **0.27x** | 1.01 / 4.80 → 0.21x | 0.53 / 2.44 → 0.22x | 0.45 / 1.84 → 0.24x |
| RT best case with refit | 0.33x | 0.28x | 0.41x | 0.60x |
| + RT stops at K = 50 | 0.14x (anomalous, see below) | 0.26x | 0.20x | 0.24x |
| + grid in FP64 | 1.71x | 5.23x | 2.56x | 1.63x |
| + grid re-uploads positions | 1.89x | 5.67x | 3.30x | 2.72x |
| + grid copies full lists to host | 3.96x | 13.18x | 6.21x | 4.16x |
| + RT copies its K-capped list (bound) | 1.73x | 2.60x | 0.89x | 0.36x |

- FP64 grid on Ada: 12.3x slower than FP32 at k = 79.
- **Anomaly:** on Ada, `kcap` at k = 79 took 12.11 ms vs 6.18 ms uncapped, the opposite of the laptop (9.80 vs 11.92). It is not explained: possibly contention, possibly an architecture difference in ray termination. Use the laptop's K-cap numbers.
- **Interpretation:** the single change "grid in FP64" turns RT's 3.7x loss into an apparent 1.7-5.8x win. Adding cuNSearch-style host round trips gives 4-13x, the order of RTNN's published 2.2-44x claims.
- Data: `SPH_PC/rtx/frnn/data/bridge_local_5070tiL_20261120.txt`, `bridge_remote_ada_20261120.txt`; analysis `analyze_bridge.py`.

### 8.4 Published regimes reproduced in our code
Laptop / Ada, min of 3 launches, all valid. Single grid cells = r_max (all-pairs when r_max ≥ box).

| Scene (positions random, box 103) | Neighbours | Single grid | Multi-level grid | RT | RT / single | RT / multi-level |
|---|---|---|---|---|---|---|
| ORCS log-normal (1,2) in [1,330], 50k | 3,210 | 9.89 / 4.74 | 1.28 / 0.72 | 17.42 / 7.62 | 1.76 / 1.61 | 13.6 / 10.5 |
| ORCS uniform [1,160], 50k | 29,007 | 9.92 / 5.15 | 6.23 / 3.13 | 182.8 / 81.5 | 18.4 / 15.8 | 29.4 / 26.0 |
| log-normal, 200k | 12,850-12,950 | 119.9 / 61.2 | 14.4 / 8.0 | 270.2 / 144.7 | 2.25 / 2.36 | 18.8 / 18.1 |
| uniform [1,160], 200k | 115,955 | 116.8 / 59.0 | 73.0 / 38.5 | 44,560 / 48,366 | 382 / 820 | 610 / 1,256 |
| DEM-like U[0.5,1.5], 1M | 5.7 | 1.13 / 0.64 | 2.89 / 8.07 | 3.65 / 6.73 | 3.23 / 10.4 | 1.27 / 0.83 |
| DEM-like U[0.2,2.0], 1M | 9.4 | 1.74 / 0.98 | 5.18 / 10.63 | 4.43 / 7.19 | 2.55 / 7.32 | 0.85 / 0.68 |
| DEM-like LN(0,0.5) in [0.3,3], 1M | 11.2 | 3.84 / 2.16 | 3.55 / 3.94 | 4.83 / 2.76 | 1.26 / 1.27 | 1.36 / 0.70 |

- Ada multi-level-grid numbers in the DEM rows look inflated (contention or the unoptimised device-to-host copy in its build). The best grid (single) still beats RT there.
- Our RT on the ORCS log-normal case (17.4 ms laptop) vs their RT-REF (8.73 ms on a 4x larger GPU) is consistent.
- ORCS's published uniform-[1,160] numbers are not internally consistent with their own log-normal throughput (about 11x apart). Their neighbour definition there may differ, so **exclude the uniform-radius case from cross-paper claims**.
- Data: `SPH_PC/rtx/frnn/data/mlbridge_local_5070tiL_20261120.txt`, `mlbridge_remote_ada_20261120.txt`.

**Surface (LiDAR-like, RTNN-style geometry), harness, laptop:**
- r = 1 (91 neighbours): dense grid 2.29 ms, compact grid 2.02 ms, RT 9.1 ms.
- r = 0.5 (22 neighbours): dense 1.24 ms, compact 1.07 ms, RT 4.2 ms.
- RT is 3.4-4.8x slower (Ada similar, Section 8.3).

### 8.5 Sparse queries with an empty domain (ORCS r = 1 style, box 1000, 1M, about 1 neighbour)

| | Dense grid | Compact grid | RT (build + trace) | RT trace only |
|---|---|---|---|---|
| Laptop, disordered | 27.85 | 1.11 | 2.58-2.61 | 0.39 |
| Laptop, lattice | 24.63 | 0.62 | 2.28-2.63 | 0.27-0.28 |
| Ada, disordered | 17.61 | 1.18 | 1.60-1.67 | 0.19-0.20 |
| Ada, lattice | 15.53 | 0.32 | 1.37-1.46 | 0.13-0.14 |

- RT's 10x "win" disappears against a compact grid: RT 1.4-4.3x slower with a rebuilt BVH.
- With a refit BVH (about 0.25 ms) RT would win this ultra-sparse case by about 1.5-2.5x (estimate from trace + refit). This is RT's genuine niche, and SPH is not in it.

## 9. Synthesis: why SPH loses and the papers win

1. **Per-candidate floor.** OptiX custom-primitive search invokes a programmable shader per candidate. That costs about 8x a grid distance test on Ada and Blackwell (5.5x even with an empty shader). The BVH removes about 2.8x of candidates (2.3 calls vs 6.4 tests per neighbour), which is not enough. SPH has 50-100 neighbours, so the per-neighbour cost dominates and RT loses 3-4x. The perfect-BVH estimate reaches only parity.
2. **Adaptive h** adds BVH heterogeneity, raising traversal cost per candidate, while a multi-level grid's cost stays flat. RT loses more as the h range grows.
3. **Hardware spheres** can't express point containment (hollow semantics). The required segment formulation costs more.
4. **Published wins come from:**
   - (a) FP64 grid baselines against FP32-only RT hardware. This is the dominant factor; it alone flips the outcome on our data. ORCS's GPU-CELL timings match an FP64 cell list (inferred).
   - (b) library baselines with host round trips inside the timer (cuNSearch);
   - (c) truncated RT searches (RTNN K-cap; 63% of SPH neighbours returned);
   - (d) grids sized by the largest radius under heterogeneous radii, or dense cell arrays over empty space;
   - (e) very sparse regimes (k ≈ 1-4) where an amortised BVH can win.
   SPH occupies none of these when a fair FP32 GPU-resident grid is used.

## 10. Novelty relative to the literature
- First controlled RT-vs-grid characterisation in the SPH regime: full neighbour sets, real SPH density and force kernels, two GPU generations, an ablation of every RT optimisation from the literature, an invocation-cost decomposition, and a validated cost model with a best-case bound.
- First comparison of RT against a multi-level grid with physically correlated h, up to 10^4 range, with a measured mechanism. ORCS's non-uniform radii are random and uncorrelated with position, and it has no multi-level or octree baseline.
- First evaluation of Blackwell hardware sphere / LSS primitives for neighbour search, with the semantic limitation verified empirically.
- A methodological finding: published RT-core neighbour-search speedups are sensitive to baseline precision and API choices. We show how much each confound contributes.
- A practical finding for our own solver: it uses the report + any-hit path, which costs 23-32% more than accumulating in the intersection program.

## 11. Threats to validity and open items
- **Not yet done: third-party head-to-head.** RTNN and cuNSearch are cloned and uploaded to `~/sph_ext_20261120` on the Ada box, but not built or run; the agent's safety check blocked it. Script ready: `SPH_PC/rtx/frnn/run_thirdparty_remote.sh` (user can run it). Until then, "not an implementation issue" rests on the per-SM comparison with ORCS's published RT numbers, our ablations, and the confound reproductions.
- **ORCS GPU-CELL precision is inferred.** Confirm with the authors.
- **RTNN's cuNSearch build precision and K setting in its paper are not verified.** Whether RTNN's result copy uses pinned memory is not verified, so the "RT copies its list" rows are bounds.
- **One API (OptiX 9.1), NVIDIA only.** Vulkan RT / DXR and AMD / Intel not tested.
- **Frozen particles:** refit quality is a best case, which favours RT. RT time excludes the Morton reorder, which also favours RT.
- **Scatter neighbour rule only** for variable h; the symmetric max(h_i, h_j) rule is not tested.
- **The multi-level grid is not optimised** (its build includes a device-to-host copy); its wins are conservative.
- **No energy measurements** (ORCS reports some).
- **The shared Ada GPU** caused outliers. The minimum over launches mitigates this. The flagged anomaly is the K-cap at k = 79 on Ada.
- **The perfect-BVH parity and refit-adjusted multi-resolution figures** are extrapolations from measured per-call costs.

## 12. Where everything is

All code and data from the two investigation worktrees (`SPH_PC_dir1`, `SPH_PC_dir2`) were copied into the main repo under `SPH_PC/rtx/frnn/` and committed on branch `multires-poc` (also pushed to `rtx`). Compiled build folders were not committed; rebuild with the `build_*.bat` / `build_*.sh` scripts.

Code:
- **Ablation and bridge harness** (`SPH_PC/rtx/frnn/`):
  - `abl.cu`, `ablPrograms.cu`, `ablShared.h`
  - `build_abl.bat` / `build_abl.sh`, `run_abl.sh` (sets: `ablate`, `ksweep`, `dists`, `profile`, `sparse`, `mech`, `bridge`)
  - `analyze_abl.py`, `analyze_bridge.py`, `remote_abl.sh`, `run_thirdparty_remote.sh`
- **Cost model** (`SPH_PC/rtx/frnn/`): `cost.cu`, `costPrograms.cu`, `costShared.h`, `run_cost.sh`, `fit_cost.py`, `remote_cost.sh`
- **Multi-level grid** (`SPH_PC/rtx/frnn/`):
  - `frnn_ml.cu` (scenes: `multires`, `profile`, `randrad`), `build_frnn_ml.bat` / `.sh`
  - `remote_sweep_ml.sh`, `run_remote_ml.sh`, `run_ml_bridge.sh`, `summarize_ml.py`
- **cuNSearch timing driver:** `SPH_PC/rtx/frnn/cuns_driver.cu` (also on the remote in `~/sph_ext_20261120/extdrv/`).

Data (all new files; `*_clocks.txt` = 1 Hz clock and power logs):
- `SPH_PC/rtx/frnn/data/`:
  - `abl_local_5070tiL_20261120.txt` (ablation + k-sweep + lattice, laptop)
  - `abl2_local_5070tiL_20261120.txt` (refit, aabbseg, compact/sparse, profile, laptop)
  - `abl_remote_ada_20261120.txt` (all of the above, Ada)
  - `bridge_local_5070tiL_20261120.txt`, `bridge_remote_ada_20261120.txt`
  - `orcs_lj_local_5070tiL_20261120.txt`
  - `cost_local_5070tiL_sweep3_20261120.txt`, `cost_local_5070tiL_orcs2_20261120.txt`, `cost_local_5070tiL_instrwarp_20261120.txt`, `cost_model_5070tiL_20261120.json`
  - `cost_remote_NVIDIA_RTX_5000_Ada_Generation_20261120*.txt`
- `SPH_PC/rtx/frnn/data/`:
  - `multires_fair_remote_20261120_193748.txt`
  - `multires_stress_remote_ada_20261120.txt`
  - `mlbridge_local_5070tiL_20261120.txt`, `mlbridge_remote_ada_20261120.txt`
- Shared point file: `SPH_PC/rtx/frnn/data/pts_disordered_1M.txt` (also on the remote in `~/sph_ext_20261120/extdrv/`).
- Earlier data in the main repo: `SPH_PC/rtx/frnn/data/` (`frnn_box1000.json`, `multires_poc.txt`), `SPH_PC/rtx/bench/data/compare_results.json`.

## 13. References
- Zhu, "RTNN: Accelerating Neighbor Search Using Hardware Ray Tracing", PPoPP 2022. arXiv:2201.01366. Code: https://github.com/horizon-research/rtnn
- Meneses et al., "Advancing RT Core-Accelerated Fixed-Radius Nearest Neighbor Search" (ORCS), Future Generation Computer Systems 2026. arXiv:2601.15633
- Meneses et al., "Ray Tracing Cores for General-Purpose Computing: A Literature Review", arXiv:2603.28771 (2026)
- Zhao et al., "Leveraging ray tracing cores for particle-based simulations on GPUs", Int. J. Numer. Methods Eng. 124:696 (2023). DOI 10.1002/nme.7139
- Mandarapu et al., Mochi (Eurographics 2024, arXiv:2402.14801); "Rethinking Collision Detection on GPU Ray Tracing Architecture", ICS 2026, arXiv:2604.23520
- Xiao et al., "A Case Study for Ray Tracing Cores: Performance Insights with Breadth-First Search and Triangle Counting in Graphs", Proc. ACM Meas. Anal. Comput. Syst. 9(2) (2025). DOI 10.1145/3727108
- cuNSearch: https://github.com/InteractiveComputerGraphics/cuNSearch
- Winchenbach & Kolb, "Multi-Level-Memory Structures for Adaptive SPH Simulations", VMV 2019; CGF 39(6) 2020
- Winchenbach, Hochstetter & Kolb, "Constrained Neighbor Lists for SPH-based Fluid Simulations", SCA 2016
- Longva et al., "Fast Octree Neighborhood Search for SPH Simulations", 2022
- Green, "Particle Simulation using CUDA", NVIDIA, 2010
- Mao et al., "A GPU accelerated mixed-precision SPH framework with cell-based relative coordinates", Eng. Anal. Bound. Elem. 161 (2024)
- NVIDIA RTX Blackwell GPU Architecture whitepaper (hardware sphere / LSS intersection); NVIDIA OptiX 9.1 Programming Guide
