# Accelerating Neighbourhood Search in Smoothed Particle Hydrodynamics

**Preliminary project report — 30 September 2026**

This report follows the development of the CPU solver from the sequential C++ baseline to OpenMP multithreading and then to spatial hashing. The starting observation is that the baseline spends most of a physics step finding neighbouring particles. The work therefore proceeds in two stages: distribute particle calculations across CPU threads, then reduce the number of candidate particles that those calculations examine.

The discussion uses the existing source code, benchmark data, and figures in the project. The earlier sequential measurements establish where the project left off; the later comparison measurements evaluate the sequential, OpenMP all-pairs, and OpenMP spatial-hash variants. Results from separate measurement campaigns are identified where they appear. All times are in milliseconds unless stated otherwise.

## 1. Where we left off: sequential scaling and the neighbourhood-search bottleneck

### 1.1 The sequential C++ baseline

The sequential implementation provides the reference point for the parallel work. It represents the fluid as particles and computes density and force contributions by scanning every particle for each target particle. A distance test determines whether a candidate falls within the smoothing kernel's support. Candidates outside that support contribute nothing, but the program still pays the cost of reading their positions and computing their separation.

For a system containing N particles, one complete all-pairs pass performs approximately N² candidate checks. The solver makes such a pass for density and another for force evaluation. The exact work differs between the two passes, but both have quadratic candidate-search complexity. Doubling N therefore produces approximately four times as many candidate checks per pass.

This baseline matters because the parallel implementation can retain the same particle model and interaction calculations. Speedup can then be measured against compiled sequential C++ rather than against the older Python exploration.

### 1.2 Scaling with particle count

The earlier sequential sweep used the hydrostatic scene and measured the physics update without rendering. The initial particle spacing was chosen to approach a requested particle count; the scene generator rounded the lattice geometry, so the actual count can differ slightly from the request.

![Earlier sequential C++ scaling results](C:/Users/Aditya/Documents/Projects/SPH_PC/sequential_c/data/scaling_plot.png)

*Figure 1. Earlier sequential scaling sweep. The increasing slope is consistent with the quadratic growth expected from an all-pairs search. The dashed curve is a quadratic fit. The saved figure's horizontal label says “fluid + ghosts,” but the associated JSON records zero ghost particles; the counts in this report are the actual fluid counts.*

| Requested particles | Actual fluid particles | Mean time per step | Standard deviation across 10-step windows |
|---:|---:|---:|---:|
| 500 | 493 | 1.873 | 0.113 |
| 1,000 | 1,025 | 7.994 | 0.249 |
| 2,000 | 2,030 | 29.299 | 0.198 |
| 3,000 | 3,053 | 62.697 | 4.040 |

The increase from 1,025 to 2,030 particles is close to a doubling of problem size. Runtime increases by approximately 3.67×. Increasing the count to 3,053 particles gives approximately 7.84× the runtime at 1,025 particles. Fixed costs, execution variability, and differences in the particle state affect the exact ratios, but the overall curve clearly becomes more expensive as the number of possible pairs grows.

The source for this sweep is [the earlier sequential scaling dataset](C:/Users/Aditya/Documents/Projects/SPH_PC/sequential_c/data/scaling_results.json). These values describe that measurement campaign. Later sections use the sequential baseline recorded alongside the parallel comparison campaign when calculating cross-implementation speedups.

### 1.3 Breaking down an average timestep

A total runtime curve identifies poor scaling but does not identify where the time is spent. The breakdown mode separates the physics update into six measured categories:

| Stage | Operation | Purpose |
|---|---|---|
| A | Density neighbourhood search | Find particles close enough to contribute to density. |
| B | Density computation | Evaluate and accumulate kernel contributions from the selected neighbours. |
| C | Force neighbourhood search | Find valid neighbours for pressure and viscosity interactions. |
| D | Force computation | Evaluate the force contributions for the selected neighbours. |
| E | Pressure computation | Convert density to pressure using the equation of state. |
| F | Boundary handling | Apply the response at the domain walls. |

![Sequential timestep breakdown at 2030 fluid particles](C:/Users/Aditya/Documents/Projects/SPH_PC/sequential_c/data/breakdown_n2000.png)

*Figure 2. Earlier sequential breakdown at 2,030 fluid particles. The two neighbourhood-search bars dominate the measured step.*

| Stage | Mean time per step |
|---|---:|
| A — Density neighbourhood search | 14.645 |
| B — Density computation | 0.394 |
| C — Force neighbourhood search | 16.953 |
| D — Force computation | 1.715 |
| E — Pressure computation | 0.015 |
| F — Boundary handling | 0.003 |
| **Complete timed update in breakdown mode** | **33.773** |

The two search stages take 31.598 ms together. Dividing this by the complete timed update gives a **93.56% search share**. The useful density and force arithmetic is much smaller than the work required to identify which particle pairs should participate in it.

The six bars do not exhaust every operation in the timed update. Velocity and position updates, control flow, and timing overhead also contribute to its total. In addition, breakdown mode explicitly materializes temporary neighbour lists so it can time search separately from arithmetic. It is a diagnostic execution path, and its total need not equal the normal solver timing. Its stage proportions should be interpreted within that path. The raw values are in [the sequential breakdown dataset](C:/Users/Aditya/Documents/Projects/SPH_PC/sequential_c/data/breakdown_n2000.json).

### 1.4 The bottleneck becomes more pronounced at larger N

The later comparison campaign also measured the sequential breakdown at 30,016 particles, matching the larger example used in the presentation.

![Sequential timestep breakdown at 30016 particles](C:/Users/Aditya/Documents/Projects/SPH_PC/parallel_c/data/breakdown_n30000_sequential_t1.png)

*Figure 3. Sequential breakdown at 30,016 particles in the later campaign. The density and force searches take approximately 1,096.64 and 1,162.81 ms respectively.*

At this size, the searches occupy 2,259.45 ms of a 2,275.15 ms timed update, or **99.31%**. The evidence therefore gives a specific optimization target. Improving pressure calculation alone would affect a very small portion of runtime; accelerating neighbourhood search can affect almost the whole step.

The next step is to identify which parts of the SPH update can run concurrently and apply multithreading to the dominant loops.

## 2. The SPH computation and opportunities for parallel execution

### 2.1 What a particle stores

Smoothed Particle Hydrodynamics is a particle-based representation of a fluid. Each particle carries a position and velocity, while density, pressure, and acceleration are calculated from the current state. A smoothing kernel assigns a contribution according to the distance between particles. Contributions vanish outside a finite support radius.

For clarity, this report uses H for the smoothing-length parameter used in the code and R for the support radius. The implemented kernel has support for distance less than 2H, so **R = 2H**. This distinction is also used when choosing the spatial-grid cell size in Section 5.

The main state arrays contain positions, velocities, accelerations, densities, and pressures. Most calculations read several of these arrays but write only the result associated with one target particle. That ownership pattern is the basis of the OpenMP implementation.

### 2.2 Initialization and density

Initialization establishes the particle positions, velocities, masses, domain, and fluid parameters. For the benchmark scenes, particles begin on a deterministic lattice. The validation driver derives H from the initial spacing and configures the pressure stiffness from a characteristic velocity. An initial force evaluation prepares the acceleration needed by the integrator.

Density at particle i is calculated from the particles within the support radius:

$$
\rho_i = \sum_{j\in\mathcal{N}(i)} m_j W(\lVert\mathbf{x}_i-\mathbf{x}_j\rVert,H).
$$

Here, W is the smoothing kernel and the neighbourhood contains the particles whose contributions are nonzero. The density sum includes the particle's own kernel contribution. The all-pairs implementation discovers this neighbourhood by testing every candidate j.

Each density calculation reads positions and masses, then writes one density value. Density for particle i does not depend on the newly calculated density of another particle, so different target particles can be assigned to different threads.

### 2.3 Pressure and force evaluation

Once density is available, the implemented equation of state computes pressure:

$$
p_i = K(\rho_i-\rho_0),
$$

where K is the pressure multiplier and rho-zero is the target density. This is an elementwise operation. Every particle can calculate its pressure independently after its density is ready.

Force evaluation then combines pressure interactions, viscosity, and gravity. The pressure contribution uses the symmetric combination of pressure divided by density squared for the two interacting particles. The viscosity calculation depends on their relative positions and velocities. The force loop skips self-interactions and zero-separation pairs before evaluating the kernel gradient.

All target particles can evaluate their accelerations concurrently provided the positions, velocities, densities, and pressures remain stable during the force pass. Each target accumulates its own acceleration. The program must finish density and pressure calculations before allowing the force stage to read those arrays.

### 2.4 Integration, boundaries, and the order of a step

The benchmark driver uses a kick–drift–kick sequence. It advances velocity by half a step using the previous acceleration, advances position using that intermediate velocity, applies the wall response, recalculates acceleration at the updated positions, and completes the remaining half-velocity update.

```text
Initialize positions and velocities
Calculate initial densities, pressures, and accelerations

Repeat for each timestep:
    Choose dt from the current velocity and acceleration limits
    Half kick: v[i] = v[i] + 0.5 * dt * a[i]
    Drift:     x[i] = x[i] + dt * v[i]
    Apply boundary response
    Calculate updated densities
    Calculate updated pressures
    Calculate updated accelerations
    Half kick: v[i] = v[i] + 0.5 * dt * a[i]
```

The time-step selection uses both an acoustic limit and a force limit. The code evaluates the maximum speed and acceleration, forms the two limits, and chooses the smaller value. A parallel implementation could calculate those maxima with reductions; the current benchmark driver performs that scan serially and outside the timed update region.

The wall response is a per-particle clamp and velocity reflection with a restitution factor. Although some existing labels call it an “elastic” boundary, the restitution factor need not equal one. These benchmark runs use the wall-response path without adding ghost particles.

### 2.5 Parallelism and synchronization

| Operation | Parallel work unit | Required input | Current parallel benchmark implementation |
|---|---|---|---|
| Density | One target particle | Positions and kernel parameters | OpenMP outer particle loop |
| Pressure | One density entry | Completed density values | OpenMP loops inside vector arithmetic |
| Force | One target particle | Positions, velocities, densities, pressures | OpenMP outer particle loop |
| Velocity updates | One particle | Acceleration and dt | OpenMP particle loops |
| Position update | One particle | Intermediate velocity and dt | OpenMP particle loop |
| Boundary response | One particle | Updated position and velocity | Serial loop; independent per-particle work is available |
| Time-step limits | Reduction over particles | Speeds and accelerations | Serial maximum scan in the driver |

Synchronization follows the data dependencies. Updated positions must be ready before neighbourhood evaluation. Densities and pressures must be ready before force evaluation. New accelerations must be ready before the second half kick. OpenMP worksharing barriers enforce the necessary order in the current implementation.

Consecutive timesteps retain a sequential dependency because the next step requires the previous step's particle state. The available parallelism is therefore across particles within each stage.

## 3. Multithreading with OpenMP

### 3.1 Distributing target particles

OpenMP provides shared-memory threading within a process. Threads can read the same particle arrays while owning different output elements. The all-pairs parallel variant preserves the search algorithm and distributes the outer loop over target particles.

The sequential density structure is:

```text
for i = 0 to N - 1:
    rho[i] = 0
    for j = 0 to N - 1:
        if distance(x[i], x[j]) < R:
            rho[i] += mass[j] * W(x[i], x[j])
```

The OpenMP version changes how the outer iterations are executed:

```cpp
// Illustrative pseudocode; each iteration owns rho[i].
#pragma omp parallel for schedule(static)
for (int i = 0; i < N; ++i) {
    double density = 0.0;
    for (int j = 0; j < N; ++j) {
        if (distance(x[i], x[j]) < R)
            density += mass[j] * W(x[i], x[j]);
    }
    rho[i] = density;
}
// The loop completes before the next dependent stage begins.
```

Each iteration accumulates the density for one target. Two threads do not update the same density entry, so the algorithm does not require an atomic operation for every interaction. The local accumulator in this pseudocode makes that ownership explicit; the implementation also achieves ownership by assigning each outer-loop index to one thread.

Static scheduling suits the all-pairs pass because every target initially scans the same number of candidates. The amount of force arithmetic can still vary with the number of accepted neighbours, but candidate scanning dominates this version.

### 3.2 Parallel pressure and force calculations

Pressure calculation is expressed in the source as vector subtraction followed by vector multiplication. The parallel vector implementation contains OpenMP loops for those operations. It therefore contributes parallel-region and intermediate-array costs even though the pressure expression itself is compact.

```text
After every density is available:
    parallel for i = 0 to N - 1:
        pressure[i] = K * (density[i] - rest_density)

After every pressure is available:
    parallel for each fluid particle i:
        a_local = gravity
        for j = 0 to N - 1:
            if i != j and separation is valid and distance(i, j) < R:
                a_local += pressure_contribution(i, j)
                a_local += viscosity_contribution(i, j)
        acceleration[i] = a_local
```

The pressure pseudocode shows the logical elementwise operation; the current vector expression implements it as separate arithmetic passes. The force pseudocode retains the target-owned accumulation used by the solver.

A pair may be visited once while evaluating i and again while evaluating j. Computing each pair only once would require coordinating updates to two particles, for example through atomics, private buffers, or another accumulation scheme. The current design accepts the repeated pair visit to keep writes independent and the threaded loop straightforward.

### 3.3 Parallel integration with ordered stages

The same ownership principle applies to the integration loops:

```text
dt = choose_timestep(current_state)

parallel for each fluid particle i:
    velocity[i] += 0.5 * dt * acceleration[i]

parallel for each fluid particle i:
    position[i] += dt * velocity[i]

apply_boundary_response()              // serial in the current driver
update_density_pressure_and_forces()   // ordered stages described above

parallel for each fluid particle i:
    velocity[i] += 0.5 * dt * acceleration[i]
```

Each `parallel for` completes before the next dependent operation. This prevents one stage from observing a mixture of old and updated particle data. The pseudocode follows the benchmark driver's integration path, which performs these steps explicitly.

### 3.4 Expected performance and remaining work

If the expensive work divided perfectly over P threads, its elapsed time would approach the sequential time divided by P. Actual performance also includes serial work, barriers, parallel-region management, cache effects, and memory traffic. For small N, those costs can be comparable to the useful work assigned to each thread.

Multithreading still leaves approximately two all-pairs candidate passes per update. At fixed P, the dominant work remains proportional to N². The implementation can lower the runtime curve while retaining its quadratic shape. The next section checks both the speedup and the remaining stage distribution.

The relevant implementation is [the OpenMP all-pairs solver](C:/Users/Aditya/Documents/Projects/SPH_PC/parallel_c/prehash/physics_prehash.cpp), with integration in [the parallel benchmark driver](C:/Users/Aditya/Documents/Projects/SPH_PC/parallel_c/validate_main.cpp) and threaded pressure operations in [the vector implementation](C:/Users/Aditya/Documents/Projects/SPH_PC/parallel_c/backend/linalg.cpp).

## 4. Multithreading results: scaling with threads and the remaining bottleneck

### 4.1 How the comparison is measured

The later comparison uses three variants: sequential C++, OpenMP all-pairs (labelled `prehash` in the data), and OpenMP with the spatial grid (labelled `hash`). The benchmark harness generates a hydrostatic lattice, runs a headless executable, and records the mean of 10-step window averages. Standard deviation describes the variation between those windows.

The timer surrounds the velocity updates, position update, boundary handling, and density/pressure/force evaluation. Scene setup, rendering, snapshot output, and time-step selection are outside this interval. The metric therefore describes the timed physics-update region rather than the entire process wall time.

![Sequential and OpenMP all-pairs scaling across particle counts](C:/Users/Aditya/Documents/Projects/SPH_PC/parallel_c/data/scaling_all_pairs.png)

*Figure 4. Sequential and OpenMP all-pairs scaling for one through 32 threads. Additional threads reduce runtime, while the curves retain the strong upward curvature associated with all-pairs work.*

The sequential and one-thread OpenMP curves are close over much of the range. This supports using the one-thread OpenMP run as a denominator for thread scaling and the separately compiled sequential run as a reference for the full implementation comparison.

### 4.2 Strong scaling at approximately 10,000 particles

Holding the particle count fixed while increasing the thread count measures strong scaling. Define speedup and thread efficiency as:

$$
S(P)=\frac{T_{\mathrm{OpenMP},1}}{T_{\mathrm{OpenMP},P}},
\qquad E(P)=\frac{S(P)}{P}.
$$

At an actual count of 9,933 particles, the saved measurements are:

| Threads | OpenMP all-pairs time per step | Speedup against one-thread OpenMP | Thread efficiency |
|---:|---:|---:|---:|
| 1 | 327.232 | 1.00× | 100.0% |
| 2 | 172.769 | 1.89× | 94.7% |
| 4 | 92.969 | 3.52× | 88.0% |
| 8 | 50.911 | 6.43× | 80.3% |
| 16 | 28.047 | 11.67× | 72.9% |
| 24 | 26.904 | 12.16× | 50.7% |
| 32 | 27.424 | 11.93× | 37.3% |

The separate sequential executable takes 327.258 ms at this particle count, giving approximately **12.16× speedup** for 24-thread OpenMP against that baseline as well. The largest reductions occur before 16 threads. Moving from 16 to 24 threads gives a smaller improvement, and 32 threads are slightly slower in this particular measurement.

These results indicate diminishing returns at this workload size. They do not identify a single hardware cause: synchronization, available execution resources, memory behavior, and runtime variation can all contribute. At larger N, 32 threads can still outperform 24 because more work is available per thread.

### 4.3 Why small particle counts behave differently

The project also contains an earlier thread-speedup figure based on median measurements. It shows the effect of insufficient work particularly clearly.

![Earlier OpenMP speedup measurements at small particle counts](C:/Users/Aditya/Documents/Projects/SPH_PC/parallel_c/data/parallel_speedup.png)

*Figure 5. Earlier OpenMP all-pairs speedup measurements for 100–3,000 particles, sourced from the accompanying median-based CSV. This is a separate measurement series from the 10-step-window dataset in Section 4.2; its values are used here to illustrate the small-workload trend.*

At 100 particles, adding threads makes the measured run slower than one thread. At 3,000 particles, the figure reports approximately 10.2× speedup with 16 threads, while increasing to 32 or 64 does not improve it. The qualitative explanation is that thread-management and synchronization costs must be amortized over enough useful particle work. Thread count alone cannot guarantee a speedup.

The source for this figure is [the earlier thread-speedup CSV](C:/Users/Aditya/Documents/Projects/SPH_PC/parallel_c/data/parallel_speedup.csv). The primary numerical comparisons in the remainder of this report use the later [common scaling dataset](C:/Users/Aditya/Documents/Projects/SPH_PC/parallel_c/data/scaling_results.json).

### 4.4 Breakdown after multithreading

![OpenMP all-pairs breakdown at 30016 particles and 24 threads](C:/Users/Aditya/Documents/Projects/SPH_PC/parallel_c/data/breakdown_n30000_prehash_t24.png)

*Figure 6. OpenMP all-pairs breakdown at 30,016 particles with 24 threads. Searches are faster than in Figure 3, but still dominate the step.*

| Stage | Sequential breakdown | OpenMP all-pairs, 24 threads |
|---|---:|---:|
| Density neighbourhood search | 1,096.639 | 81.072 |
| Density computation | 3.363 | 0.962 |
| Force neighbourhood search | 1,162.807 | 93.502 |
| Force computation | 12.076 | 1.291 |
| Pressure computation | 0.066 | 0.676 |
| Boundary handling | 0.018 | 0.061 |
| **Complete timed update in breakdown mode** | **2,275.148** | **178.329** |

Multithreading reduces this breakdown-mode total by approximately 12.76×. The two search stages fall from 2,259.45 ms to 174.57 ms, but they still account for **97.89%** of the OpenMP step. Their absolute cost is lower; their dominance remains.

The table also shows that individual cheap stages do not necessarily improve with threading. For example, pressure evaluation takes longer in this run. Its vector operations create short parallel passes, for which parallel overhead can exceed the benefit. A small-stage timing should be interpreted with its measurement variability, but the broader result is clear: nearly all remaining time is still spent in search.

![OpenMP all-pairs stage costs as particle count increases](C:/Users/Aditya/Documents/Projects/SPH_PC/parallel_c/data/breakdown_vs_n_prehash_t24.png)

*Figure 7. Stage costs across particle counts for the 24-thread all-pairs variant. The two search curves dominate and continue to grow sharply.*

The all-pairs search therefore remains the limiting algorithm. Additional CPU threads divide its work across more workers; the next optimization must reduce the work that each target particle needs to perform.

## 5. Spatial hashing, neighbourhood queries, and CSR-like storage

### 5.1 Using the finite interaction radius

The smoothing kernel already tells us that distant particles cannot contribute. Spatial hashing organizes particles by position so those distant candidates can be skipped before calculating their pairwise separation.

The implementation uses a uniform grid over the bounded domain. Each cell is assigned a flattened integer ID, and each particle is placed in the cell containing its position. The repository refers to this as a spatial hash. More specifically, it is a directly indexed grid of particle buckets: the finite grid provides a unique cell index without a general-purpose hash table or hash-collision resolution.

The cell side is chosen as R = 2H, matching the kernel support. In two dimensions, any particle less than R away from a target must lie in the same cell or one of the eight cells touching it. A query therefore examines a 3×3 region:

```text
             Candidate cells around the target cell

                +---------+---------+---------+
                | NW      | N       | NE      |
                +---------+---------+---------+
                | W       | target  | E       |
                +---------+---------+---------+
                | SW      | S       | SE      |
                +---------+---------+---------+

              Cell side = R = 2H; query 9 cells in 2D.
```

The cell query provides a conservative candidate set. The program still applies the exact distance test to each candidate because a square cell region includes points outside the circular kernel support. In three dimensions the same idea uses 27 nearby cells, but the present solver and measurements are two-dimensional.

### 5.2 Mapping a position to a cell

For domain origin (xmin, ymin), cell side s = R, and grid width Cx, the cell coordinates and flattened index are:

$$
c_x=\left\lfloor\frac{x-x_{\min}}{s}\right\rfloor,\qquad
c_y=\left\lfloor\frac{y-y_{\min}}{s}\right\rfloor,\qquad
c=c_x+c_y C_x.
$$

The origin subtraction places the grid in the correct world-coordinate frame. The driver sizes the grid from the scene domain and includes room for the upper edge. The implementation clamps cell coordinates to valid ranges when assigning particles, preventing a particle exactly on a boundary from generating an invalid array index.

Cell IDs are recalculated after particles move. A previously built grid could otherwise contain outdated membership and omit newly nearby particles. The current solver rebuilds the grid during each force-update call, then uses that grid for both density and force queries at the same particle positions.

### 5.3 Why use a CSR-like representation?

A cell needs to store a variable number of particle IDs. Allocating a maximum-length particle array for every cell would waste memory, and maintaining a separate dynamically allocated container per cell would introduce additional allocation and traversal overhead.

The implemented representation places all particle IDs in one contiguous array. Another array stores the beginning of each cell's range. This resembles compressed sparse row storage: a cell acts like a row, `cell_start` acts like row offsets, and `cell_particles` holds the entries belonging to those rows.

| Array | Length | Meaning |
|---|---:|---|
| `cell_id` | N | Cell containing each particle. |
| `cell_count` | C | Number of particles assigned to each grid cell. |
| `cell_start` | C + 1 | Start offsets, including a sentinel at the end. |
| `cell_particles` | N | Particle IDs grouped by cell. |
| Temporary `cursor` | C | Next free slot in each cell's range during construction. |

If cell c starts at offset `cell_start[c]` and the next cell starts at `cell_start[c+1]`, all its members are stored in the half-open range between those offsets. An empty cell has equal start and end offsets. The final sentinel equals N.

### 5.4 Worked example of construction

Consider seven particles distributed across four cells:

```text
Particle ID:       0  1  2  3  4  5  6
Assigned cell:     2  0  2  1  0  3  2

cell_count:       [2, 1, 3, 1]
cell_start:       [0, 2, 3, 6, 7]
cell_particles:   [1, 4, 3, 0, 2, 6, 5]
                   ----  -  -------  -
                    c0  c1     c2   c3
```

Cell 2 contains three particles. Its range starts at offset 3 and ends before offset 6, so its particle IDs are `[0, 2, 6]`. Retrieving that cell requires reading two offsets and traversing three entries. It does not require searching all seven particles.

The prefix sum converts counts into non-overlapping output ranges. Scattering each particle ID into its cell's range then completes the representation. This groups indices by cell; it does not physically reorder every particle state array or allocate a full list of particle-to-particle neighbours.

### 5.5 Construction pseudocode and parallel work

```text
function build_grid(position, N, C):
    parallel for i = 0 to N - 1:
        cx = floor((position[i].x - xmin) / R)
        cy = floor((position[i].y - ymin) / R)
        clamp cx and cy to valid grid coordinates
        cell_id[i] = cx + cy * grid_width

    fill cell_count with zero

    for i = 0 to N - 1:
        cell_count[cell_id[i]] += 1

    cell_start[0] = 0
    for c = 0 to C - 1:
        cell_start[c + 1] = cell_start[c] + cell_count[c]

    cursor = copy of cell_start[0 : C]
    for i = 0 to N - 1:
        c = cell_id[i]
        cell_particles[cursor[c]] = i
        cursor[c] += 1
```

The current implementation parallelizes particle-to-cell assignment. Counting, prefix summation, and scattering are serial. This avoids races in increments to shared cell counters and cursors. Making those construction stages parallel would require an explicit coordination strategy, such as per-thread counts or atomic updates; the current results already include their serial cost.

### 5.6 Querying the grid for density and force

```text
function candidates_for(i):
    for each valid cell c in the 3x3 region around particle i:
        for k = cell_start[c] to cell_start[c + 1] - 1:
            yield cell_particles[k]

build_grid(updated_positions)

parallel for each particle i:
    density[i] = 0
    for j in candidates_for(i):
        if distance(i, j) < R:
            density[i] += mass[j] * W(i, j)

calculate pressures from completed densities

parallel for each fluid particle i:
    acceleration[i] = gravity
    for j in candidates_for(i):
        if i != j and separation is valid and distance(i, j) < R:
            acceleration[i] += pressure_contribution(i, j)
            acceleration[i] += viscosity_contribution(i, j)
```

This pseudocode expresses the geometric query and uses coordinate bounds to identify valid neighbouring cells. The current source uses flattened cell-ID offsets and checks their global index range. At a row edge, that shortcut can enumerate extra wrapped cells; the exact distance filter remains necessary. The intended candidate region is the 3×3 neighbourhood shown above.

The density and force formulas and target-owned threading structure remain conceptually the same as before. The inner loop now visits particles in nearby buckets instead of scanning the full particle array. A completed grid is read-only while those queries execute, so multiple threads can traverse its ranges concurrently.

### 5.7 Time and memory complexity

Let C be the number of grid cells and k-bar the average number of candidates visited per target particle. Building the grid requires O(N + C) work: assign N particles, clear and prefix-sum C cells, and scatter N indices. Querying all particles requires O(N × k-bar) candidate work. A useful model for the complete search is therefore:

$$
T_{\mathrm{search}}=O(N+C+N\bar{k}).
$$

When cell occupancy remains bounded and C grows no faster than N, the expected search cost is approximately O(N). This condition fits the benchmark's resolution scaling: the driver sets H proportional to particle spacing, so refining the initial lattice also reduces the support radius and cell size. The typical initial number of particles in a support region stays approximately constant.

This is not an unconditional worst-case guarantee. If many particles accumulate in a few cells, a query must visit more candidates. Increasing N in a fixed domain while keeping H fixed would also change the occupancy assumption. The near-linear claim is tied to the tested scene family and its smoothing-length choice.

The grid arrays require N entries for `cell_id`, N for `cell_particles`, C for `cell_count`, C + 1 for `cell_start`, and C temporary cursor entries. The construction peak is therefore approximately **2N + 3C + 1 integer entries**, excluding the physical state arrays and allocation overhead. Memory is O(N + C), becoming **O(N)** when C = O(N), as in this benchmark setup. The main solver avoids storing an N-by-N neighbour matrix. The diagnostic breakdown path does additionally materialize neighbour lists to time search and arithmetic separately; that extra storage is distinct from the grid representation itself.

## 6. Spatial-hashing results: scaling, breakdown, and evidence of improvement

### 6.1 Scaling across particle counts

![Spatial-hash scaling for one through 32 OpenMP threads](C:/Users/Aditya/Documents/Projects/SPH_PC/parallel_c/data/scaling_spatial_hash.png)

*Figure 8. OpenMP spatial-hash scaling over the tested particle-count range. The dashed fits in this plot assume linear scaling; the supporting evidence is the measured trend together with the bounded-neighbour argument, rather than the fitted line alone.*

The one-thread spatial-grid implementation takes approximately 7.805 ms at 9,933 particles, 25.632 ms at 30,016, and 76.469 ms at 99,960. Increasing the count by roughly ten times from the first to the last point increases runtime by approximately 9.80×. This is consistent with near-linear growth across those endpoints and is substantially different from the all-pairs scaling.

The curves contain local spikes and variability, so the measured behavior is not a perfectly straight line. The data does not establish the cause of each spike. The broad trend nevertheless shows that the grid makes much larger particle counts practical within the measured CPU update cost.

### 6.2 Thread scaling after spatial hashing

At 99,960 particles, the spatial-hash variant has the following thread scaling:

| Threads | Time per step | Speedup against one-thread spatial hash |
|---:|---:|---:|
| 1 | 76.469 | 1.00× |
| 2 | 39.932 | 1.91× |
| 4 | 22.702 | 3.37× |
| 8 | 13.660 | 5.60× |
| 16 | 10.954 | 6.98× |
| 24 | 10.737 | 7.12× |
| 32 | 10.171 | 7.52× |

The grid and multithreading work together: the grid reduces candidate work and OpenMP distributes the remaining particle calculations. However, the reduced work per target makes other costs more significant. Serial grid construction, barriers, pressure passes, boundary work, and memory accesses now represent a larger fraction of runtime. The current measurements show much smaller gains above 16 threads than below it.

At approximately 10,000 particles, the saved spatial-hash measurements are fastest at eight threads among the tested choices: approximately 1.660 ms, compared with 1.844 ms at 24 threads. The best thread count therefore depends on problem size and should not be assumed to be the largest available count.

### 6.3 Comparing sequential, multithreaded, and spatial-hash implementations

![Sequential versus OpenMP all-pairs and spatial hash at 16 threads](C:/Users/Aditya/Documents/Projects/SPH_PC/parallel_c/data/comparison_seq_vs_16t.png)

*Figure 9. The three implementation stages compared on a linear time axis. Both parallel curves use 16 threads. The spatial-hash curve lies near the axis because the all-pairs times are much larger.*

The following values use 16 threads to match Figure 9:

| Actual particles | Sequential all-pairs | OpenMP all-pairs, 16 threads | OpenMP spatial hash, 16 threads | All-pairs / hash ratio at 16 threads |
|---:|---:|---:|---:|---:|
| 9,933 | 327.258 | 28.047 | 1.959 | 14.31× |
| 30,016 | 2,953.674 | 254.084 | 4.525 | 56.16× |
| 99,960 | 37,461.026 | 3,058.771 | 10.954 | 279.24× |

The ratio grows with N because the all-pairs variant continues examining the whole particle set for each target, whereas the grid usually restricts a query to a small local candidate set. Holding thread count fixed is useful here: it makes the improvement attributable to the changed implementation of neighbourhood search rather than to an increase in workers.

![Log-log comparison of sequential, OpenMP all-pairs, and OpenMP spatial hash](C:/Users/Aditya/Documents/Projects/SPH_PC/parallel_c/data/comparison_seq_vs_16t_loglog.png)

*Figure 10. The existing logarithmic comparison figure makes the spatial-hash trend visible alongside the larger all-pairs timings. Both parallel series use 16 threads; local numerical values in this report are taken from the saved scaling JSON.*

A logarithmic view also helps distinguish the trend shapes. The all-pairs curves rise much more steeply with particle count. The spatial-hash curve grows more gradually, consistent with the expected effect of reducing the candidate set.

For comparison with the 24-thread breakdown figures, the corresponding normal-mode measurements are:

| Actual particles | Sequential all-pairs | OpenMP all-pairs, 24 threads | OpenMP spatial hash, 24 threads | All-pairs / hash ratio at 24 threads |
|---:|---:|---:|---:|---:|
| 9,933 | 327.258 | 26.904 | 1.844 | 14.59× |
| 30,016 | 2,953.674 | 225.064 | 4.002 | 56.24× |
| 99,960 | 37,461.026 | 2,458.834 | 10.737 | 229.01× |

At 30,016 particles, moving from sequential all-pairs to 24-thread all-pairs reduces the measured update from approximately 2.95 seconds to 225 ms. Adding the spatial grid reduces it to approximately 4.00 ms. The first improvement uses more CPU execution resources; the second removes most distant-particle checks.

### 6.4 Does the breakdown confirm that neighbourhood search improved?

![Spatial-hash breakdown at 30016 particles and 24 threads](C:/Users/Aditya/Documents/Projects/SPH_PC/parallel_c/data/breakdown_n30000_hash_t24.png)

*Figure 11. Spatial-hash breakdown at 30,016 particles with 24 threads. Grid construction is included in stage A. This is the corresponding optimized result for the all-pairs breakdown in Figure 6.*

| Stage | OpenMP all-pairs, 24 threads | OpenMP spatial hash, 24 threads |
|---|---:|---:|
| A — Density search, including grid construction for hash | 81.072 | 0.779 |
| B — Density computation | 0.962 | 0.286 |
| C — Force neighbourhood search | 93.502 | 0.638 |
| D — Force computation | 1.291 | 1.088 |
| E — Pressure computation | 0.676 | 0.199 |
| F — Boundary handling | 0.061 | 0.049 |
| **Combined search, A + C** | **174.574** | **1.417** |
| **Complete timed update in breakdown mode** | **178.329** | **3.331** |
| **Search share of complete timed update** | **97.89%** | **42.55%** |

The combined search stages become approximately **123.18× faster**, including the cost of building the grid in stage A. The breakdown-mode total improves by approximately **53.54×**. The normal-mode comparison at the same requested count gives approximately 56.24×, but these two totals come from different execution paths and should retain their separate labels.

The size of the change in A and C is the direct evidence that the intended bottleneck was addressed. Force arithmetic changes much less: stage D falls from 1.291 to 1.088 ms. After optimization it becomes the largest individual bar, while the two search categories together still represent a substantial share. Neighbourhood search remains necessary, but it no longer consumes nearly the entire measured update.

![Spatial-hash stage costs across particle counts](C:/Users/Aditya/Documents/Projects/SPH_PC/parallel_c/data/breakdown_vs_n_hash_t24.png)

*Figure 12. Stage costs for the 24-thread spatial-hash variant across particle counts. Compare the millisecond scale here with the hundreds-of-milliseconds search curves in Figure 7.*

The stage curves show that the optimization changes the balance of work across the tested range. Density arithmetic, force arithmetic, and search are now all visible. At the largest counts the search stages can again be individually significant, so the evidence supports a large reduction in search cost rather than its elimination.

### 6.5 What the measurements establish, and their practical limits

The results support three connected findings. First, the sequential baseline is dominated by neighbourhood search. Second, OpenMP substantially lowers runtime while leaving the all-pairs scaling and search dominance intact. Third, a spatial grid with contiguous cell membership sharply reduces the measured search cost and produces much gentler growth over the tested resolution range.

The largest ratios should be read with the sampling method in mind. The common harness varies run length to keep expensive runs manageable. At 30,016 particles, normal scaling uses 20 steps for sequential, 80 for 24-thread all-pairs, and 150 for 24-thread hash. At 99,960 particles, sequential uses only 10 steps, producing one timing window; its recorded standard deviation of zero is not evidence of zero runtime uncertainty. Different run lengths also sample different extents of the evolving particle trajectory. These are observed per-update measurement ratios, and a final benchmark campaign should use a defined common sampling protocol and repeated runs.

The current measurements do not include a dedicated discarded warmup interval, and the JSON does not record the complete CPU, compiler-version, power-mode, and thread-affinity metadata needed for a fully reproducible final comparison. Those details should accompany final reported results. The existing figures and measurements are retained here as the evidence available for this preliminary report.

### 6.6 Validation context and the next stage of the project

The presentation also introduces hydrostatic, dam-break, momentum-conservation, structural, and time-step checks. These provide a separate test of whether faster execution preserves acceptable simulation behavior. The benchmark figures in this report use the hydrostatic scene; their timing improvement alone does not establish agreement with an experimental dam-break trajectory or equivalence of every particle state between implementations.

The existing sequential validation report records strong momentum conservation, but also a pressure-gradient discrepancy, free-surface compaction, and an initial dam-break energy spike. The experimental front-reference directory currently contains no digitized reference curve. Those findings should remain explicit when the report is developed into a final validation claim. Comparing sequential, all-pairs OpenMP, and spatial-hash outputs under identical initial conditions and time-step settings would directly connect the performance work to numerical consistency.

The next compute direction described in the presentation is a CUDA implementation, particularly the construction and traversal of the cell structure on the GPU. The present implementation discussed here uses OpenMP CPU threads. A GPU port would need its own measurements and validation, and would build on the demonstrated reduction in candidate-search work and the O(N + C) grid representation.

### 6.7 Data and implementation references

The figures above are embedded from the existing project data folders. The following files provide the numerical and implementation basis for the report:

| Material | Source |
|---|---|
| Earlier sequential particle-count sweep | [sequential scaling results](C:/Users/Aditya/Documents/Projects/SPH_PC/sequential_c/data/scaling_results.json) |
| Earlier sequential 2,030-particle breakdown | [sequential breakdown results](C:/Users/Aditya/Documents/Projects/SPH_PC/sequential_c/data/breakdown_n2000.json) |
| Earlier median-based thread-speedup figure | [thread-speedup CSV](C:/Users/Aditya/Documents/Projects/SPH_PC/parallel_c/data/parallel_speedup.csv) |
| Later comparison of all three implementations | [common scaling results](C:/Users/Aditya/Documents/Projects/SPH_PC/parallel_c/data/scaling_results.json) |
| Later stage timings, including the 30,016-particle comparisons | [common breakdown results](C:/Users/Aditya/Documents/Projects/SPH_PC/parallel_c/data/breakdown_results.json) |
| Scene generation, variant selection, and measurement helpers | [benchmark helpers](C:/Users/Aditya/Documents/Projects/SPH_PC/parallel_c/scaling/bench_common.py) |
| Sequential interaction loops | [sequential physics](C:/Users/Aditya/Documents/Projects/SPH_PC/sequential_c/physics.cpp) |
| OpenMP all-pairs interaction loops | [prehash physics](C:/Users/Aditya/Documents/Projects/SPH_PC/parallel_c/prehash/physics_prehash.cpp) |
| Spatial grid construction and queries | [parallel physics](C:/Users/Aditya/Documents/Projects/SPH_PC/parallel_c/physics.cpp) |
| Integration, timing boundaries, and diagnostic stages | [parallel validation driver](C:/Users/Aditya/Documents/Projects/SPH_PC/parallel_c/validate_main.cpp) |
| Threaded elementwise vector operations | [parallel vector operations](C:/Users/Aditya/Documents/Projects/SPH_PC/parallel_c/backend/linalg.cpp) |
| Existing physical-validation observations | [sequential validation report](C:/Users/Aditya/Documents/Projects/SPH_PC/sequential_c/data/validation_report.md) |

