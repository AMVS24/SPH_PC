# sequential_c

## Performance benchmarks
- Before running any performance benchmark (scaling sweeps, timing runs, thread/speedup
  comparisons), stop and ask the user whether the laptop is plugged in and whether energy
  saver is off. Results change with power source and power mode, so don't benchmark until
  they confirm.
- Benchmark only headless builds (build/validate.exe, headless/build/headless.exe,
  timing/build/timing.exe). Never measure performance through build/sph.exe or anything
  that opens the renderer.
- The particle-count scaling sweep is `scaling/scaling_test.py` (drives `build/validate.exe`
  with `--max-steps`); it writes `data/scaling_results.json` and `data/scaling_plot.png`.
  `scaling/breakdown_test.py` gives the per-stage breakdown at one particle count.
