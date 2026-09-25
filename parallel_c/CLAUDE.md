# parallel_c

## Performance benchmarks
- Before running any performance benchmark (scaling sweeps, timing runs, thread/speedup
  comparisons), stop and ask the user whether the laptop is plugged in and whether energy
  saver is off. Results change with power source and power mode, so don't benchmark until
  they confirm.
- Benchmark only headless builds (build/validate.exe, headless/build/headless.exe,
  timing/build/timing.exe). Never measure performance through build/sph.exe or anything
  that opens the renderer.
- The thread-count speedup sweep is `scaling/speedup_test.py` (drives `build/validate.exe`
  with `--max-steps` and `--threads`); it writes `data/speedup_results.json` and
  `data/speedup_plot.png`.
