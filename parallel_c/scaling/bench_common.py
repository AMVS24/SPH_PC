"""
Shared helpers for parallel_c's benchmark scripts (scaling_test.py, breakdown_test.py).

Every measurement uses the same scene and method as sequential_c/scaling/scaling_test.py:
a hydrostatic scene generated at the dx that gives the requested particle count, run
through a headless validate.exe for a fixed number of steps, with ms/step taken as the
mean +/- stdev of 10-step window averages.

Variants:
    sequential  sequential_c/build/validate.exe        (single thread, O(N^2))
    prehash     parallel_c/build/validate_prehash.exe  (OpenMP all-pairs, O(N^2))
    hash        parallel_c/build/validate.exe          (OpenMP + spatial hash grid, O(N))
"""

import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAR_C = os.path.dirname(HERE)
REPO = os.path.dirname(PAR_C)
DATA = os.path.join(PAR_C, "data")

EXES = {
    "sequential": os.path.join(REPO, "sequential_c", "build", "validate.exe"),
    "prehash": os.path.join(PAR_C, "build", "validate_prehash.exe"),
    "hash": os.path.join(PAR_C, "build", "validate.exe"),
}
LABELS = {
    "sequential": "Sequential C++",
    "prehash": "OpenMP, all-pairs (before spatial hash)",
    "hash": "OpenMP + spatial hash",
}
ORDER = {"sequential": 2, "prehash": 2, "hash": 1}

SCENES_PY = os.path.join(REPO, "validation", "scenes.py")
SCENES_DIR = os.path.join(HERE, "scenes")

HYDRO_W = 0.5
HYDRO_FILL_H = 0.3

WINDOW = 10
MIN_STEPS = 2 * WINDOW
MAX_STEPS = 150
RUN_BUDGET_MS = 20000.0
SLOW_STEP_MS = 5000.0

COLOURS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]


def dx_for_particle_count(n_target):
    return math.sqrt(HYDRO_W * HYDRO_FILL_H / n_target)


def generate_scene(n_target):
    os.makedirs(SCENES_DIR, exist_ok=True)
    dx = dx_for_particle_count(n_target)
    proc = subprocess.run(
        [sys.executable, SCENES_PY, "hydrostatic", "--dx", f"{dx:.10f}", "--out", SCENES_DIR],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"scene generation failed for n={n_target}: {proc.stderr}")
    return os.path.join(SCENES_DIR, "hydrostatic.json"), os.path.join(SCENES_DIR, "hydrostatic.init.csv")


def steps_for(est_ms):
    if est_ms is None or est_ms <= 0:
        return MAX_STEPS
    if est_ms > SLOW_STEP_MS:
        return WINDOW
    s = int(RUN_BUDGET_MS / est_ms)
    s -= s % WINDOW
    return max(MIN_STEPS, min(MAX_STEPS, s))


def parse_fields(stdout, prefix):
    fields = {}
    for line in stdout.splitlines():
        if line.startswith(prefix):
            for tok in line[len(prefix):].split():
                if "=" in tok:
                    k, v = tok.split("=")
                    fields[k] = v
    return fields


def run_validate(variant, scene, threads, steps, breakdown=False):
    scene_json, scene_csv = scene
    out_csv = os.path.join(HERE, "_scratch_out.csv")
    out_steps = os.path.join(HERE, "_scratch_steps.csv")
    cmd = [EXES[variant], scene_json, scene_csv, out_csv, out_steps, "--max-steps", str(steps)]
    if variant != "sequential":
        cmd += ["--threads", str(threads)]
    if breakdown:
        cmd.append("--breakdown")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for p in (out_csv, out_steps):
        if os.path.exists(p):
            os.remove(p)
    if proc.returncode != 0:
        return {"error": proc.stderr[-2000:] or f"exit code {proc.returncode}"}

    done = parse_fields(proc.stdout, "done:")
    window = parse_fields(proc.stdout, "window_stats:")
    r = {
        "n_total": int(done["n_fluid"]),
        "steps": int(done["steps"]),
        "ms_per_step": float(window["mean_ms_per_step"]),
        "ms_per_step_stdev": float(window["stdev_ms_per_step"]),
    }
    if breakdown:
        b = parse_fields(proc.stdout, "breakdown:")
        r["stages"] = {k: {"ms": float(b[f"{k}_mean"]), "stdev": float(b[f"{k}_stdev"])} for k in "ABCDEF"}
    return r


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_json(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2)
    os.replace(tmp, path)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    ax.tick_params(axis="both", labelsize=10, width=1.2)
    ax.grid(False)
