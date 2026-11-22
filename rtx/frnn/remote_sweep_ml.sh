#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
ARCH="sm_${CC}"
CUDA_HOME="${CUDA_HOME:-}"
if [ -z "$CUDA_HOME" ]; then
    for c in /usr/local/cuda $(ls -d /usr/local/cuda-* 2>/dev/null | sort -V -r); do
        if [ -x "$c/bin/nvcc" ]; then CUDA_HOME="$c"; break; fi
    done
fi
echo "INFO host=$(hostname) date=$(date -Iseconds)"
echo "INFO gpu=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)"
echo "INFO arch=$ARCH cuda_home=$CUDA_HOME nvcc=$("$CUDA_HOME/bin/nvcc" --version | tail -1)"
echo "INFO gpu_busy_before=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1)"
CUDA_HOME="$CUDA_HOME" ARCH="$ARCH" bash build_frnn_ml.sh || { echo "BUILD FAILED"; exit 1; }
B=./build/frnn_ml
N="${N:-1000000}"
run(){ timeout 300 "$B" "$@" || echo "RUNFAIL rc=$? args=$*"; }
for m in "" "--density"; do
    for r in 1 2 4 8 16; do run --dist multires --n "$N" --box 100 --ratio "$r" --budget-s 3 --single-cap-s 5 $m; done
    for al in 0 1 2 2.5 3; do run --dist profile --n "$N" --box 100 --alpha "$al" --budget-s 3 --single-cap-s 5 $m; done
done
for r in 8 16; do run --dist multires --n 4000000 --box 100 --ratio "$r" --budget-s 3 --single-cap-s 5 --density; done
run --dist profile --n 4000000 --box 100 --alpha 2 --budget-s 3 --single-cap-s 5 --density
echo "INFO done=$(date -Iseconds)"
