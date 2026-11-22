#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
ARCH="${ARCH:-sm_89}"
NVCC="$CUDA_HOME/bin/nvcc"
mkdir -p build_abl
for M in 0 1 2; do
    "$NVCC" -O2 --optix-ir -arch=compute_75 -std=c++17 --use_fast_math -DABL_MODE=$M -I. -I../third_party/optix/include -o build_abl/abl_m$M.ir ablPrograms.cu
    "$NVCC" -O2 --optix-ir -arch=compute_75 -std=c++17 --use_fast_math -DABL_MODE=$M -DABL_INSTR -I. -I../third_party/optix/include -o build_abl/abl_m${M}_instr.ir ablPrograms.cu
    "$CUDA_HOME/bin/bin2c" --const --name abl_m$M build_abl/abl_m$M.ir > build_abl/abl_m$M.h
    "$CUDA_HOME/bin/bin2c" --const --name abl_m${M}_instr build_abl/abl_m${M}_instr.ir > build_abl/abl_m${M}_instr.h
done
"$NVCC" -O2 -arch="$ARCH" -std=c++17 --use_fast_math -I. -Ibuild_abl -I../third_party/optix/include -o build_abl/abl abl.cu -ldl
"$NVCC" -O2 -arch="$ARCH" -std=c++17 --use_fast_math -DABL_INSTR -I. -Ibuild_abl -I../third_party/optix/include -o build_abl/abl_instr abl.cu -ldl
