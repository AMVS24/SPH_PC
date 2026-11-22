#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
ARCH="${ARCH:-sm_89}"
NVCC="$CUDA_HOME/bin/nvcc"
mkdir -p build_cost
"$NVCC" -O2 --optix-ir -arch=compute_75 -std=c++17 --use_fast_math -I. -I../third_party/optix/include -o build_cost/cost_programs.ir costPrograms.cu
"$NVCC" -O2 --optix-ir -arch=compute_75 -std=c++17 --use_fast_math -DCOST_INSTR -I. -I../third_party/optix/include -o build_cost/cost_programs_instr.ir costPrograms.cu
"$CUDA_HOME/bin/bin2c" --const --name cost_programs build_cost/cost_programs.ir > build_cost/cost_programs.h
"$CUDA_HOME/bin/bin2c" --const --name cost_programs_instr build_cost/cost_programs_instr.ir > build_cost/cost_programs_instr.h
"$NVCC" -O2 -arch="$ARCH" -std=c++17 -I. -Ibuild_cost -I../third_party/optix/include -o build_cost/cost cost.cu -ldl
"$NVCC" -O2 -arch="$ARCH" -std=c++17 -DCOST_INSTR -I. -Ibuild_cost -I../third_party/optix/include -o build_cost/cost_instr cost.cu -ldl
