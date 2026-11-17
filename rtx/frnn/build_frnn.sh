#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
ARCH="${ARCH:-sm_89}"
NVCC="$CUDA_HOME/bin/nvcc"
mkdir -p build
"$NVCC" -O2 --optix-ir -arch=compute_75 -std=c++17 --use_fast_math -I. -I../third_party/optix/include -o build/frnn_programs.ir frnnPrograms.cu
"$CUDA_HOME/bin/bin2c" --const --name frnn_programs build/frnn_programs.ir > build/frnn_programs.h
"$NVCC" -O2 -arch="$ARCH" -std=c++17 -I. -Ibuild -I../third_party/optix/include -o build/frnn frnn.cu -ldl
