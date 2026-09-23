#!/usr/bin/env bash
# Build the SPH sim + renderer on Windows (MinGW-w64, OpenGL 3.3 core).
# Run from the sequential_c/ directory (this script cd's there itself).
#
# Requires a MinGW-w64 gcc/g++ on PATH (e.g. the WinLibs UCRT toolchain).
# Uses the prebuilt GLFW 3.4 static lib at
# ../Rendering/dependencies/library/win64/libglfw3.a (see Rendering/build_windows.sh
# for where that came from).
set -euo pipefail
cd "$(dirname "$0")"

CC=${CC:-gcc}
CXX=${CXX:-g++}

INC_R="../Rendering/dependencies/include"
LIB_R="../Rendering/dependencies/library/win64"

echo "CC=$CC  CXX=$CXX"
mkdir -p build

# glad's loader is C; compile it separately so its linkage matches glad.h.
$CC -c "$INC_R/../src/glad.c" -I "$INC_R" -o build/glad.o

$CXX -std=c++17 -O2 -I. -Ibackend -I"$INC_R" -L"$LIB_R" \
    main.cpp physics.cpp backend/linalg.cpp profiler/Profiler.cpp ../Rendering/Renderer.cpp build/glad.o \
    -static-libgcc -static-libstdc++ -lglfw3 -lgdi32 -lopengl32 \
    -o build/sph.exe

echo "Built build/sph.exe — run it from the sequential_c/ directory:"
echo "    ./build/sph.exe <n>     (n = number of particles, e.g. 500)"
