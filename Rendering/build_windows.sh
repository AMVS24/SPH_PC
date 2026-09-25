#!/usr/bin/env bash
# Build the square-rendering demo on Windows (MinGW-w64, OpenGL 3.3 core).
# Run from the Rendering/ directory (this script cd's there itself).
#
# Requires a MinGW-w64 gcc/g++ (e.g. the WinLibs UCRT toolchain) on PATH.
# Uses the prebuilt GLFW 3.4 static lib for MinGW-w64 at
# dependencies/library/win64/libglfw3.a (matches the bundled GLFW 3.4 headers
# in dependencies/include/GLFW — fetched from the official GLFW GitHub release,
# glfw-3.4.bin.WIN64.zip, lib-mingw-w64/libglfw3.a).
set -euo pipefail
cd "$(dirname "$0")"

CC=${CC:-gcc}
CXX=${CXX:-g++}

INC="dependencies/include"
LIB="dependencies/library/win64"

echo "CC=$CC  CXX=$CXX"
mkdir -p build

# glad's loader is C; compile it separately so its linkage matches glad.h.
$CC -c dependencies/src/glad.c -I "$INC" -o build/glad.o

$CXX -std=c++17 -I "$INC" -L "$LIB" \
    main.cpp Renderer.cpp build/glad.o \
    -lglfw3 -lgdi32 -lopengl32 \
    -o build/particles.exe

echo "Built build/particles.exe — run it from the Rendering/ directory:"
echo "    ./build/particles.exe"
