#!/usr/bin/env bash
# Build the square-rendering demo (macOS, OpenGL 3.3 core).
# Run shaders relative to this directory, so launch the binary from Rendering/.
#
# Compiler: override with CC / CXX, otherwise this prefers a GNU gcc/g++ if one
# is installed (e.g. Homebrew's gcc-16) and falls back to cc/c++ (Apple clang).
set -euo pipefail
cd "$(dirname "$0")"

# Pick a C and C++ compiler.
if [ -z "${CC:-}" ]; then
    CC=$(ls /opt/homebrew/bin/gcc-* /usr/local/bin/gcc-* 2>/dev/null | grep -E 'gcc-[0-9]+$' | sort -V | tail -1 || true)
    CC=${CC:-cc}
fi
if [ -z "${CXX:-}" ]; then
    CXX=$(ls /opt/homebrew/bin/g++-* /usr/local/bin/g++-* 2>/dev/null | grep -E 'g\+\+-[0-9]+$' | sort -V | tail -1 || true)
    CXX=${CXX:-c++}
fi

# GNU gcc on macOS needs the SDK path to find system headers; clang finds it
# itself. Passing -isysroot is harmless for clang, so add it whenever available.
SYSROOT=""
if command -v xcrun >/dev/null 2>&1; then
    SYSROOT="-isysroot $(xcrun --show-sdk-path)"
fi

INC="dependencies/include"
GLFW="dependencies/library/libglfw.3.dylib"

echo "CC=$CC  CXX=$CXX"
mkdir -p build

# glad's loader is C; compile it separately so its linkage matches glad.h.
$CC $SYSROOT -c dependencies/src/glad.c -I "$INC" -o build/glad.o

$CXX $SYSROOT -std=c++17 -I "$INC" \
    main.cpp Renderer.cpp build/glad.o \
    "$GLFW" \
    -framework Cocoa -framework OpenGL -framework IOKit -framework CoreVideo \
    -Wl,-rpath,@executable_path/../dependencies/library \
    -o build/particles

echo "Built build/particles — run it from the Rendering/ directory:"
echo "    ./build/particles"
