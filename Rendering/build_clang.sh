#!/usr/bin/env bash
# Build + run the square demo with clang++ (Apple clang).
# Mirrors the known-working invocation; run from the Rendering/ directory.
set -euo pipefail
cd "$(dirname "$0")"

clang++ -std=c++17 -fcolor-diagnostics -Wall -fansi-escape-codes -g \
    -I./dependencies/include -L./dependencies/library \
    dependencies/library/libglfw.3.dylib \
    -rpath ./dependencies/library \
    main.cpp Renderer.cpp dependencies/src/glad.c \
    -o app \
    -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo -framework CoreFoundation \
    -Wno-deprecated && ./app
