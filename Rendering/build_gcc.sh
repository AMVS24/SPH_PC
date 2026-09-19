#!/usr/bin/env bash
# Build + run the square demo with GNU gcc (Homebrew gcc-16/g++-16).
# Same result as build_clang.sh, but with the flag spellings GNU gcc wants:
#   -fdiagnostics-color   instead of clang's -fcolor-diagnostics/-fansi-escape-codes
#   -Wl,-rpath,...        instead of the bare -rpath clang accepts
#   -isysroot ...         so GNU gcc can find the macOS SDK system headers
# Run from the Rendering/ directory.
set -euo pipefail
cd "$(dirname "$0")"

g++-16 -std=c++17 -Wall -g -isysroot "$(xcrun --show-sdk-path)" -I./dependencies/include -L./dependencies/library dependencies/library/libglfw.3.dylib -Wl,-rpath,./dependencies/library main.cpp Renderer.cpp dependencies/src/glad.c -o app -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo -framework CoreFoundation -Wno-deprecated && ./app
