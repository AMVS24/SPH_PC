# Build + run the SPH sim (OpenGL 3.3 core) on Windows (MinGW-w64) or macOS.
# Run from the sequential_c/ directory.
#
#   make / mingw32-make            build only
#   make run / mingw32-make run    build (if needed) and run with 500 particles
#   make run N=2000                build (if needed) and run with 2000 particles
#   make run QUIET=1               run without console output (profiler CSV still written)
#   make clean                     remove build/
#
# On Windows use mingw32-make, not `make` -- the MinGW toolchain doesn't put a
# plain `make` on PATH. `mingw32-make run` is equivalent to running
# build_windows.sh followed by launching build/sph.exe from Rendering/, but
# only rebuilds when a source file is actually newer than the last binary.
#
# The OS is detected automatically. CC/CXX can be overridden on the command
# line (e.g. `make CXX=g++-16 CC=gcc-16` on macOS).

RENDERING_DIR := ../Rendering
INC_R := $(RENDERING_DIR)/dependencies/include
GLAD_SRC := $(RENDERING_DIR)/dependencies/src/glad.c

ifeq ($(OS),Windows_NT)
    # ---- Windows (MinGW-w64) ----
    # make's built-in defaults (CC=cc, CXX=g++) are replaced unless the user
    # set them explicitly; plain `?=` wouldn't work since CC is always defined.
    ifeq ($(origin CC),default)
        CC := gcc
    endif
    ifeq ($(origin CXX),default)
        CXX := g++
    endif
    BIN := build/sph.exe
    LIB_R := $(RENDERING_DIR)/dependencies/library/win64

    CXXFLAGS := -std=c++17 -O2 -I. -Ibackend -I$(INC_R)
    LDFLAGS  := -L$(LIB_R) -static-libgcc -static-libstdc++
    LDLIBS   := -lglfw3 -lgdi32 -lopengl32
    GLADFLAGS :=
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Darwin)
        # ---- macOS (Apple clang by default; override with CXX=g++-16 CC=gcc-16) ----
        ifeq ($(origin CC),default)
            CC := cc
        endif
        ifeq ($(origin CXX),default)
            CXX := clang++
        endif
        BIN := build/sph
        LIB_R := $(RENDERING_DIR)/dependencies/library

        # GNU gcc on macOS needs -isysroot to find the system headers; clang finds
        # the SDK itself and -isysroot is harmless for it. Add it when available.
        SYSROOT := $(shell xcrun --show-sdk-path 2>/dev/null)
        ifneq ($(SYSROOT),)
            SYSROOT_FLAG := -isysroot $(SYSROOT)
        endif

        CXXFLAGS := -std=c++17 -O2 $(SYSROOT_FLAG) -I. -Ibackend -I$(INC_R) -Wno-deprecated
        # -rpath so the loader finds libglfw.3.dylib (@rpath install name) at run
        # time; link the dylib directly plus the macOS frameworks GLFW needs.
        LDFLAGS  := $(SYSROOT_FLAG) -Wl,-rpath,$(abspath $(LIB_R))
        LDLIBS   := $(LIB_R)/libglfw.3.dylib \
                    -framework Cocoa -framework OpenGL -framework IOKit \
                    -framework CoreVideo -framework CoreFoundation
        GLADFLAGS := $(SYSROOT_FLAG)
    else
        $(error Unsupported OS '$(UNAME_S)' -- this Makefile targets Windows (MinGW) and macOS)
    endif
endif

SRCS := main.cpp physics.cpp backend/linalg.cpp profiler/Profiler.cpp $(RENDERING_DIR)/Renderer.cpp

N ?= 500
QUIET ?= 0
ifeq ($(QUIET),1)
    RUN_FLAGS := --quiet
endif

.PHONY: all run clean

all: $(BIN)

build/glad.o: $(GLAD_SRC)
	mkdir -p build
	$(CC) -c $(GLAD_SRC) $(GLADFLAGS) -I $(INC_R) -o build/glad.o

$(BIN): $(SRCS) backend/linalg.h backend/constants.h physics.h profiler/Profiler.h $(RENDERING_DIR)/Renderer.h build/glad.o
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(SRCS) build/glad.o $(LDFLAGS) $(LDLIBS) -o $(BIN)

run: $(BIN)
	cd $(RENDERING_DIR) && ../sequential_c/$(BIN) $(N) $(RUN_FLAGS)

clean:
	rm -rf build
