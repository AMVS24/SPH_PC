# Build + run the OpenMP-parallel SPH sim (OpenGL 3.3 core) on Windows
# (MinGW-w64) or macOS. Run from the parallel_c/ directory.
#
#   make / mingw32-make            build only
#   make run / mingw32-make run    build (if needed) and run with 500 particles
#   make run N=2000                build (if needed) and run with 2000 particles
#   make run THREADS=4             pin the run to 4 OpenMP threads
#   make clean                     remove build/
#
# On Windows use mingw32-make, not `make` -- the MinGW toolchain doesn't put a
# plain `make` on PATH.
#
# The OS is detected automatically. On macOS this needs a real GCC (Apple
# clang doesn't support OpenMP without extra libomp wiring); it's found
# automatically via Homebrew, or set CXX/CC explicitly, e.g.
# `make CXX=g++-16 CC=gcc-16`.

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
    EXE_EXT := .exe
    LIB_R := $(RENDERING_DIR)/dependencies/library/win64

    CXXFLAGS := -std=c++17 -O2 -I. -Ibackend -I$(INC_R) -fopenmp
    LDFLAGS  := -L$(LIB_R) -static-libgcc -static-libstdc++ -fopenmp
    LDLIBS   := -lglfw3 -lgdi32 -lopengl32
    GLADFLAGS :=
    OMPFLAGS := -fopenmp
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Darwin)
        # ---- macOS ----
        # Apple clang has no OpenMP support out of the box, so this needs a
        # real GCC. Auto-detect a Homebrew one (gcc-13, gcc-16, ...) unless
        # CC/CXX were set explicitly on the command line.
        ifeq ($(origin CXX),default)
            HOMEBREW_GXX := $(firstword $(wildcard /opt/homebrew/bin/g++-* /usr/local/bin/g++-*))
            ifeq ($(HOMEBREW_GXX),)
                $(error No Homebrew GCC found (parallel_c needs OpenMP, which Apple clang doesn't support). Install one with 'brew install gcc', or set CXX/CC explicitly, e.g. 'make CXX=g++-16 CC=gcc-16')
            endif
            CXX := $(HOMEBREW_GXX)
        endif
        ifeq ($(origin CC),default)
            # Match the CXX version, e.g. g++-16 -> gcc-16.
            CC := $(subst g++-,gcc-,$(CXX))
        endif
        EXE_EXT :=
        LIB_R := $(RENDERING_DIR)/dependencies/library

        # GNU gcc on macOS needs -isysroot to find the system headers.
        SYSROOT := $(shell xcrun --show-sdk-path 2>/dev/null)
        ifneq ($(SYSROOT),)
            SYSROOT_FLAG := -isysroot $(SYSROOT)
        endif

        CXXFLAGS := -std=c++17 -O2 $(SYSROOT_FLAG) -I. -Ibackend -I$(INC_R) -Wno-deprecated -fopenmp
        # -rpath so the loader finds libglfw.3.dylib (@rpath install name) at run
        # time; link the dylib directly plus the macOS frameworks GLFW needs.
        LDFLAGS  := $(SYSROOT_FLAG) -Wl,-rpath,$(abspath $(LIB_R)) -fopenmp
        LDLIBS   := $(LIB_R)/libglfw.3.dylib \
                    -framework Cocoa -framework OpenGL -framework IOKit \
                    -framework CoreVideo -framework CoreFoundation
        GLADFLAGS := $(SYSROOT_FLAG)
        OMPFLAGS := -fopenmp
    else
        $(error Unsupported OS '$(UNAME_S)' -- this Makefile targets Windows (MinGW) and macOS)
    endif
endif

BIN := build/sph$(EXE_EXT)

SRCS := main.cpp physics.cpp backend/linalg.cpp $(RENDERING_DIR)/Renderer.cpp

N ?= 500
DATA ?= headless/positions.bin
SPEED ?= 1
STEPS ?= 1000
THREADS ?=
VIDEO ?= headless/replay.mp4
FPS ?= 60

.PHONY: all run clean timing run-timing headless run-headless run-replay run-video validate

all: $(BIN)

validate: build/validate$(EXE_EXT)

build/validate$(EXE_EXT): validate_main.cpp physics.cpp backend/linalg.cpp backend/linalg.h backend/constants.h physics.h
	mkdir -p build
	$(CXX) -std=c++17 -O2 $(OMPFLAGS) -I. -Ibackend validate_main.cpp physics.cpp backend/linalg.cpp -static-libgcc -static-libstdc++ -o build/validate$(EXE_EXT)

build/glad.o: $(GLAD_SRC)
	mkdir -p build
	$(CC) -c $(GLAD_SRC) $(GLADFLAGS) -I $(INC_R) -o build/glad.o

$(BIN): $(SRCS) backend/linalg.h backend/constants.h physics.h $(RENDERING_DIR)/Renderer.h build/glad.o
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(SRCS) build/glad.o $(LDFLAGS) $(LDLIBS) -o $(BIN)

run: $(BIN)
	cd $(RENDERING_DIR) && ../parallel_c/$(BIN) $(N) $(THREADS)

timing: timing/build/timing$(EXE_EXT)

timing/build/timing$(EXE_EXT): timing/timing_main.cpp physics.cpp backend/linalg.cpp backend/linalg.h backend/constants.h physics.h
	mkdir -p timing/build
	$(CXX) -std=c++17 -O2 $(OMPFLAGS) -I. -Ibackend timing/timing_main.cpp physics.cpp backend/linalg.cpp -static-libgcc -static-libstdc++ -o timing/build/timing$(EXE_EXT)

run-timing: timing/build/timing$(EXE_EXT)
	./timing/build/timing$(EXE_EXT) $(N) $(THREADS)

headless: headless/build/headless$(EXE_EXT) headless/build/replay$(EXE_EXT)

headless/build/headless$(EXE_EXT): headless/headless_main.cpp physics.cpp backend/linalg.cpp backend/linalg.h backend/constants.h physics.h
	mkdir -p headless/build
	$(CXX) -std=c++17 -O2 $(OMPFLAGS) -I. -Ibackend headless/headless_main.cpp physics.cpp backend/linalg.cpp -static-libgcc -static-libstdc++ -o headless/build/headless$(EXE_EXT)

headless/build/replay$(EXE_EXT): headless/replay_main.cpp backend/linalg.cpp backend/linalg.h backend/constants.h $(RENDERING_DIR)/Renderer.cpp $(RENDERING_DIR)/Renderer.h build/glad.o
	mkdir -p headless/build
	$(CXX) $(CXXFLAGS) headless/replay_main.cpp backend/linalg.cpp $(RENDERING_DIR)/Renderer.cpp build/glad.o $(LDFLAGS) $(LDLIBS) -o headless/build/replay$(EXE_EXT)

run-headless: headless/build/headless$(EXE_EXT)
	./headless/build/headless$(EXE_EXT) $(N) $(STEPS) $(DATA) $(THREADS)

run-replay: headless/build/replay$(EXE_EXT)
	cd $(RENDERING_DIR) && ../parallel_c/headless/build/replay$(EXE_EXT) $(CURDIR)/$(DATA) $(SPEED)

headless/build/video$(EXE_EXT): headless/video_main.cpp backend/linalg.cpp backend/linalg.h backend/constants.h $(RENDERING_DIR)/Renderer.cpp $(RENDERING_DIR)/Renderer.h build/glad.o
	mkdir -p headless/build
	$(CXX) $(CXXFLAGS) headless/video_main.cpp backend/linalg.cpp $(RENDERING_DIR)/Renderer.cpp build/glad.o $(LDFLAGS) $(LDLIBS) -o headless/build/video$(EXE_EXT)

run-video: headless/build/video$(EXE_EXT)
	cd $(RENDERING_DIR) && ../parallel_c/headless/build/video$(EXE_EXT) $(CURDIR)/$(DATA) $(CURDIR)/$(VIDEO) $(FPS) $(SPEED)

clean:
	rm -rf build
