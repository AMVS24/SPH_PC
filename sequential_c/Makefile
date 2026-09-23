# Build + run the SPH sim on Windows (MinGW-w64, OpenGL 3.3 core).
# Run from the sequential_c/ directory.
#
#   mingw32-make            build only
#   mingw32-make run        build (if needed) and run with 500 particles
#   mingw32-make run N=2000 build (if needed) and run with 2000 particles
#   mingw32-make clean      remove build/
#
# Uses mingw32-make, not `make` -- this toolchain doesn't put a plain `make`
# on PATH. `mingw32-make run` is equivalent to running build_windows.sh
# followed by launching build/sph.exe from Rendering/, but only rebuilds when
# a source file is actually newer than the last build/sph.exe.

CC  := gcc
CXX := g++

RENDERING_DIR := ../Rendering
INC_R := $(RENDERING_DIR)/dependencies/include
LIB_R := $(RENDERING_DIR)/dependencies/library/win64

CXXFLAGS := -std=c++17 -O2 -I. -Ibackend -I$(INC_R)
LDFLAGS  := -L$(LIB_R) -static-libgcc -static-libstdc++
LDLIBS   := -lglfw3 -lgdi32 -lopengl32

SRCS := main.cpp physics.cpp backend/linalg.cpp $(RENDERING_DIR)/Renderer.cpp
GLAD_SRC := $(INC_R)/../src/glad.c

N ?= 500
DATA ?= headless/positions.bin
SPEED ?= 1
STEPS ?= 1000

.PHONY: all run clean timing run-timing headless run-headless run-replay

all: build/sph.exe

build/glad.o: $(GLAD_SRC)
	mkdir -p build
	$(CC) -c $(GLAD_SRC) -I $(INC_R) -o build/glad.o

build/sph.exe: $(SRCS) backend/linalg.h backend/constants.h physics.h $(RENDERING_DIR)/Renderer.h build/glad.o
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(SRCS) build/glad.o $(LDFLAGS) $(LDLIBS) -o build/sph.exe

run: build/sph.exe
	cd $(RENDERING_DIR) && ../sequential_c/build/sph.exe $(N)

timing: timing/build/timing.exe

timing/build/timing.exe: timing/timing_main.cpp physics.cpp backend/linalg.cpp backend/linalg.h backend/constants.h physics.h
	mkdir -p timing/build
	$(CXX) -std=c++17 -O2 -I. -Ibackend timing/timing_main.cpp physics.cpp backend/linalg.cpp -static-libgcc -static-libstdc++ -o timing/build/timing.exe

run-timing: timing/build/timing.exe
	./timing/build/timing.exe $(N)

headless: headless/build/headless.exe headless/build/replay.exe

headless/build/headless.exe: headless/headless_main.cpp physics.cpp backend/linalg.cpp backend/linalg.h backend/constants.h physics.h
	mkdir -p headless/build
	$(CXX) -std=c++17 -O2 -I. -Ibackend headless/headless_main.cpp physics.cpp backend/linalg.cpp -static-libgcc -static-libstdc++ -o headless/build/headless.exe

headless/build/replay.exe: headless/replay_main.cpp backend/linalg.cpp backend/linalg.h backend/constants.h $(RENDERING_DIR)/Renderer.cpp $(RENDERING_DIR)/Renderer.h build/glad.o
	mkdir -p headless/build
	$(CXX) $(CXXFLAGS) headless/replay_main.cpp backend/linalg.cpp $(RENDERING_DIR)/Renderer.cpp build/glad.o $(LDFLAGS) $(LDLIBS) -o headless/build/replay.exe

run-headless: headless/build/headless.exe
	./headless/build/headless.exe $(N) $(STEPS) $(DATA)

run-replay: headless/build/replay.exe
	cd $(RENDERING_DIR) && ../sequential_c/headless/build/replay.exe $(CURDIR)/$(DATA) $(SPEED)

clean:
	rm -rf build
