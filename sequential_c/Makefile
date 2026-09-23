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

SRCS := main.cpp physics.cpp backend/linalg.cpp profiler/Profiler.cpp $(RENDERING_DIR)/Renderer.cpp
GLAD_SRC := $(INC_R)/../src/glad.c

N ?= 500

.PHONY: all run clean

all: build/sph.exe

build/glad.o: $(GLAD_SRC)
	mkdir -p build
	$(CC) -c $(GLAD_SRC) -I $(INC_R) -o build/glad.o

build/sph.exe: $(SRCS) backend/linalg.h backend/constants.h physics.h profiler/Profiler.h $(RENDERING_DIR)/Renderer.h build/glad.o
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(SRCS) build/glad.o $(LDFLAGS) $(LDLIBS) -o build/sph.exe

run: build/sph.exe
	cd $(RENDERING_DIR) && ../sequential_c/build/sph.exe $(N)

clean:
	rm -rf build
