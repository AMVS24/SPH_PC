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

CXXFLAGS := -std=c++17 -O2 -I. -Ibackend -I$(INC_R) -fopenmp
LDFLAGS  := -L$(LIB_R) -static-libgcc -static-libstdc++ -fopenmp
LDLIBS   := -lglfw3 -lgdi32 -lopengl32

SRCS := main.cpp physics.cpp backend/linalg.cpp $(RENDERING_DIR)/Renderer.cpp
GLAD_SRC := $(INC_R)/../src/glad.c

N ?= 500
DATA ?= headless/positions.bin
SPEED ?= 1
STEPS ?= 1000
THREADS ?=
SAVE ?=
VIDEO ?= headless/replay.mp4
FPS ?= 60

.PHONY: all run clean timing run-timing headless run-headless run-replay run-video validate validate-prehash

all: build/sph.exe

validate: build/validate.exe

build/validate.exe: validate_main.cpp physics.cpp backend/linalg.cpp backend/linalg.h backend/constants.h physics.h
	mkdir -p build
	$(CXX) -std=c++17 -O2 -fopenmp -I. -Ibackend validate_main.cpp physics.cpp backend/linalg.cpp -static-libgcc -static-libstdc++ -o build/validate.exe

validate-prehash: build/validate_prehash.exe

build/validate_prehash.exe: validate_main.cpp prehash/physics_prehash.cpp backend/linalg.cpp backend/linalg.h backend/constants.h physics.h
	mkdir -p build
	$(CXX) -std=c++17 -O2 -fopenmp -DPREHASH -I. -Ibackend validate_main.cpp prehash/physics_prehash.cpp backend/linalg.cpp -static-libgcc -static-libstdc++ -o build/validate_prehash.exe

build/cudaPhysics.dll: cudaPhysics.cu cudaPhysics.h backend/linalg.h build_cuda.bat
	mkdir -p build
	cmd //c "$$(cygpath -w $(CURDIR)/build_cuda.bat)"

build/glad.o: $(GLAD_SRC)
	mkdir -p build
	$(CC) -c $(GLAD_SRC) -I $(INC_R) -o build/glad.o

build/sph.exe: $(SRCS) backend/linalg.h backend/constants.h physics.h cudaPhysics.h $(RENDERING_DIR)/Renderer.h build/glad.o build/cudaPhysics.dll
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(SRCS) build/glad.o build/cudaPhysics.lib $(LDFLAGS) $(LDLIBS) -o build/sph.exe

run: build/sph.exe
	cd $(RENDERING_DIR) && ../gpu/build/sph.exe $(N) $(THREADS)

timing: timing/build/timing.exe

timing/build/timing.exe: timing/timing_main.cpp physics.cpp backend/linalg.cpp backend/linalg.h backend/constants.h physics.h
	mkdir -p timing/build
	$(CXX) -std=c++17 -O2 -fopenmp -I. -Ibackend timing/timing_main.cpp physics.cpp backend/linalg.cpp -static-libgcc -static-libstdc++ -o timing/build/timing.exe

run-timing: timing/build/timing.exe
	./timing/build/timing.exe $(N) $(THREADS)

headless: headless/build/headless.exe headless/build/replay.exe

headless/build/headless.exe: headless/headless_main.cpp physics.cpp backend/linalg.cpp backend/linalg.h backend/constants.h physics.h cudaPhysics.h build/cudaPhysics.dll
	mkdir -p headless/build
	$(CXX) -std=c++17 -O2 -fopenmp -I. -Ibackend headless/headless_main.cpp physics.cpp backend/linalg.cpp build/cudaPhysics.lib -static-libgcc -static-libstdc++ -o headless/build/headless.exe
	cp build/cudaPhysics.dll headless/build/cudaPhysics.dll

headless/build/replay.exe: headless/replay_main.cpp backend/linalg.cpp backend/linalg.h backend/constants.h $(RENDERING_DIR)/Renderer.cpp $(RENDERING_DIR)/Renderer.h build/glad.o
	mkdir -p headless/build
	$(CXX) $(CXXFLAGS) headless/replay_main.cpp backend/linalg.cpp $(RENDERING_DIR)/Renderer.cpp build/glad.o $(LDFLAGS) $(LDLIBS) -o headless/build/replay.exe

run-headless: headless/build/headless.exe
	./headless/build/headless.exe $(N) $(STEPS) $(DATA) $(THREADS) $(if $(SAVE),--save)

run-replay: headless/build/replay.exe
	cd $(RENDERING_DIR) && ../gpu/headless/build/replay.exe $(CURDIR)/$(DATA) $(SPEED)

headless/build/video.exe: headless/video_main.cpp backend/linalg.cpp backend/linalg.h backend/constants.h $(RENDERING_DIR)/Renderer.cpp $(RENDERING_DIR)/Renderer.h build/glad.o
	mkdir -p headless/build
	$(CXX) $(CXXFLAGS) headless/video_main.cpp backend/linalg.cpp $(RENDERING_DIR)/Renderer.cpp build/glad.o $(LDFLAGS) $(LDLIBS) -o headless/build/video.exe

run-video: headless/build/video.exe
	cd $(RENDERING_DIR) && ../gpu/headless/build/video.exe $(CURDIR)/$(DATA) $(CURDIR)/$(VIDEO) $(FPS) $(SPEED)

bench: bench/build/bench.exe

bench/build/bench.exe: bench/bench_main.cpp physics.cpp backend/linalg.cpp backend/linalg.h backend/constants.h physics.h cudaPhysics.h build/cudaPhysics.dll
	mkdir -p bench/build
	$(CXX) -std=c++17 -O2 -fopenmp -I. -Ibackend bench/bench_main.cpp physics.cpp backend/linalg.cpp build/cudaPhysics.lib -static-libgcc -static-libstdc++ -o bench/build/bench.exe
	cp build/cudaPhysics.dll bench/build/cudaPhysics.dll

clean:
	rm -rf build
