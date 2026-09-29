#pragma once

#define PI 3.14159265358979323846
constexpr int DIM = 20;
inline double H = 2;
inline double PRESSURE_MULTIPLIER = 1000000.0;
inline double TARGET_DENSITY = 0.3;
inline double PARTICLE_MASS = 1.0;
inline double GRAVITY = 500.0;      // magnitude; applied downward (-y) in update_forces
inline double VISCOSITY_ALPHA = 0.1;

// Rendered particle half-extent, as a fraction of the smoothing radius H --
// not a fixed NDC number. Ratio taken from sequential_py/naive_py.py, which
// draws each particle at radius=3 against smoothing_radius=10.0 (0.3x) and
// looks right for a fluid rather than a sparse point cloud; main.cpp turns
// this into an actual NDC size via world_to_ndc's own scale factor.
constexpr double PARTICLE_SIZE_TO_H = 5;

// World bounds, in the same coordinates as particle positions. Single source
// of truth for both the boundary condition (physics.cpp) and the world->NDC
// mapping (main.cpp).
constexpr double WORLD_MIN_X = 0.0;
constexpr double WORLD_MIN_Y = 0.0;
constexpr double WORLD_MAX_X = 1000.0;
constexpr double WORLD_MAX_Y = 1000.0;

// Half-width/height of the box particles spawn in, centered on the world
// centre (see main.cpp). World units.
constexpr double SPAWN_HALF_EXTENT = 230.0;

constexpr double WALL_RESTITUTION = 0.4;

constexpr double BOUNDARY_WIDTH = 500.0;
constexpr double BOUNDARY_HEIGHT = 500.0;
inline int grid_w = (int)(BOUNDARY_WIDTH/(2*H));
inline int grid_h = (int)(BOUNDARY_HEIGHT/(2*H));

constexpr double PARTICLE_NDC_PER_WORLD_UNIT = 0.004;
