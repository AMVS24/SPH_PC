#pragma once

#include <backend/linalg.h>

#ifdef BUILDING_CUDAPHYSICS
#define GPU_API __declspec(dllexport)
#else
#define GPU_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double H, particle_mass, target_density, pressure_multiplier;
    double gravity, viscosity_alpha, wall_restitution;
    double cell_size;
    int grid_w, grid_h, grid_d;
    double grid_min_x, grid_min_y, grid_min_z;
    double min_x, max_x, min_y, max_y, min_z, max_z;
    double kernel_norm, grad_norm, c0;
} GpuParams;

GPU_API void gpuInit(int n, int n_fluid, const vec3* position_array, const vec3* velocity_array, const GpuParams* params);
GPU_API void gpuSetParams(const GpuParams* params);
GPU_API void gpuFree();

GPU_API void gpuDrift(double dt);
GPU_API void gpuEnforceBoundary();
GPU_API void gpuBuildGrid();
GPU_API void gpuDensity();
GPU_API void gpuPressure();
GPU_API void gpuForces();
GPU_API void gpuKick(double dt);
GPU_API void gpuSync();

GPU_API void gpuDownloadPositions(vec3* position_array);
GPU_API void gpuDownloadVelocities(vec3* velocity_array);
GPU_API void gpuDownloadDensity(double* density_array);

#ifdef __cplusplus
}
#endif
