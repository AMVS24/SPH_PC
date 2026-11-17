#pragma once

#include <stddef.h>
#include <backend/linalg.h>

#ifdef BUILDING_RTXPHYSICS
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

GPU_API void gpuDensitySearchOnly();
GPU_API void gpuForcesSearchOnly();
GPU_API size_t gpuAllocatedBytes();

GPU_API void gpuDownloadPositions(vec3* position_array);
GPU_API void gpuDownloadVelocities(vec3* velocity_array);
GPU_API void gpuDownloadDensity(double* density_array);

GPU_API int gpuNeighbourCapacity();
GPU_API long long gpuNeighbourOverflow();
GPU_API long long gpuNeighbourTotal();
GPU_API void gpuDownloadNeighbourCounts(int* counts_in_original_order);
GPU_API void gpuSearchTimes(double* ms_out4);
GPU_API void gpuEnableSearchTiming(int on);

#ifdef __cplusplus
}
#endif
