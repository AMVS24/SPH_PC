#pragma once

#include <optix.h>
#include <cuda_runtime.h>

struct CostParams {
    OptixTraversableHandle handle;
    const float3* pos;
    const float* radius;
    int* counts;
    int* is_counts;
    int n;
    int offset;
};
