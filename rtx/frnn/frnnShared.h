#pragma once

#include <optix.h>
#include <cuda_runtime.h>

struct FrnnParams {
    OptixTraversableHandle handle;
    const float3* pos;
    const float* radius;
    int* counts;
    int* nbr;
    int n;
    int cap;
    int offset;
    int mode;
};
