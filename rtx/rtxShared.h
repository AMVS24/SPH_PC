#pragma once

#include <optix.h>
#include <cuda_runtime.h>

struct RtxLaunchParams {
    OptixTraversableHandle handle;
    const float3* pos;
    int* nbr;
    int* counts;
    int n;
    int cap;
    float cutoff2;
};
