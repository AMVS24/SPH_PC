#include <optix_device.h>
#include "costShared.h"

extern "C" __constant__ CostParams params;

extern "C" __global__ void __raygen__cost(){
    const unsigned s = params.offset + optixGetLaunchIndex().x;
    unsigned count = 0;
#ifdef COST_INSTR
    unsigned calls = 0;
    optixTrace(params.handle, params.pos[s], make_float3(1.f, 0.f, 0.f), 0.f, 1e-16f, 0.f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, count, calls);
    params.is_counts[s] = (int)calls;
#else
    optixTrace(params.handle, params.pos[s], make_float3(1.f, 0.f, 0.f), 0.f, 1e-16f, 0.f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, count);
#endif
    params.counts[s] = (int)count;
}

extern "C" __global__ void __miss__cost(){
}

extern "C" __global__ void __intersection__cost(){
    const float3 q = optixGetWorldRayOrigin();
    const unsigned j = optixGetPrimitiveIndex();
    const float3 p = params.pos[j];
    const float r = params.radius[j];
    const float dx = p.x - q.x;
    const float dy = p.y - q.y;
    const float dz = p.z - q.z;
    const float d2 = dx*dx + dy*dy + dz*dz;
#ifdef COST_INSTR
    optixSetPayload_1(optixGetPayload_1() + 1);
#endif
    if(d2 < r*r) optixSetPayload_0(optixGetPayload_0() + 1);
}
