#include <optix_device.h>
#include "rtxShared.h"

extern "C" __constant__ RtxLaunchParams params;

extern "C" __global__ void __raygen__nbr(){
    const unsigned s = optixGetLaunchIndex().x;
    const float3 origin = params.pos[s];
    unsigned count = 0;
    optixTrace(params.handle, origin, make_float3(1.f, 0.f, 0.f), 0.f, 1e-16f, 0.f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 0, 1, 0, count);
    params.counts[s] = (int)count;
}

extern "C" __global__ void __miss__nbr(){
}

extern "C" __global__ void __intersection__nbr(){
    const float3 q = optixGetWorldRayOrigin();
    const float3 p = params.pos[optixGetPrimitiveIndex()];
    const float dx = p.x - q.x;
    const float dy = p.y - q.y;
    const float dz = p.z - q.z;
    const float r2 = dx*dx + dy*dy + dz*dz;
    if(r2 < params.cutoff2) optixReportIntersection(0.f, 0);
}

extern "C" __global__ void __anyhit__nbr(){
    const unsigned c = optixGetPayload_0();
    if((int)c < params.cap) params.nbr[(size_t)c*(size_t)params.n + optixGetLaunchIndex().x] = (int)optixGetPrimitiveIndex();
    optixSetPayload_0(c + 1);
    optixIgnoreIntersection();
}
