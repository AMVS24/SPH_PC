#include <optix_device.h>
#include "frnnShared.h"

extern "C" __constant__ FrnnParams params;

extern "C" __global__ void __raygen__frnn(){
    const unsigned s = params.offset + optixGetLaunchIndex().x;
    unsigned count = 0;
    optixTrace(params.handle, params.pos[s], make_float3(1.f, 0.f, 0.f), 0.f, 1e-16f, 0.f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 0, 1, 0, count);
    params.counts[s] = (int)count;
}

extern "C" __global__ void __miss__frnn(){
}

extern "C" __global__ void __intersection__frnn(){
    const float3 q = optixGetWorldRayOrigin();
    const unsigned j = optixGetPrimitiveIndex();
    const float3 p = params.pos[j];
    const float r = params.radius[j];
    const float dx = p.x - q.x;
    const float dy = p.y - q.y;
    const float dz = p.z - q.z;
    const float d2 = dx*dx + dy*dy + dz*dz;
    if(!(d2 < r*r)) return;
    if(params.mode == 0) optixSetPayload_0(optixGetPayload_0() + 1);
    else if(params.mode == 2){
        const float h = 0.5f*r;
        const float q = sqrtf(d2)/h;
        const float w = q < 1.f ? 1.f - 1.5f*q*q + 0.75f*q*q*q : 0.25f*(2.f - q)*(2.f - q)*(2.f - q);
        const float m = r*r*r*0.05236f;
        optixSetPayload_0(__float_as_uint(__uint_as_float(optixGetPayload_0()) + m*w/(3.14159265f*h*h*h)));
    }
    else optixReportIntersection(0.f, 0);
}

extern "C" __global__ void __anyhit__frnn(){
    const unsigned c = optixGetPayload_0();
    if((int)c < params.cap){
        const unsigned s = params.offset + optixGetLaunchIndex().x;
        params.nbr[(size_t)c*(size_t)params.n + s] = (int)optixGetPrimitiveIndex();
    }
    optixSetPayload_0(c + 1);
    optixIgnoreIntersection();
}
