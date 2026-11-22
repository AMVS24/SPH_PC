#include <optix_device.h>
#include "ablShared.h"

extern "C" __constant__ AblParams params;

#ifndef ABL_MODE
#define ABL_MODE 0
#endif

#if ABL_MODE == 2
#define NPB 8
#else
#define NPB 1
#endif

#define PG2(n) optixGetPayload_##n()
#define PG(n) PG2(n)
#define PS2(n, v) optixSetPayload_##n(v)
#define PS(n, v) PS2(n, v)

#if NPB == 8
#define SC1 8
#define SC2 9
#else
#define SC1 1
#define SC2 2
#endif

static __forceinline__ __device__ void bump1(){
#ifdef ABL_INSTR
    PS(SC1, PG(SC1) + 1);
#endif
}

static __forceinline__ __device__ void bump2(){
#ifdef ABL_INSTR
    PS(SC2, PG(SC2) + 1);
#endif
}

template<bool CHECK>
static __forceinline__ __device__ bool accumulate(unsigned j, float3 q){
    const float3 p = params.pos[j];
    const float r = params.rad[j];
    const float dx = p.x - q.x, dy = p.y - q.y, dz = p.z - q.z;
    const float d2 = dx*dx + dy*dy + dz*dz;
    if(CHECK && !(d2 < r*r)) return false;
#if ABL_MODE == 0
    PS(0, PG(0) + 1);
#elif ABL_MODE == 1
    PS(0, __float_as_uint(__uint_as_float(PG(0)) + ablW(d2, r)));
#else
    float ax = __uint_as_float(PG(0)), ay = __uint_as_float(PG(1)), az = __uint_as_float(PG(2));
    const float3 vi = make_float3(__uint_as_float(PG(3)), __uint_as_float(PG(4)), __uint_as_float(PG(5)));
    ablForce(q, vi, __uint_as_float(PG(6)), __uint_as_float(PG(7)), p, params.vel[j], params.A[j], params.rho[j], r, d2, ax, ay, az);
    PS(0, __float_as_uint(ax)); PS(1, __float_as_uint(ay)); PS(2, __float_as_uint(az));
#endif
    return true;
}

extern "C" __global__ void __raygen__q(){
    const int l = params.offset + (int)optixGetLaunchIndex().x;
    const int s = params.qidx ? params.qidx[l] : l;
    const float3 q = params.pos[s];
    unsigned a0 = 0;
#if NPB == 8
    unsigned a1 = 0, a2 = 0;
    const float3 v = params.vel[s];
    unsigned a3 = __float_as_uint(v.x), a4 = __float_as_uint(v.y), a5 = __float_as_uint(v.z);
    unsigned a6 = __float_as_uint(params.A[s]), a7 = __float_as_uint(params.rho[s]);
#endif
#ifdef ABL_INSTR
    unsigned c1 = 0, c2 = 0;
#endif
    optixTrace(params.handle, make_float3(q.x - params.shift, q.y, q.z), make_float3(1.f, 0.f, 0.f), 0.f, params.tmax, 0.f,
               OptixVisibilityMask(255), params.rayflags, 0, 1, 0, a0
#if NPB == 8
               , a1, a2, a3, a4, a5, a6, a7
#endif
#ifdef ABL_INSTR
               , c1, c2
#endif
               );
#if ABL_MODE == 0
    params.counts[s] = (int)a0;
#elif ABL_MODE == 1
    params.out[s] = __uint_as_float(a0);
#else
    params.out[3*(size_t)s + 0] = __uint_as_float(a0);
    params.out[3*(size_t)s + 1] = __uint_as_float(a1);
    params.out[3*(size_t)s + 2] = __uint_as_float(a2);
#endif
#ifdef ABL_INSTR
    params.calls[s] = (int)c1;
    params.calls2[s] = (int)c2;
#endif
}

extern "C" __global__ void __miss__q(){
}

extern "C" __global__ void __intersection__base(){
    bump1();
    accumulate<true>(optixGetPrimitiveIndex(), optixGetWorldRayOrigin());
}

extern "C" __global__ void __intersection__seg(){
    bump1();
    const float3 q = params.pos[params.offset + (int)optixGetLaunchIndex().x];
    if(!accumulate<true>(optixGetPrimitiveIndex(), q)) bump2();
}

extern "C" __global__ void __intersection__kcap(){
    bump1();
    const float3 q = optixGetWorldRayOrigin();
    const unsigned j = optixGetPrimitiveIndex();
    const float3 p = params.pos[j];
    const float r = params.rad[j];
    const float dx = p.x - q.x, dy = p.y - q.y, dz = p.z - q.z;
    if(dx*dx + dy*dy + dz*dz < r*r){
        const unsigned c = PG(0) + 1;
        PS(0, c);
        if(c >= params.kcap) optixReportIntersection(0.f, 0);
    }
}

extern "C" __global__ void __anyhit__term(){
    optixTerminateRay();
}

extern "C" __global__ void __intersection__empty(){
    bump1();
    PS(0, PG(0) + 1);
}

extern "C" __global__ void __intersection__report(){
    bump1();
    const float3 q = optixGetWorldRayOrigin();
    const unsigned j = optixGetPrimitiveIndex();
    const float3 p = params.pos[j];
    const float r = params.rad[j];
    const float dx = p.x - q.x, dy = p.y - q.y, dz = p.z - q.z;
    if(dx*dx + dy*dy + dz*dz < r*r) optixReportIntersection(0.f, 0);
}

extern "C" __global__ void __anyhit__report(){
    bump2();
    accumulate<false>(optixGetPrimitiveIndex(), optixGetWorldRayOrigin());
    optixIgnoreIntersection();
}

extern "C" __global__ void __anyhit__sphereany(){
    bump1();
    if(!accumulate<true>(optixGetPrimitiveIndex(), optixGetWorldRayOrigin())) bump2();
    optixIgnoreIntersection();
}

extern "C" __global__ void __anyhit__spherefront(){
    bump1();
    const float3 q = params.pos[params.offset + (int)optixGetLaunchIndex().x];
    if(!accumulate<true>(optixGetPrimitiveIndex(), q)) bump2();
    optixIgnoreIntersection();
}

extern "C" __global__ void __anyhit__sphere(){
    bump1();
    if(optixIsBackFaceHit(optixGetHitKind())) accumulate<true>(optixGetPrimitiveIndex(), optixGetWorldRayOrigin());
    else bump2();
    optixIgnoreIntersection();
}
