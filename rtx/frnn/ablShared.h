#pragma once

#include <optix.h>
#include <cuda_runtime.h>

struct AblParams {
    OptixTraversableHandle handle;
    const float3* pos;
    const float* rad;
    const float3* vel;
    const float* rho;
    const float* A;
    const int* qidx;
    int* counts;
    float* out;
    int* calls;
    int* calls2;
    int n;
    int offset;
    float tmax;
    float shift;
    unsigned int kcap;
    unsigned int rayflags;
};

#define ABL_PI 3.14159265f
#define ABL_ALPHA 1.0f
#define ABL_C0 10.0f

__device__ __forceinline__ float ablW(float d2, float r){
    const float h = 0.5f*r;
    const float q = sqrtf(d2)/h;
    const float w = q < 1.f ? 1.f - 1.5f*q*q + 0.75f*q*q*q : 0.25f*(2.f - q)*(2.f - q)*(2.f - q);
    return w/(ABL_PI*h*h*h);
}

__device__ __forceinline__ void ablForce(float3 xi, float3 vi, float Ai, float rhoi,
                                         float3 xj, float3 vj, float Aj, float rhoj, float r, float d2,
                                         float& ax, float& ay, float& az){
    if(d2 <= 0.f) return;
    const float h = 0.5f*r;
    const float d = sqrtf(d2);
    const float q = d/h;
    const float dw = (q < 1.f ? (-3.f*q + 2.25f*q*q) : (-0.75f*(2.f - q)*(2.f - q)))/(ABL_PI*h*h*h*h);
    const float rx = xi.x - xj.x, ry = xi.y - xj.y, rz = xi.z - xj.z;
    const float gx = dw*rx/d, gy = dw*ry/d, gz = dw*rz/d;
    float c = Ai + Aj;
    const float vr = (vi.x - vj.x)*rx + (vi.y - vj.y)*ry + (vi.z - vj.z)*rz;
    if(vr < 0.f){
        const float mu = h*vr/(d2 + 0.01f*h*h);
        c += -ABL_ALPHA*ABL_C0*mu/(0.5f*(rhoi + rhoj));
    }
    ax -= c*gx;
    ay -= c*gy;
    az -= c*gz;
}
