#define NOMINMAX
#define BUILDING_RTXPHYSICS
#include "rtxPhysics.h"
#include "rtxShared.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "rtx_programs.h"

#ifndef RTX_K
#define RTX_K 128
#endif

typedef float real;
#define R(x) ((real)(x))

#define CK(call) do{ cudaError_t e_ = (call); if(e_ != cudaSuccess){ fprintf(stderr, "CUDA error: %s (%s:%d): %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } }while(0)

static char g_log[8192];
static size_t g_log_size = 0;

#define OK(call) do{ g_log_size = sizeof(g_log); OptixResult r_ = (call); if(r_ != OPTIX_SUCCESS){ fprintf(stderr, "OptiX error: %s (%s:%d): %s\n%s\n", #call, __FILE__, __LINE__, optixGetErrorName(r_), g_log); exit(1); } }while(0)

struct rvec3 { real x, y, z; };

struct KParams {
    real H, particle_mass, target_density, pressure_multiplier;
    real gravity, viscosity_alpha, wall_restitution, cell_size;
    real kernel_norm, grad_norm, c0, cutoff2;
    real grid_min_x, grid_min_y, grid_min_z;
    real min_x, max_x, min_y, max_y, min_z, max_z;
    int grid_w, grid_h, grid_d;
};

struct SbtRecord { __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };

static const int BLOCK = 256;
static const double PI_D = 3.14159265358979323846;
static const int NBR_CAP = RTX_K;

static GpuParams g_p;
static KParams g_kp;
static int g_n = 0, g_n_fluid = 0, g_sort_bits = 1;
static rvec3 *d_pos = nullptr, *d_vel = nullptr, *d_pos2 = nullptr, *d_vel2 = nullptr, *d_acc = nullptr;
static real *d_density = nullptr, *d_pressure = nullptr;
static int *d_id = nullptr, *d_id2 = nullptr, *d_iota = nullptr, *d_perm = nullptr;
static unsigned int *d_keys = nullptr, *d_keys_sorted = nullptr;
static int *d_nbr = nullptr, *d_counts = nullptr;
static unsigned long long* d_stat = nullptr;
static OptixAabb* d_aabb = nullptr;
static void *d_sort_temp = nullptr, *d_gas_temp = nullptr, *d_gas_out = nullptr, *d_params = nullptr;
static size_t g_sort_temp_bytes = 0, g_gas_temp_bytes = 0, g_gas_out_bytes = 0;
static size_t g_alloc_bytes = 0;

static OptixDeviceContext g_ctx = nullptr;
static OptixModule g_module = nullptr;
static OptixProgramGroup g_pg_raygen = nullptr, g_pg_miss = nullptr, g_pg_hit = nullptr;
static OptixPipeline g_pipeline = nullptr;
static OptixShaderBindingTable g_sbt = {};
static void* d_sbt_records = nullptr;
static OptixBuildInput g_gas_input = {};
static CUdeviceptr g_aabb_ptr = 0;
static unsigned int g_gas_flags[1] = { OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL };
static OptixTraversableHandle g_gas_handle = 0;

static bool g_timing = false;
static cudaEvent_t g_ev[4];
static bool g_ev_ready = false;

#define ALLOC(ptr, bytes) do{ CK(cudaMalloc(&(ptr), (bytes))); g_alloc_bytes += (bytes); }while(0)

static int blocksFor(int count){ return (count + BLOCK - 1) / BLOCK; }

static void stamp(int k){
    if(g_timing) CK(cudaEventRecord(g_ev[k], 0));
}

__global__ void iotaKernel(int* out, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) out[i] = i;
}

__global__ void driftKernel(rvec3* pos, const rvec3* vel, int n_fluid, real dt){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n_fluid) return;
    pos[i].x += vel[i].x*dt;
    pos[i].y += vel[i].y*dt;
    pos[i].z += vel[i].z*dt;
}

__device__ void bounceAxis(real& x, real& v, real lo, real hi, real restitution){
    if(x < lo){
        x = lo;
        v = -v*restitution;
    }
    else if(x > hi){
        x = hi;
        v = -v*restitution;
    }
}

__global__ void boundaryKernel(KParams p, rvec3* pos, rvec3* vel, int n_fluid){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n_fluid) return;
    bounceAxis(pos[i].x, vel[i].x, p.min_x, p.max_x, p.wall_restitution);
    bounceAxis(pos[i].y, vel[i].y, p.min_y, p.max_y, p.wall_restitution);
    bounceAxis(pos[i].z, vel[i].z, p.min_z, p.max_z, p.wall_restitution);
}

__device__ unsigned int expandBits(unsigned int v){
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v << 8))  & 0x0300F00Fu;
    v = (v | (v << 4))  & 0x030C30C3u;
    v = (v | (v << 2))  & 0x09249249u;
    return v;
}

__global__ void mortonKernel(KParams p, const rvec3* pos, unsigned int* keys, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;
    int cx = (int)((pos[i].x - p.grid_min_x)/p.cell_size);
    int cy = (int)((pos[i].y - p.grid_min_y)/p.cell_size);
    int cz = (int)((pos[i].z - p.grid_min_z)/p.cell_size);
    cx = min(max(cx, 0), min(p.grid_w - 1, 1023));
    cy = min(max(cy, 0), min(p.grid_h - 1, 1023));
    cz = min(max(cz, 0), min(p.grid_d - 1, 1023));
    keys[i] = expandBits((unsigned int)cx) | (expandBits((unsigned int)cy) << 1) | (expandBits((unsigned int)cz) << 2);
}

__global__ void gatherKernel(const int* perm, const rvec3* pos_in, const rvec3* vel_in, const int* id_in, rvec3* pos_out, rvec3* vel_out, int* id_out, int n){
    int s = blockIdx.x*blockDim.x + threadIdx.x;
    if(s >= n) return;
    int o = perm[s];
    pos_out[s] = pos_in[o];
    vel_out[s] = vel_in[o];
    id_out[s] = id_in[o];
}

__global__ void aabbKernel(const rvec3* pos, OptixAabb* aabb, real half, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;
    OptixAabb b;
    b.minX = pos[i].x - half; b.minY = pos[i].y - half; b.minZ = pos[i].z - half;
    b.maxX = pos[i].x + half; b.maxY = pos[i].y + half; b.maxZ = pos[i].z + half;
    aabb[i] = b;
}

__device__ real kernelW(real q, real norm){
    if(q < R(1)){
        real q2 = q*q;
        return norm * (R(1) - R(1.5) * q2 + R(0.75) * q2*q);
    }
    else if(q < R(2)){
        return norm * (R(0.25) * (R(2) - q) * (R(2) - q) * (R(2) - q));
    }
    return R(0);
}

__global__ void densityListKernel(KParams p, const rvec3* pos, const int* nbr, const int* counts, real* density, int n, int cap){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;
    int cnt = min(counts[i], cap);
    real xi = pos[i].x, yi = pos[i].y, zi = pos[i].z;
    real sum = R(0);
    for(int c = 0; c < cnt; c++){
        int j = nbr[(size_t)c*(size_t)n + i];
        real rx = pos[j].x - xi;
        real ry = pos[j].y - yi;
        real rz = pos[j].z - zi;
        real r2 = rx*rx + ry*ry + rz*rz;
        sum += p.particle_mass*kernelW(sqrt(r2)/p.H, p.kernel_norm);
    }
    density[i] = sum;
}

__global__ void pressureKernel(KParams p, const real* density, real* pressure, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;
    pressure[i] = (density[i] - p.target_density)*p.pressure_multiplier;
}

__global__ void forceListKernel(KParams p, const rvec3* pos, const rvec3* vel, const real* density, const real* pressure,
                                const int* nbr, const int* counts, rvec3* acc, int n, int cap){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;
    int cnt = min(counts[i], cap);
    real xi = pos[i].x, yi = pos[i].y, zi = pos[i].z;
    real vxi = vel[i].x, vyi = vel[i].y, vzi = vel[i].z;
    real rho_i = density[i], p_i = pressure[i];
    real ax = R(0), ay = R(0), az = R(0);

    for(int c = 0; c < cnt; c++){
        int j = nbr[(size_t)c*(size_t)n + i];
        if(i == j) continue;

        real rx = pos[j].x - xi;
        real ry = pos[j].y - yi;
        real rz = pos[j].z - zi;
        real r2 = rx*rx + ry*ry + rz*rz;
        if(r2 == R(0) || r2 >= p.cutoff2) continue;
        real rho_j = density[j];
        if(rho_j == R(0)) continue;

        real norm = sqrt(r2);
        real q = norm/p.H;
        real coef;
        if(q < R(1)) coef = p.grad_norm * (R(-3) * q + R(2.25) * q * q);
        else         coef = p.grad_norm * (R(-3) * (R(2) - q) * (R(2) - q) / R(4));
        real gx = -(rx/norm)*coef;
        real gy = -(ry/norm)*coef;
        real gz = -(rz/norm)*coef;

        real pterm = p_i/(rho_i*rho_i) + pressure[j]/(rho_j*rho_j);
        ax -= pterm*gx;
        ay -= pterm*gy;
        az -= pterm*gz;

        real dot_prod = rx*(vel[j].x - vxi) + ry*(vel[j].y - vyi) + rz*(vel[j].z - vzi);
        if(dot_prod < R(0)){
            real mu = p.H*dot_prod/(norm*norm + R(0.0001)*p.H*p.H);
            real avg_density = (rho_i + rho_j)*R(0.5);
            real visc = (-p.viscosity_alpha*p.c0*mu)/(avg_density + R(1e-9));
            ax -= visc*gx;
            ay -= visc*gy;
            az -= visc*gz;
        }
    }
    ay -= p.gravity;
    acc[i].x = ax;
    acc[i].y = ay;
    acc[i].z = az;
}

__global__ void kickKernel(rvec3* vel, const rvec3* acc, int n_fluid, real dt){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n_fluid) return;
    vel[i].x += acc[i].x*dt;
    vel[i].y += acc[i].y*dt;
    vel[i].z += acc[i].z*dt;
}

__global__ void statKernel(const int* counts, int n, int cap, unsigned long long* out){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;
    int c = counts[i];
    atomicAdd(&out[0], (unsigned long long)c);
    if(c > cap) atomicAdd(&out[1], (unsigned long long)(c - cap));
}

static std::vector<rvec3> toReal(const vec3* a, int n){
    std::vector<rvec3> out(n);
    for(int i = 0; i < n; i++) out[i] = rvec3{R(a[i].x), R(a[i].y), R(a[i].z)};
    return out;
}

static void fromRealById(const rvec3* d_src, vec3* out, int n){
    std::vector<rvec3> tmp(n);
    std::vector<int> ids(n);
    CK(cudaMemcpy(tmp.data(), d_src, n*sizeof(rvec3), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ids.data(), d_id, n*sizeof(int), cudaMemcpyDeviceToHost));
    for(int s = 0; s < n; s++) out[ids[s]] = vec3{(double)tmp[s].x, (double)tmp[s].y, (double)tmp[s].z};
}

static void logCb(unsigned int level, const char* tag, const char* msg, void*){
    fprintf(stderr, "[optix %u][%s] %s\n", level, tag, msg);
}

static void createPipelineOnce(){
    if(g_ctx) return;
    CK(cudaFree(0));
    OK(optixInit());

    OptixDeviceContextOptions co = {};
    co.logCallbackFunction = &logCb;
    co.logCallbackLevel = 2;
    if(getenv("RTX_VALIDATE")) co.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
    OK(optixDeviceContextCreate(0, &co, &g_ctx));

    OptixModuleCompileOptions mco = {};
    mco.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pco = {};
    pco.usesMotionBlur = 0;
    pco.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pco.numPayloadValues = 1;
    pco.numAttributeValues = 2;
    pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pco.pipelineLaunchParamsVariableName = "params";
    pco.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    OK(optixModuleCreate(g_ctx, &mco, &pco, (const char*)rtx_programs, sizeof(rtx_programs), g_log, &g_log_size, &g_module));

    OptixProgramGroupOptions pgo = {};
    OptixProgramGroupDesc d = {};

    d.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    d.raygen.module = g_module;
    d.raygen.entryFunctionName = "__raygen__nbr";
    OK(optixProgramGroupCreate(g_ctx, &d, 1, &pgo, g_log, &g_log_size, &g_pg_raygen));

    d = {};
    d.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    d.miss.module = g_module;
    d.miss.entryFunctionName = "__miss__nbr";
    OK(optixProgramGroupCreate(g_ctx, &d, 1, &pgo, g_log, &g_log_size, &g_pg_miss));

    d = {};
    d.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    d.hitgroup.moduleIS = g_module;
    d.hitgroup.entryFunctionNameIS = "__intersection__nbr";
    d.hitgroup.moduleAH = g_module;
    d.hitgroup.entryFunctionNameAH = "__anyhit__nbr";
    OK(optixProgramGroupCreate(g_ctx, &d, 1, &pgo, g_log, &g_log_size, &g_pg_hit));

    OptixProgramGroup groups[3] = { g_pg_raygen, g_pg_miss, g_pg_hit };
    OptixPipelineLinkOptions plo = {};
    plo.maxTraceDepth = 1;
    OK(optixPipelineCreate(g_ctx, &pco, &plo, groups, 3, g_log, &g_log_size, &g_pipeline));

    SbtRecord recs[3];
    OK(optixSbtRecordPackHeader(g_pg_raygen, &recs[0]));
    OK(optixSbtRecordPackHeader(g_pg_miss, &recs[1]));
    OK(optixSbtRecordPackHeader(g_pg_hit, &recs[2]));
    CK(cudaMalloc(&d_sbt_records, sizeof(recs)));
    CK(cudaMemcpy(d_sbt_records, recs, sizeof(recs), cudaMemcpyHostToDevice));
    g_sbt = {};
    g_sbt.raygenRecord = (CUdeviceptr)d_sbt_records;
    g_sbt.missRecordBase = (CUdeviceptr)d_sbt_records + sizeof(SbtRecord);
    g_sbt.missRecordStrideInBytes = sizeof(SbtRecord);
    g_sbt.missRecordCount = 1;
    g_sbt.hitgroupRecordBase = (CUdeviceptr)d_sbt_records + 2*sizeof(SbtRecord);
    g_sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord);
    g_sbt.hitgroupRecordCount = 1;

    if(!g_ev_ready){
        for(int k = 0; k < 4; k++) CK(cudaEventCreate(&g_ev[k]));
        g_ev_ready = true;
    }
}

static OptixAccelBuildOptions gasOptions(){
    OptixAccelBuildOptions ao = {};
    ao.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
    ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    return ao;
}

void gpuSetParams(const GpuParams* params){
    g_p = *params;
    g_p.kernel_norm = 1.0/(PI_D*g_p.H*g_p.H*g_p.H);
    g_p.grad_norm = 1.0/(PI_D*g_p.H*g_p.H*g_p.H*g_p.H);
    g_p.c0 = sqrt(g_p.pressure_multiplier);

    g_kp.H = R(g_p.H);
    g_kp.particle_mass = R(g_p.particle_mass);
    g_kp.target_density = R(g_p.target_density);
    g_kp.pressure_multiplier = R(g_p.pressure_multiplier);
    g_kp.gravity = R(g_p.gravity);
    g_kp.viscosity_alpha = R(g_p.viscosity_alpha);
    g_kp.wall_restitution = R(g_p.wall_restitution);
    g_kp.cell_size = R(g_p.cell_size);
    g_kp.kernel_norm = R(g_p.kernel_norm);
    g_kp.grad_norm = R(g_p.grad_norm);
    g_kp.c0 = R(g_p.c0);
    g_kp.cutoff2 = R(4.0*g_p.H*g_p.H);
    g_kp.grid_min_x = R(g_p.grid_min_x);
    g_kp.grid_min_y = R(g_p.grid_min_y);
    g_kp.grid_min_z = R(g_p.grid_min_z);
    g_kp.min_x = R(g_p.min_x); g_kp.max_x = R(g_p.max_x);
    g_kp.min_y = R(g_p.min_y); g_kp.max_y = R(g_p.max_y);
    g_kp.min_z = R(g_p.min_z); g_kp.max_z = R(g_p.max_z);
    g_kp.grid_w = g_p.grid_w;
    g_kp.grid_h = g_p.grid_h;
    g_kp.grid_d = g_p.grid_d;
}

void gpuFree(){
    cudaFree(d_pos); cudaFree(d_vel); cudaFree(d_pos2); cudaFree(d_vel2); cudaFree(d_acc);
    cudaFree(d_density); cudaFree(d_pressure);
    cudaFree(d_id); cudaFree(d_id2); cudaFree(d_iota); cudaFree(d_perm);
    cudaFree(d_keys); cudaFree(d_keys_sorted);
    cudaFree(d_nbr); cudaFree(d_counts); cudaFree(d_stat); cudaFree(d_aabb);
    cudaFree(d_sort_temp); cudaFree(d_gas_temp); cudaFree(d_gas_out); cudaFree(d_params);
    d_pos = d_vel = d_pos2 = d_vel2 = d_acc = nullptr;
    d_density = nullptr; d_pressure = nullptr;
    d_id = d_id2 = d_iota = d_perm = nullptr;
    d_keys = d_keys_sorted = nullptr;
    d_nbr = d_counts = nullptr; d_stat = nullptr; d_aabb = nullptr;
    d_sort_temp = d_gas_temp = d_gas_out = d_params = nullptr;
    g_sort_temp_bytes = g_gas_temp_bytes = g_gas_out_bytes = 0;
    g_alloc_bytes = 0;
}

void gpuInit(int n, int n_fluid, const vec3* position_array, const vec3* velocity_array, const GpuParams* params){
    if(n != n_fluid){
        fprintf(stderr, "rtxPhysics: n_fluid must equal n\n");
        exit(1);
    }
    gpuFree();
    createPipelineOnce();
    g_n = n;
    g_n_fluid = n_fluid;
    gpuSetParams(params);

    int max_dim = std::max(params->grid_w, std::max(params->grid_h, params->grid_d));
    int bits = 1;
    while((1 << bits) < max_dim) bits++;
    g_sort_bits = std::min(30, 3*std::min(bits, 10));

    ALLOC(d_pos, n*sizeof(rvec3));
    ALLOC(d_vel, n*sizeof(rvec3));
    ALLOC(d_pos2, n*sizeof(rvec3));
    ALLOC(d_vel2, n*sizeof(rvec3));
    ALLOC(d_acc, n*sizeof(rvec3));
    ALLOC(d_density, n*sizeof(real));
    ALLOC(d_pressure, n*sizeof(real));
    ALLOC(d_id, n*sizeof(int));
    ALLOC(d_id2, n*sizeof(int));
    ALLOC(d_iota, n*sizeof(int));
    ALLOC(d_perm, n*sizeof(int));
    ALLOC(d_keys, n*sizeof(unsigned int));
    ALLOC(d_keys_sorted, n*sizeof(unsigned int));
    ALLOC(d_counts, n*sizeof(int));
    ALLOC(d_aabb, n*sizeof(OptixAabb));
    ALLOC(d_stat, 2*sizeof(unsigned long long));
    ALLOC(d_params, sizeof(RtxLaunchParams));
    ALLOC(d_nbr, (size_t)NBR_CAP*(size_t)n*sizeof(int));

    std::vector<rvec3> pos_r = toReal(position_array, n);
    std::vector<rvec3> vel_r = toReal(velocity_array, n);
    CK(cudaMemcpy(d_pos, pos_r.data(), n*sizeof(rvec3), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_vel, vel_r.data(), n*sizeof(rvec3), cudaMemcpyHostToDevice));
    CK(cudaMemset(d_acc, 0, n*sizeof(rvec3)));
    CK(cudaMemset(d_density, 0, n*sizeof(real)));
    CK(cudaMemset(d_pressure, 0, n*sizeof(real)));
    CK(cudaMemset(d_counts, 0, n*sizeof(int)));

    iotaKernel<<<blocksFor(n), BLOCK>>>(d_iota, n);
    CK(cudaGetLastError());
    CK(cudaMemcpy(d_id, d_iota, n*sizeof(int), cudaMemcpyDeviceToDevice));

    CK(cub::DeviceRadixSort::SortPairs(nullptr, g_sort_temp_bytes, d_keys, d_keys_sorted, d_iota, d_perm, n, 0, g_sort_bits));
    ALLOC(d_sort_temp, g_sort_temp_bytes);

    g_aabb_ptr = (CUdeviceptr)d_aabb;
    g_gas_input = {};
    g_gas_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    g_gas_input.customPrimitiveArray.aabbBuffers = &g_aabb_ptr;
    g_gas_input.customPrimitiveArray.numPrimitives = (unsigned int)n;
    g_gas_input.customPrimitiveArray.strideInBytes = sizeof(OptixAabb);
    g_gas_input.customPrimitiveArray.flags = g_gas_flags;
    g_gas_input.customPrimitiveArray.numSbtRecords = 1;

    OptixAccelBuildOptions ao = gasOptions();
    OptixAccelBufferSizes sizes;
    OK(optixAccelComputeMemoryUsage(g_ctx, &ao, &g_gas_input, 1, &sizes));
    g_gas_temp_bytes = sizes.tempSizeInBytes;
    g_gas_out_bytes = sizes.outputSizeInBytes;
    ALLOC(d_gas_temp, g_gas_temp_bytes);
    ALLOC(d_gas_out, g_gas_out_bytes);

    CK(cudaDeviceSynchronize());
}

void gpuDrift(double dt){
    driftKernel<<<blocksFor(g_n_fluid), BLOCK>>>(d_pos, d_vel, g_n_fluid, R(dt));
    CK(cudaGetLastError());
}

void gpuEnforceBoundary(){
    boundaryKernel<<<blocksFor(g_n_fluid), BLOCK>>>(g_kp, d_pos, d_vel, g_n_fluid);
    CK(cudaGetLastError());
}

void gpuBuildGrid(){
    stamp(0);
    mortonKernel<<<blocksFor(g_n), BLOCK>>>(g_kp, d_pos, d_keys, g_n);
    CK(cudaGetLastError());
    CK(cub::DeviceRadixSort::SortPairs(d_sort_temp, g_sort_temp_bytes, d_keys, d_keys_sorted, d_iota, d_perm, g_n, 0, g_sort_bits));
    gatherKernel<<<blocksFor(g_n), BLOCK>>>(d_perm, d_pos, d_vel, d_id, d_pos2, d_vel2, d_id2, g_n);
    CK(cudaGetLastError());
    std::swap(d_pos, d_pos2);
    std::swap(d_vel, d_vel2);
    std::swap(d_id, d_id2);
    stamp(1);

    aabbKernel<<<blocksFor(g_n), BLOCK>>>(d_pos, d_aabb, g_kp.H*R(2.0005), g_n);
    CK(cudaGetLastError());
    OptixAccelBuildOptions ao = gasOptions();
    OK(optixAccelBuild(g_ctx, 0, &ao, &g_gas_input, 1, (CUdeviceptr)d_gas_temp, g_gas_temp_bytes, (CUdeviceptr)d_gas_out, g_gas_out_bytes, &g_gas_handle, nullptr, 0));
    stamp(2);

    RtxLaunchParams lp;
    lp.handle = g_gas_handle;
    lp.pos = (const float3*)d_pos;
    lp.nbr = d_nbr;
    lp.counts = d_counts;
    lp.n = g_n;
    lp.cap = NBR_CAP;
    lp.cutoff2 = g_kp.cutoff2;
    CK(cudaMemcpyAsync(d_params, &lp, sizeof(lp), cudaMemcpyHostToDevice, 0));
    OK(optixLaunch(g_pipeline, 0, (CUdeviceptr)d_params, sizeof(RtxLaunchParams), &g_sbt, (unsigned int)g_n, 1, 1));
    stamp(3);
}

void gpuDensity(){
    densityListKernel<<<blocksFor(g_n), BLOCK>>>(g_kp, d_pos, d_nbr, d_counts, d_density, g_n, NBR_CAP);
    CK(cudaGetLastError());
}

void gpuPressure(){
    pressureKernel<<<blocksFor(g_n), BLOCK>>>(g_kp, d_density, d_pressure, g_n);
    CK(cudaGetLastError());
}

void gpuForces(){
    forceListKernel<<<blocksFor(g_n_fluid), BLOCK>>>(g_kp, d_pos, d_vel, d_density, d_pressure, d_nbr, d_counts, d_acc, g_n_fluid, NBR_CAP);
    CK(cudaGetLastError());
}

void gpuKick(double dt){
    kickKernel<<<blocksFor(g_n_fluid), BLOCK>>>(d_vel, d_acc, g_n_fluid, R(dt));
    CK(cudaGetLastError());
}

void gpuDensitySearchOnly(){
}

void gpuForcesSearchOnly(){
}

size_t gpuAllocatedBytes(){
    return g_alloc_bytes;
}

void gpuSync(){
    CK(cudaDeviceSynchronize());
}

void gpuDownloadPositions(vec3* position_array){
    fromRealById(d_pos, position_array, g_n);
}

void gpuDownloadVelocities(vec3* velocity_array){
    fromRealById(d_vel, velocity_array, g_n);
}

void gpuDownloadDensity(double* density_array){
    std::vector<real> tmp(g_n);
    std::vector<int> ids(g_n);
    CK(cudaMemcpy(tmp.data(), d_density, g_n*sizeof(real), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ids.data(), d_id, g_n*sizeof(int), cudaMemcpyDeviceToHost));
    for(int s = 0; s < g_n; s++) density_array[ids[s]] = (double)tmp[s];
}

int gpuNeighbourCapacity(){
    return NBR_CAP;
}

static void readStats(unsigned long long* out2){
    CK(cudaMemset(d_stat, 0, 2*sizeof(unsigned long long)));
    statKernel<<<blocksFor(g_n), BLOCK>>>(d_counts, g_n, NBR_CAP, d_stat);
    CK(cudaGetLastError());
    CK(cudaMemcpy(out2, d_stat, 2*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
}

long long gpuNeighbourOverflow(){
    unsigned long long s[2];
    readStats(s);
    return (long long)s[1];
}

long long gpuNeighbourTotal(){
    unsigned long long s[2];
    readStats(s);
    return (long long)s[0];
}

void gpuDownloadNeighbourCounts(int* counts_in_original_order){
    std::vector<int> tmp(g_n), ids(g_n);
    CK(cudaMemcpy(tmp.data(), d_counts, g_n*sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ids.data(), d_id, g_n*sizeof(int), cudaMemcpyDeviceToHost));
    for(int s = 0; s < g_n; s++) counts_in_original_order[ids[s]] = tmp[s];
}

void gpuEnableSearchTiming(int on){
    g_timing = on != 0;
    if(g_timing && !g_ev_ready){
        for(int k = 0; k < 4; k++) CK(cudaEventCreate(&g_ev[k]));
        g_ev_ready = true;
    }
}

void gpuSearchTimes(double* ms_out4){
    float a = 0, b = 0, c = 0;
    CK(cudaEventSynchronize(g_ev[3]));
    CK(cudaEventElapsedTime(&a, g_ev[0], g_ev[1]));
    CK(cudaEventElapsedTime(&b, g_ev[1], g_ev[2]));
    CK(cudaEventElapsedTime(&c, g_ev[2], g_ev[3]));
    ms_out4[0] = a;
    ms_out4[1] = b;
    ms_out4[2] = c;
    ms_out4[3] = (double)a + b + c;
}
