#define NOMINMAX
#include "frnnShared.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <functional>

#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "frnn_programs.h"

#define CK(call) do{ cudaError_t e_ = (call); if(e_ != cudaSuccess){ fprintf(stderr, "CUDA error: %s (%s:%d): %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } }while(0)

static char g_log[8192];
static size_t g_log_size = 0;
#define OK(call) do{ g_log_size = sizeof(g_log); OptixResult r_ = (call); if(r_ != OPTIX_SUCCESS){ fprintf(stderr, "OptiX error: %s (%s:%d): %s\n%s\n", #call, __FILE__, __LINE__, optixGetErrorName(r_), g_log); exit(1); } }while(0)

struct SbtRecord { __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };

static const int BLOCK = 256;
static int blocksFor(long long count){ return (int)((count + BLOCK - 1) / BLOCK); }

__device__ unsigned int expandBits(unsigned int v){
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v << 8))  & 0x0300F00Fu;
    v = (v | (v << 4))  & 0x030C30C3u;
    v = (v | (v << 2))  & 0x09249249u;
    return v;
}

__global__ void iotaKernel(int* out, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) out[i] = i;
}

__global__ void mortonKernel(const float3* pos, unsigned int* keys, int n, float inv_cell){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;
    int cx = min(max((int)(pos[i].x*inv_cell), 0), 1023);
    int cy = min(max((int)(pos[i].y*inv_cell), 0), 1023);
    int cz = min(max((int)(pos[i].z*inv_cell), 0), 1023);
    keys[i] = expandBits((unsigned)cx) | (expandBits((unsigned)cy) << 1) | (expandBits((unsigned)cz) << 2);
}

__global__ void gatherKernel(const int* perm, const float3* pos_in, const float* rad_in, float3* pos_out, float* rad_out, int n){
    int s = blockIdx.x*blockDim.x + threadIdx.x;
    if(s >= n) return;
    int o = perm[s];
    pos_out[s] = pos_in[o];
    rad_out[s] = rad_in[o];
}

__global__ void aabbKernel(const float3* pos, const float* rad, OptixAabb* aabb, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;
    float h = rad[i]*1.0005f;
    OptixAabb b;
    b.minX = pos[i].x - h; b.minY = pos[i].y - h; b.minZ = pos[i].z - h;
    b.maxX = pos[i].x + h; b.maxY = pos[i].y + h; b.maxZ = pos[i].z + h;
    aabb[i] = b;
}

__global__ void cellIdKernel(const float3* pos, int* cell_id, int n, float inv_cell, int gdim){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;
    int cx = min(max((int)(pos[i].x*inv_cell), 0), gdim - 1);
    int cy = min(max((int)(pos[i].y*inv_cell), 0), gdim - 1);
    int cz = min(max((int)(pos[i].z*inv_cell), 0), gdim - 1);
    cell_id[i] = cx + cy*gdim + cz*gdim*gdim;
}

__global__ void cellStartKernel(const int* sorted_keys, int* cell_start, int n, int num_cells){
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if(c > num_cells) return;
    int lo = 0, hi = n;
    while(lo < hi){
        int mid = (lo + hi) >> 1;
        if(sorted_keys[mid] < c) lo = mid + 1; else hi = mid;
    }
    cell_start[c] = lo;
}

__global__ void gridCountKernel(const float3* pos, const float* rad, const int* cell_id, const int* cell_start, const int* cell_particles,
                                int gdim, int offset, int count, int* counts){
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if(t >= count) return;
    int i = offset + t;
    int cid = cell_id[i];
    int cx = cid % gdim, cy = (cid / gdim) % gdim, cz = cid / (gdim*gdim);
    float xi = pos[i].x, yi = pos[i].y, zi = pos[i].z;
    int k = 0;
    for(int dz = -1; dz <= 1; dz++){
        int nz = cz + dz;
        if(nz < 0 || nz >= gdim) continue;
        for(int dy = -1; dy <= 1; dy++){
            int ny = cy + dy;
            if(ny < 0 || ny >= gdim) continue;
            for(int dx = -1; dx <= 1; dx++){
                int nx = cx + dx;
                if(nx < 0 || nx >= gdim) continue;
                int c = nx + ny*gdim + nz*gdim*gdim;
                for(int m = cell_start[c]; m < cell_start[c+1]; m++){
                    int j = cell_particles[m];
                    float rx = pos[j].x - xi, ry = pos[j].y - yi, rz = pos[j].z - zi;
                    float r = rad[j];
                    if(rx*rx + ry*ry + rz*rz < r*r) k++;
                }
            }
        }
    }
    counts[i] = k;
}

__global__ void statsKernel(const int* counts, int n, unsigned long long* sum, int* mx){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;
    atomicAdd(sum, (unsigned long long)counts[i]);
    atomicMax(mx, counts[i]);
}

struct Args {
    std::string dist = "disordered", radius = "const:1";
    int n = 50000, cap = 128, seed = 1, clusters = 4;
    double box = 103.0, budget_s = 15.0, sigma = 25.0;
};

static void scene(const Args& a, std::vector<float3>& pos, std::vector<float>& rad, double& rmin, double& rmax){
    std::mt19937_64 rng((unsigned long long)a.seed*7919ull + (unsigned long long)a.n);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    double L = a.box;
    pos.resize(a.n);
    rad.resize(a.n);
    if(a.dist == "lattice"){
        int side = (int)ceil(cbrt((double)a.n));
        double sp = L/side;
        for(int idx = 0; idx < a.n; idx++){
            int i = idx % side, j = (idx / side) % side, k = idx / (side*side);
            pos[idx] = make_float3((float)((i + 0.5)*sp), (float)((j + 0.5)*sp), (float)((k + 0.5)*sp));
        }
    }
    else if(a.dist == "disordered"){
        for(auto& p : pos) p = make_float3((float)(u01(rng)*L), (float)(u01(rng)*L), (float)(u01(rng)*L));
    }
    else if(a.dist == "cluster"){
        std::vector<double> cx(a.clusters), cy(a.clusters), cz(a.clusters);
        for(int c = 0; c < a.clusters; c++){ cx[c] = u01(rng)*L; cy[c] = u01(rng)*L; cz[c] = u01(rng)*L; }
        std::normal_distribution<double> g(0.0, a.sigma);
        std::uniform_int_distribution<int> pick(0, a.clusters - 1);
        for(auto& p : pos){
            int c = pick(rng);
            double x, y, z;
            do { x = cx[c] + g(rng); } while(x < 0 || x >= L);
            do { y = cy[c] + g(rng); } while(y < 0 || y >= L);
            do { z = cz[c] + g(rng); } while(z < 0 || z >= L);
            p = make_float3((float)x, (float)y, (float)z);
        }
    }
    else { fprintf(stderr, "unknown --dist %s\n", a.dist.c_str()); exit(1); }

    std::vector<std::string> f;
    size_t st = 0, e;
    while((e = a.radius.find(':', st)) != std::string::npos){ f.push_back(a.radius.substr(st, e - st)); st = e + 1; }
    f.push_back(a.radius.substr(st));
    if(f[0] == "const"){
        double r = atof(f[1].c_str());
        for(auto& x : rad) x = (float)r;
    }
    else if(f[0] == "uniform"){
        std::uniform_real_distribution<double> ud(atof(f[1].c_str()), atof(f[2].c_str()));
        for(auto& x : rad) x = (float)ud(rng);
    }
    else if(f[0] == "lognormal"){
        std::lognormal_distribution<double> ln(atof(f[1].c_str()), atof(f[2].c_str()));
        double lo = atof(f[3].c_str()), hi = atof(f[4].c_str());
        for(auto& x : rad) x = (float)std::min(hi, std::max(lo, ln(rng)));
    }
    else { fprintf(stderr, "unknown --radius %s\n", a.radius.c_str()); exit(1); }
    rmin = 1e30; rmax = 0;
    for(float x : rad){ rmin = std::min(rmin, (double)x); rmax = std::max(rmax, (double)x); }
}

struct Timer {
    cudaEvent_t a, b;
    Timer(){ CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); }
    void start(){ CK(cudaEventRecord(a, 0)); }
    double stop(){ CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b)); float ms = 0; CK(cudaEventElapsedTime(&ms, a, b)); return ms; }
};

static double chunked(int n, long long& chunk, const std::function<void(int, int)>& launch){
    Timer t;
    double total = 0;
    int off = 0;
    while(off < n){
        int cnt = (int)std::min<long long>(chunk, n - off);
        t.start();
        launch(off, cnt);
        double ms = t.stop();
        total += ms;
        off += cnt;
        if(ms > 0.05) chunk = std::max<long long>(1024, std::min<long long>(n, (long long)(cnt*150.0/ms)));
        else chunk = n;
    }
    return total;
}

int main(int argc, char** argv){
    Args a;
    for(int i = 1; i < argc; i++){
        std::string k = argv[i];
        auto val = [&](){ if(i + 1 >= argc){ fprintf(stderr, "missing value for %s\n", k.c_str()); exit(1); } return std::string(argv[++i]); };
        if(k == "--dist") a.dist = val();
        else if(k == "--radius") a.radius = val();
        else if(k == "--n") a.n = atoi(val().c_str());
        else if(k == "--cap") a.cap = atoi(val().c_str());
        else if(k == "--seed") a.seed = atoi(val().c_str());
        else if(k == "--box") a.box = atof(val().c_str());
        else if(k == "--budget-s") a.budget_s = atof(val().c_str());
        else if(k == "--clusters") a.clusters = atoi(val().c_str());
        else { fprintf(stderr, "unknown option %s\n", k.c_str()); return 1; }
    }
    const int n = a.n;

    std::vector<float3> h_pos;
    std::vector<float> h_rad;
    double rmin, rmax;
    scene(a, h_pos, h_rad, rmin, rmax);

    float3 *d_pos0, *d_pos, *d_pos_rt, *d_pos_rt2;
    float *d_rad0, *d_rad, *d_rad_rt, *d_rad_rt2;
    unsigned int *d_mk, *d_mk_sorted;
    int *d_iota, *d_perm, *d_counts_grid, *d_counts_rt, *d_mx;
    unsigned long long* d_sum;
    CK(cudaMalloc(&d_pos0, n*sizeof(float3))); CK(cudaMalloc(&d_pos, n*sizeof(float3)));
    CK(cudaMalloc(&d_pos_rt, n*sizeof(float3))); CK(cudaMalloc(&d_pos_rt2, n*sizeof(float3)));
    CK(cudaMalloc(&d_rad0, n*sizeof(float))); CK(cudaMalloc(&d_rad, n*sizeof(float)));
    CK(cudaMalloc(&d_rad_rt, n*sizeof(float))); CK(cudaMalloc(&d_rad_rt2, n*sizeof(float)));
    CK(cudaMalloc(&d_mk, n*sizeof(unsigned))); CK(cudaMalloc(&d_mk_sorted, n*sizeof(unsigned)));
    CK(cudaMalloc(&d_iota, n*sizeof(int))); CK(cudaMalloc(&d_perm, n*sizeof(int)));
    CK(cudaMalloc(&d_counts_grid, n*sizeof(int))); CK(cudaMalloc(&d_counts_rt, n*sizeof(int)));
    CK(cudaMalloc(&d_mx, sizeof(int))); CK(cudaMalloc(&d_sum, sizeof(unsigned long long)));
    CK(cudaMemcpy(d_pos0, h_pos.data(), n*sizeof(float3), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_rad0, h_rad.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    iotaKernel<<<blocksFor(n), BLOCK>>>(d_iota, n);

    float inv_mcell = (float)(1024.0/a.box);
    size_t msort_bytes = 0;
    void* d_msort = nullptr;
    CK(cub::DeviceRadixSort::SortPairs(nullptr, msort_bytes, d_mk, d_mk_sorted, d_iota, d_perm, n, 0, 30));
    CK(cudaMalloc(&d_msort, msort_bytes));
    mortonKernel<<<blocksFor(n), BLOCK>>>(d_pos0, d_mk, n, inv_mcell);
    CK(cub::DeviceRadixSort::SortPairs(d_msort, msort_bytes, d_mk, d_mk_sorted, d_iota, d_perm, n, 0, 30));
    gatherKernel<<<blocksFor(n), BLOCK>>>(d_perm, d_pos0, d_rad0, d_pos, d_rad, n);
    CK(cudaMemcpy(d_pos_rt, d_pos, n*sizeof(float3), cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(d_rad_rt, d_rad, n*sizeof(float), cudaMemcpyDeviceToDevice));
    CK(cudaDeviceSynchronize());

    double cell = rmax;
    int gdim = std::max(1, (int)floor(a.box/cell));
    if(gdim > 1024) gdim = 1024;
    float inv_cell = (float)(gdim/a.box);
    int num_cells = gdim*gdim*gdim;
    int cbits = 1;
    while((1 << cbits) < num_cells) cbits++;
    int *d_cell_id, *d_sorted_keys, *d_cell_particles, *d_cell_start;
    CK(cudaMalloc(&d_cell_id, n*sizeof(int))); CK(cudaMalloc(&d_sorted_keys, n*sizeof(int)));
    CK(cudaMalloc(&d_cell_particles, n*sizeof(int))); CK(cudaMalloc(&d_cell_start, (num_cells + 1)*sizeof(int)));
    size_t gsort_bytes = 0;
    void* d_gsort = nullptr;
    CK(cub::DeviceRadixSort::SortPairs(nullptr, gsort_bytes, d_cell_id, d_sorted_keys, d_iota, d_cell_particles, n, 0, cbits));
    CK(cudaMalloc(&d_gsort, gsort_bytes));

    CK(cudaFree(0));
    OK(optixInit());
    OptixDeviceContext ctx;
    OptixDeviceContextOptions co = {};
    OK(optixDeviceContextCreate(0, &co, &ctx));
    OptixModuleCompileOptions mco = {};
    mco.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    OptixPipelineCompileOptions pco = {};
    pco.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pco.numPayloadValues = 1;
    pco.numAttributeValues = 2;
    pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pco.pipelineLaunchParamsVariableName = "params";
    pco.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
    OptixModule module;
    OK(optixModuleCreate(ctx, &mco, &pco, (const char*)frnn_programs, sizeof(frnn_programs), g_log, &g_log_size, &module));
    OptixProgramGroupOptions pgo = {};
    OptixProgramGroup pg[3];
    OptixProgramGroupDesc d = {};
    d.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d.raygen.module = module; d.raygen.entryFunctionName = "__raygen__frnn";
    OK(optixProgramGroupCreate(ctx, &d, 1, &pgo, g_log, &g_log_size, &pg[0]));
    d = {}; d.kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d.miss.module = module; d.miss.entryFunctionName = "__miss__frnn";
    OK(optixProgramGroupCreate(ctx, &d, 1, &pgo, g_log, &g_log_size, &pg[1]));
    d = {}; d.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    d.hitgroup.moduleIS = module; d.hitgroup.entryFunctionNameIS = "__intersection__frnn";
    d.hitgroup.moduleAH = module; d.hitgroup.entryFunctionNameAH = "__anyhit__frnn";
    OK(optixProgramGroupCreate(ctx, &d, 1, &pgo, g_log, &g_log_size, &pg[2]));
    OptixPipelineLinkOptions plo = {};
    plo.maxTraceDepth = 1;
    OptixPipeline pipeline;
    OK(optixPipelineCreate(ctx, &pco, &plo, pg, 3, g_log, &g_log_size, &pipeline));
    SbtRecord recs[3];
    for(int k = 0; k < 3; k++) OK(optixSbtRecordPackHeader(pg[k], &recs[k]));
    void* d_recs;
    CK(cudaMalloc(&d_recs, sizeof(recs)));
    CK(cudaMemcpy(d_recs, recs, sizeof(recs), cudaMemcpyHostToDevice));
    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord = (CUdeviceptr)d_recs;
    sbt.missRecordBase = (CUdeviceptr)d_recs + sizeof(SbtRecord);
    sbt.missRecordStrideInBytes = sizeof(SbtRecord);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = (CUdeviceptr)d_recs + 2*sizeof(SbtRecord);
    sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord);
    sbt.hitgroupRecordCount = 1;

    OptixAabb* d_aabb;
    CK(cudaMalloc(&d_aabb, n*sizeof(OptixAabb)));
    CUdeviceptr aabb_ptr = (CUdeviceptr)d_aabb;
    unsigned int gflags[1] = { OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL };
    OptixBuildInput bi = {};
    bi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    bi.customPrimitiveArray.aabbBuffers = &aabb_ptr;
    bi.customPrimitiveArray.numPrimitives = (unsigned)n;
    bi.customPrimitiveArray.strideInBytes = sizeof(OptixAabb);
    bi.customPrimitiveArray.flags = gflags;
    bi.customPrimitiveArray.numSbtRecords = 1;
    OptixAccelBuildOptions ao = {};
    ao.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
    ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes bs;
    OK(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs));
    void *d_gtemp, *d_gout, *d_params;
    CK(cudaMalloc(&d_gtemp, bs.tempSizeInBytes));
    CK(cudaMalloc(&d_gout, bs.outputSizeInBytes));
    CK(cudaMalloc(&d_params, sizeof(FrnnParams)));
    int* d_nbr = nullptr;

    long long chunk_grid = std::max(1024, n/1024), chunk_rt = std::max(1024, n/1024), chunk_list = std::max(1024, n/1024);

    auto grid_call = [&](double* t){
        Timer tm;
        tm.start();
        cellIdKernel<<<blocksFor(n), BLOCK>>>(d_pos, d_cell_id, n, inv_cell, gdim);
        CK(cub::DeviceRadixSort::SortPairs(d_gsort, gsort_bytes, d_cell_id, d_sorted_keys, d_iota, d_cell_particles, n, 0, cbits));
        cellStartKernel<<<blocksFor(num_cells + 1), BLOCK>>>(d_sorted_keys, d_cell_start, n, num_cells);
        t[0] = tm.stop();
        t[1] = chunked(n, chunk_grid, [&](int off, int cnt){
            gridCountKernel<<<blocksFor(cnt), BLOCK>>>(d_pos, d_rad, d_cell_id, d_cell_start, d_cell_particles, gdim, off, cnt, d_counts_grid);
        });
        CK(cudaGetLastError());
    };

    OptixTraversableHandle handle = 0;
    auto rt_call = [&](int mode, double* t){
        Timer tm;
        tm.start();
        mortonKernel<<<blocksFor(n), BLOCK>>>(d_pos_rt, d_mk, n, inv_mcell);
        CK(cub::DeviceRadixSort::SortPairs(d_msort, msort_bytes, d_mk, d_mk_sorted, d_iota, d_perm, n, 0, 30));
        gatherKernel<<<blocksFor(n), BLOCK>>>(d_perm, d_pos_rt, d_rad_rt, d_pos_rt2, d_rad_rt2, n);
        std::swap(d_pos_rt, d_pos_rt2);
        std::swap(d_rad_rt, d_rad_rt2);
        t[0] = tm.stop();
        tm.start();
        aabbKernel<<<blocksFor(n), BLOCK>>>(d_pos_rt, d_rad_rt, d_aabb, n);
        OK(optixAccelBuild(ctx, 0, &ao, &bi, 1, (CUdeviceptr)d_gtemp, bs.tempSizeInBytes, (CUdeviceptr)d_gout, bs.outputSizeInBytes, &handle, nullptr, 0));
        t[1] = tm.stop();
        long long& ch = mode == 0 ? chunk_rt : chunk_list;
        t[2] = chunked(n, ch, [&](int off, int cnt){
            FrnnParams p;
            p.handle = handle; p.pos = d_pos_rt; p.radius = d_rad_rt; p.counts = d_counts_rt; p.nbr = d_nbr;
            p.n = n; p.cap = a.cap; p.offset = off; p.mode = mode;
            CK(cudaMemcpyAsync(d_params, &p, sizeof(p), cudaMemcpyHostToDevice, 0));
            OK(optixLaunch(pipeline, 0, (CUdeviceptr)d_params, sizeof(FrnnParams), &sbt, (unsigned)cnt, 1, 1));
        });
    };

    auto stats = [&](const int* counts, unsigned long long& sum, int& mx){
        CK(cudaMemset(d_sum, 0, sizeof(unsigned long long)));
        CK(cudaMemset(d_mx, 0, sizeof(int)));
        statsKernel<<<blocksFor(n), BLOCK>>>(counts, n, d_sum, d_mx);
        CK(cudaMemcpy(&sum, d_sum, sizeof(sum), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&mx, d_mx, sizeof(mx), cudaMemcpyDeviceToHost));
    };

    double tg[2], tr[3], tl[3];
    grid_call(tg);
    rt_call(0, tr);
    unsigned long long sum_g, sum_r;
    int max_g, max_r;
    stats(d_counts_grid, sum_g, max_g);
    stats(d_counts_rt, sum_r, max_r);
    bool do_list = max_r <= a.cap;
    unsigned long long sum_l = 0;
    std::vector<int> cg(n), cr(n), cl;
    CK(cudaMemcpy(cg.data(), d_counts_grid, n*sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(cr.data(), d_counts_rt, n*sizeof(int), cudaMemcpyDeviceToHost));
    if(do_list){
        CK(cudaMalloc(&d_nbr, (size_t)a.cap*(size_t)n*sizeof(int)));
        rt_call(1, tl);
        int max_l;
        stats(d_counts_rt, sum_l, max_l);
        cl.resize(n);
        CK(cudaMemcpy(cl.data(), d_counts_rt, n*sizeof(int), cudaMemcpyDeviceToHost));
    }
    int mismatched = 0, worst = 0, list_mismatched = 0;
    for(int i = 0; i < n; i++){
        if(cr[i] != cg[i]) mismatched++;
        worst = std::max(worst, abs(cr[i] - cg[i]));
        if(do_list && cl[i] != cr[i]) list_mismatched++;
    }
    bool valid = worst <= 1 && mismatched <= std::max(1, n/1000) && list_mismatched == 0;

    double first = tg[0] + tg[1] + tr[0] + tr[1] + tr[2] + (do_list ? tl[0] + tl[1] + tl[2] : 0);
    std::vector<double> G0, G1, R0, R1, R2, L2;
    bool warm_counts = first > 10000.0;
    if(warm_counts){
        G0.push_back(tg[0]); G1.push_back(tg[1]);
        R0.push_back(tr[0]); R1.push_back(tr[1]); R2.push_back(tr[2]);
        if(do_list) L2.push_back(tl[2]);
    }
    double spent = first;
    while(!warm_counts && (G1.size() < 3 || (spent < a.budget_s*1000.0 && G1.size() < 10))){
        grid_call(tg);
        rt_call(0, tr);
        G0.push_back(tg[0]); G1.push_back(tg[1]);
        R0.push_back(tr[0]); R1.push_back(tr[1]); R2.push_back(tr[2]);
        spent += tg[0] + tg[1] + tr[0] + tr[1] + tr[2];
        if(do_list){ rt_call(1, tl); L2.push_back(tl[2]); spent += tl[0] + tl[1] + tl[2]; }
    }

    auto mean = [](const std::vector<double>& v){ double s = 0; for(double x : v) s += x; return v.empty() ? -1.0 : s/v.size(); };
    auto sdev = [&](const std::vector<double>& v){ double m = mean(v), s = 0; for(double x : v) s += (x - m)*(x - m); return v.size() > 1 ? sqrt(s/(v.size() - 1)) : 0.0; };
    std::vector<double> grid_tot, rt_tot;
    for(size_t k = 0; k < G1.size(); k++){ grid_tot.push_back(G0[k] + G1[k]); rt_tot.push_back(R0[k] + R1[k] + R2[k]); }

    printf("FRNN mismatched_queries=%d worst_count_diff=%d list_mismatched=%d sum_grid=%llu sum_rt=%llu ", mismatched, worst, list_mismatched, sum_g, sum_r);
    printf("dist=%s radius=%s n=%d box=%.1f rmin=%.4f rmax=%.4f gdim=%d cap=%d reps=%zu valid=%d "
           "mean_nbrs=%.3f max_nbrs=%d grid_build_ms=%.4f grid_pass_ms=%.4f rt_sched_ms=%.4f rt_bvh_ms=%.4f rt_trace_count_ms=%.4f rt_trace_list_ms=%.4f "
           "grid_total_ms=%.4f grid_total_sd=%.4f rt_total_ms=%.4f rt_total_sd=%.4f\n",
           a.dist.c_str(), a.radius.c_str(), n, a.box, rmin, rmax, gdim, a.cap, G1.size(), valid ? 1 : 0,
           (double)sum_r/n, max_r, mean(G0), mean(G1), mean(R0), mean(R1), mean(R2), do_list ? mean(L2) : -1.0,
           mean(grid_tot), sdev(grid_tot), mean(rt_tot), sdev(rt_tot));
    return valid ? 0 : 2;
}
