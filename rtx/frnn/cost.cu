#define NOMINMAX
#include "costShared.h"

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

#ifdef COST_INSTR
#include "cost_programs_instr.h"
#define PROGRAMS cost_programs_instr
#define NUM_PAYLOAD 2
#else
#include "cost_programs.h"
#define PROGRAMS cost_programs
#define NUM_PAYLOAD 1
#endif

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

__global__ void cellStartKernel(const int* sorted_keys, int* cell_start, int n, long long num_cells){
    long long c = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if(c > num_cells) return;
    int lo = 0, hi = n;
    while(lo < hi){
        int mid = (lo + hi) >> 1;
        if(sorted_keys[mid] < c) lo = mid + 1; else hi = mid;
    }
    cell_start[c] = lo;
}

__global__ void gridCountKernel(const float3* pos, const float* rad, const int* cell_id, const int* cell_start, const int* cell_particles,
                                int gdim, int offset, int count, int* counts, int* cand){
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if(t >= count) return;
    int i = offset + t;
    int cid = cell_id[i];
    int cx = cid % gdim, cy = (cid / gdim) % gdim, cz = cid / (gdim*gdim);
    float xi = pos[i].x, yi = pos[i].y, zi = pos[i].z;
    int k = 0;
#ifdef COST_INSTR
    int tests = 0;
#endif
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
                int e = cell_start[c+1];
#ifdef COST_INSTR
                tests += e - cell_start[c];
#endif
                for(int m = cell_start[c]; m < e; m++){
                    int j = cell_particles[m];
                    float rx = pos[j].x - xi, ry = pos[j].y - yi, rz = pos[j].z - zi;
                    float r = rad[j];
                    if(rx*rx + ry*ry + rz*rz < r*r) k++;
                }
            }
        }
    }
    counts[i] = k;
#ifdef COST_INSTR
    cand[i] = tests;
#endif
}

__global__ void sumKernel(const int* v, int n, unsigned long long* sum, int* mx){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;
    atomicAdd(sum, (unsigned long long)v[i]);
    atomicMax(mx, v[i]);
}

struct Args {
    std::string dist = "disordered", radius = "";
    int n = 1000000, seed = 1, clusters = 4, reps = 7, loop = 10;
    double box = 103.0, sigma = 25.0, k = -1.0, warm_s = 1.0;
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

    std::string spec = a.radius;
    if(a.k > 0){
        double rho = a.n/(L*L*L);
        char buf[64];
        snprintf(buf, sizeof(buf), "const:%.6f", cbrt(3.0*a.k/(4.0*3.14159265358979*rho)));
        spec = buf;
    }
    if(spec.empty()){ fprintf(stderr, "need --radius or --k\n"); exit(1); }
    std::vector<std::string> f;
    size_t st = 0, e;
    while((e = spec.find(':', st)) != std::string::npos){ f.push_back(spec.substr(st, e - st)); st = e + 1; }
    f.push_back(spec.substr(st));
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
    else { fprintf(stderr, "unknown --radius %s\n", spec.c_str()); exit(1); }
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

static double median(std::vector<double> v){
    if(v.empty()) return -1.0;
    std::sort(v.begin(), v.end());
    size_t m = v.size()/2;
    return v.size() % 2 ? v[m] : 0.5*(v[m-1] + v[m]);
}

static double vmin(const std::vector<double>& v){
    double m = 1e300;
    for(double x : v) m = std::min(m, x);
    return v.empty() ? -1.0 : m;
}

int main(int argc, char** argv){
    Args a;
    for(int i = 1; i < argc; i++){
        std::string k = argv[i];
        auto val = [&](){ if(i + 1 >= argc){ fprintf(stderr, "missing value for %s\n", k.c_str()); exit(1); } return std::string(argv[++i]); };
        if(k == "--dist") a.dist = val();
        else if(k == "--radius") a.radius = val();
        else if(k == "--k") a.k = atof(val().c_str());
        else if(k == "--n") a.n = atoi(val().c_str());
        else if(k == "--seed") a.seed = atoi(val().c_str());
        else if(k == "--box") a.box = atof(val().c_str());
        else if(k == "--clusters") a.clusters = atoi(val().c_str());
        else if(k == "--sigma") a.sigma = atof(val().c_str());
        else if(k == "--reps") a.reps = atoi(val().c_str());
        else if(k == "--loop") a.loop = atoi(val().c_str());
        else if(k == "--warm-s") a.warm_s = atof(val().c_str());
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
    int *d_iota, *d_perm, *d_counts_grid, *d_counts_rt, *d_cand, *d_iscnt, *d_mx;
    unsigned long long* d_sum;
    CK(cudaMalloc(&d_pos0, n*sizeof(float3))); CK(cudaMalloc(&d_pos, n*sizeof(float3)));
    CK(cudaMalloc(&d_pos_rt, n*sizeof(float3))); CK(cudaMalloc(&d_pos_rt2, n*sizeof(float3)));
    CK(cudaMalloc(&d_rad0, n*sizeof(float))); CK(cudaMalloc(&d_rad, n*sizeof(float)));
    CK(cudaMalloc(&d_rad_rt, n*sizeof(float))); CK(cudaMalloc(&d_rad_rt2, n*sizeof(float)));
    CK(cudaMalloc(&d_mk, n*sizeof(unsigned))); CK(cudaMalloc(&d_mk_sorted, n*sizeof(unsigned)));
    CK(cudaMalloc(&d_iota, n*sizeof(int))); CK(cudaMalloc(&d_perm, n*sizeof(int)));
    CK(cudaMalloc(&d_counts_grid, n*sizeof(int))); CK(cudaMalloc(&d_counts_rt, n*sizeof(int)));
    CK(cudaMalloc(&d_cand, n*sizeof(int))); CK(cudaMalloc(&d_iscnt, n*sizeof(int)));
    CK(cudaMalloc(&d_mx, sizeof(int))); CK(cudaMalloc(&d_sum, sizeof(unsigned long long)));
    CK(cudaMemcpy(d_pos0, h_pos.data(), n*sizeof(float3), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_rad0, h_rad.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemset(d_cand, 0, n*sizeof(int)));
    CK(cudaMemset(d_iscnt, 0, n*sizeof(int)));
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
    long long num_cells = (long long)gdim*gdim*gdim;
    int cbits = 1;
    while((1ll << cbits) < num_cells) cbits++;
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
    pco.numPayloadValues = NUM_PAYLOAD;
    pco.numAttributeValues = 2;
    pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pco.pipelineLaunchParamsVariableName = "params";
    pco.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
    OptixModule module;
    OK(optixModuleCreate(ctx, &mco, &pco, (const char*)PROGRAMS, sizeof(PROGRAMS), g_log, &g_log_size, &module));
    OptixProgramGroupOptions pgo = {};
    OptixProgramGroup pg[3];
    OptixProgramGroupDesc d = {};
    d.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d.raygen.module = module; d.raygen.entryFunctionName = "__raygen__cost";
    OK(optixProgramGroupCreate(ctx, &d, 1, &pgo, g_log, &g_log_size, &pg[0]));
    d = {}; d.kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d.miss.module = module; d.miss.entryFunctionName = "__miss__cost";
    OK(optixProgramGroupCreate(ctx, &d, 1, &pgo, g_log, &g_log_size, &pg[1]));
    d = {}; d.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    d.hitgroup.moduleIS = module; d.hitgroup.entryFunctionNameIS = "__intersection__cost";
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
    unsigned int gflags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
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
    CK(cudaMalloc(&d_params, sizeof(CostParams)));

    long long chunk_grid = std::max(1024, n/1024), chunk_rt = std::max(1024, n/1024);

    auto grid_call = [&](double* t){
        Timer tm;
        tm.start();
        cellIdKernel<<<blocksFor(n), BLOCK>>>(d_pos, d_cell_id, n, inv_cell, gdim);
        CK(cub::DeviceRadixSort::SortPairs(d_gsort, gsort_bytes, d_cell_id, d_sorted_keys, d_iota, d_cell_particles, n, 0, cbits));
        t[0] = tm.stop();
        tm.start();
        cellStartKernel<<<blocksFor(num_cells + 1), BLOCK>>>(d_sorted_keys, d_cell_start, n, num_cells);
        t[1] = tm.stop();
        CK(cudaGetLastError());
        t[2] = chunked(n, chunk_grid, [&](int off, int cnt){
            gridCountKernel<<<blocksFor(cnt), BLOCK>>>(d_pos, d_rad, d_cell_id, d_cell_start, d_cell_particles, gdim, off, cnt, d_counts_grid, d_cand);
        });
        CK(cudaGetLastError());
    };

    OptixTraversableHandle handle = 0;
    auto rt_call = [&](double* t){
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
        t[2] = chunked(n, chunk_rt, [&](int off, int cnt){
            CostParams p;
            p.handle = handle; p.pos = d_pos_rt; p.radius = d_rad_rt; p.counts = d_counts_rt; p.is_counts = d_iscnt;
            p.n = n; p.offset = off;
            CK(cudaMemcpyAsync(d_params, &p, sizeof(p), cudaMemcpyHostToDevice, 0));
            OK(optixLaunch(pipeline, 0, (CUdeviceptr)d_params, sizeof(CostParams), &sbt, (unsigned)cnt, 1, 1));
        });
    };

    auto stats = [&](const int* v, unsigned long long& sum, int& mx){
        CK(cudaMemset(d_sum, 0, sizeof(unsigned long long)));
        CK(cudaMemset(d_mx, 0, sizeof(int)));
        sumKernel<<<blocksFor(n), BLOCK>>>(v, n, d_sum, d_mx);
        CK(cudaMemcpy(&sum, d_sum, sizeof(sum), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&mx, d_mx, sizeof(mx), cudaMemcpyDeviceToHost));
    };

    double tg[3], tr[3];
    grid_call(tg);
    rt_call(tr);
    CK(cudaDeviceSynchronize());

    unsigned long long sum_g, sum_r, sum_cand = 0, sum_is = 0;
    int max_g, max_r, max_cand = 0, max_is = 0;
    stats(d_counts_grid, sum_g, max_g);
    stats(d_counts_rt, sum_r, max_r);
    unsigned long long warp_cand = 0, warp_is = 0, warp_nbr = 0;
#ifdef COST_INSTR
    stats(d_cand, sum_cand, max_cand);
    stats(d_iscnt, sum_is, max_is);
    {
        std::vector<int> hc(n), hi(n), hn(n);
        CK(cudaMemcpy(hc.data(), d_cand, n*sizeof(int), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hi.data(), d_iscnt, n*sizeof(int), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hn.data(), d_counts_rt, n*sizeof(int), cudaMemcpyDeviceToHost));
        for(int w = 0; w < n; w += 32){
            int mc = 0, mi = 0, mn = 0, e = std::min(n, w + 32);
            for(int i = w; i < e; i++){ mc = std::max(mc, hc[i]); mi = std::max(mi, hi[i]); mn = std::max(mn, hn[i]); }
            warp_cand += 32ull*mc; warp_is += 32ull*mi; warp_nbr += 32ull*mn;
        }
    }
#endif
    std::vector<int> cg(n), crt(n);
    CK(cudaMemcpy(cg.data(), d_counts_grid, n*sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(crt.data(), d_counts_rt, n*sizeof(int), cudaMemcpyDeviceToHost));
    int mismatched = 0, worst = 0;
    {
        std::vector<float3> pg_h(n), pr_h(n);
        CK(cudaMemcpy(pg_h.data(), d_pos, n*sizeof(float3), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(pr_h.data(), d_pos_rt, n*sizeof(float3), cudaMemcpyDeviceToHost));
        bool same = true;
        for(int i = 0; i < n && same; i++) same = pg_h[i].x == pr_h[i].x && pg_h[i].y == pr_h[i].y && pg_h[i].z == pr_h[i].z;
        if(!same){ fprintf(stderr, "rt order differs from grid order\n"); exit(1); }
    }
    for(int i = 0; i < n; i++){
        if(crt[i] != cg[i]) mismatched++;
        worst = std::max(worst, abs(crt[i] - cg[i]));
    }
    bool valid = worst <= 1 && mismatched <= std::max(1, n/1000);

    double warmed = 0;
    int warm_iters = 0;
    while(warmed < a.warm_s*1000.0 && warm_iters < 1000){
        grid_call(tg);
        rt_call(tr);
        warmed += tg[0] + tg[1] + tg[2] + tr[0] + tr[1] + tr[2];
        warm_iters++;
    }

    std::vector<double> G0, G1, G2, R0, R1, R2;
    for(int r = 0; r < a.reps; r++){
        grid_call(tg);
        rt_call(tr);
        G0.push_back(tg[0]); G1.push_back(tg[1]); G2.push_back(tg[2]);
        R0.push_back(tr[0]); R1.push_back(tr[1]); R2.push_back(tr[2]);
    }

    double loop_bvh = -1, loop_sched = -1, loop_gsort = -1;
    if(a.loop > 0){
        Timer tm;
        tm.start();
        for(int r = 0; r < a.loop; r++){
            aabbKernel<<<blocksFor(n), BLOCK>>>(d_pos_rt, d_rad_rt, d_aabb, n);
            OK(optixAccelBuild(ctx, 0, &ao, &bi, 1, (CUdeviceptr)d_gtemp, bs.tempSizeInBytes, (CUdeviceptr)d_gout, bs.outputSizeInBytes, &handle, nullptr, 0));
        }
        loop_bvh = tm.stop()/a.loop;
        tm.start();
        for(int r = 0; r < a.loop; r++){
            mortonKernel<<<blocksFor(n), BLOCK>>>(d_pos_rt, d_mk, n, inv_mcell);
            CK(cub::DeviceRadixSort::SortPairs(d_msort, msort_bytes, d_mk, d_mk_sorted, d_iota, d_perm, n, 0, 30));
            gatherKernel<<<blocksFor(n), BLOCK>>>(d_perm, d_pos_rt, d_rad_rt, d_pos_rt2, d_rad_rt2, n);
            std::swap(d_pos_rt, d_pos_rt2);
            std::swap(d_rad_rt, d_rad_rt2);
        }
        loop_sched = tm.stop()/a.loop;
        tm.start();
        for(int r = 0; r < a.loop; r++){
            cellIdKernel<<<blocksFor(n), BLOCK>>>(d_pos, d_cell_id, n, inv_cell, gdim);
            CK(cub::DeviceRadixSort::SortPairs(d_gsort, gsort_bytes, d_cell_id, d_sorted_keys, d_iota, d_cell_particles, n, 0, cbits));
        }
        loop_gsort = tm.stop()/a.loop;
    }

    int dev = 0;
    cudaDeviceProp prop;
    CK(cudaGetDevice(&dev));
    CK(cudaGetDeviceProperties(&prop, dev));
    std::string gpu = prop.name;
    for(auto& ch : gpu) if(ch == ' ') ch = '_';

#ifdef COST_INSTR
    const char* build = "instr";
#else
    const char* build = "timing";
#endif
    printf("COST build=%s gpu=%s dist=%s radius=%s k=%.3f n=%d box=%.3f rmin=%.6f rmax=%.6f gdim=%d cells=%lld cbits=%d reps=%d valid=%d mismatched=%d worst=%d "
           "sum_nbr=%llu max_nbr=%d sum_grid_nbr=%llu sum_cand=%llu max_cand=%d sum_is=%llu max_is=%d "
           "grid_sort_ms=%.5f grid_cellstart_ms=%.5f grid_pass_ms=%.5f rt_sched_ms=%.5f rt_bvh_ms=%.5f rt_trace_ms=%.5f "
           "grid_sort_min=%.5f grid_cellstart_min=%.5f grid_pass_min=%.5f rt_sched_min=%.5f rt_bvh_min=%.5f rt_trace_min=%.5f "
           "loop=%d loop_rt_bvh_ms=%.5f loop_rt_sched_ms=%.5f loop_grid_sort_ms=%.5f warm_iters=%d "
           "warp_cand=%llu warp_is=%llu warp_nbr=%llu\n",
           build, gpu.c_str(), a.dist.c_str(), a.k > 0 ? "k" : a.radius.c_str(), a.k, n, a.box, rmin, rmax, gdim, num_cells, cbits, a.reps,
           valid ? 1 : 0, mismatched, worst,
           sum_r, max_r, sum_g, sum_cand, max_cand, sum_is, max_is,
           median(G0), median(G1), median(G2), median(R0), median(R1), median(R2),
           vmin(G0), vmin(G1), vmin(G2), vmin(R0), vmin(R1), vmin(R2),
           a.loop, loop_bvh, loop_sched, loop_gsort, warm_iters, warp_cand, warp_is, warp_nbr);
    return valid ? 0 : 2;
}
