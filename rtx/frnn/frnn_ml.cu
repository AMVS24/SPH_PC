#define NOMINMAX
#define _USE_MATH_DEFINES
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
static const int MAXL = 16;
static int blocksFor(long long count){ return (int)((count + BLOCK - 1) / BLOCK); }

struct MLLevel { float lox, loy, loz, inv_cell; int gx, gy, gz, off; };
struct MLParams { int nl; float rmax; MLLevel lv[MAXL]; };

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

__device__ __forceinline__ float densW(float d2, float r){
    float h = 0.5f*r;
    float q = sqrtf(d2)/h;
    float w = q < 1.f ? 1.f - 1.5f*q*q + 0.75f*q*q*q : 0.25f*(2.f - q)*(2.f - q)*(2.f - q);
    return r*r*r*0.05236f*w/(3.14159265f*h*h*h);
}

__global__ void gridKernel(const float3* pos, const float* rad, const int* cell_id, const int* cell_start, const int* cell_particles,
                           int gdim, int offset, int count, int density, int* out){
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if(t >= count) return;
    int i = offset + t;
    int cid = cell_id[i];
    int cx = cid % gdim, cy = (cid / gdim) % gdim, cz = cid / (gdim*gdim);
    float xi = pos[i].x, yi = pos[i].y, zi = pos[i].z;
    int k = 0;
    float sum = 0.f;
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
                    float d2 = rx*rx + ry*ry + rz*rz;
                    if(d2 < r*r){
                        if(density) sum += densW(d2, r);
                        else k++;
                    }
                }
            }
        }
    }
    out[i] = density ? __float_as_int(sum) : k;
}

__global__ void mlInitKernel(int* bbox){
    int t = threadIdx.x;
    if(t < 3*MAXL) bbox[t] = 0x7f7fffff;
    else if(t < 6*MAXL) bbox[t] = 0;
    else if(t < 7*MAXL) bbox[t] = 0;
}

__global__ void mlLevelKernel(const float3* pos, const float* rad, int* level, int* bbox, int n, float rmax, int nl){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;
    float r = rad[i];
    int L = (int)floorf(log2f(rmax/r));
    L = min(max(L, 0), nl - 1);
    while(L > 0 && r > rmax*exp2f(-(float)L)) L--;
    level[i] = L;
    float3 p = pos[i];
    atomicMin(&bbox[3*L + 0], __float_as_int(p.x));
    atomicMin(&bbox[3*L + 1], __float_as_int(p.y));
    atomicMin(&bbox[3*L + 2], __float_as_int(p.z));
    atomicMax(&bbox[3*MAXL + 3*L + 0], __float_as_int(p.x));
    atomicMax(&bbox[3*MAXL + 3*L + 1], __float_as_int(p.y));
    atomicMax(&bbox[3*MAXL + 3*L + 2], __float_as_int(p.z));
    atomicAdd(&bbox[6*MAXL + L], 1);
}

__global__ void mlKeyKernel(MLParams P, const float3* pos, const int* level, int* key, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;
    const MLLevel& l = P.lv[level[i]];
    float3 p = pos[i];
    int cx = min(max((int)floorf((p.x - l.lox)*l.inv_cell), 0), l.gx - 1);
    int cy = min(max((int)floorf((p.y - l.loy)*l.inv_cell), 0), l.gy - 1);
    int cz = min(max((int)floorf((p.z - l.loz)*l.inv_cell), 0), l.gz - 1);
    key[i] = l.off + cx + l.gx*(cy + l.gy*cz);
}

__global__ void mlQueryKernel(MLParams P, const float3* spos, const float* srad, const int* cell_start, const int* perm,
                              int offset, int count, int density, int* out){
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if(t >= count) return;
    int s = offset + t;
    float3 pi = spos[s];
    int k = 0;
    float sum = 0.f;
    for(int L = 0; L < P.nl; L++){
        const MLLevel l = P.lv[L];
        if(l.gx == 0) continue;
        int cx = (int)floorf((pi.x - l.lox)*l.inv_cell);
        int cy = (int)floorf((pi.y - l.loy)*l.inv_cell);
        int cz = (int)floorf((pi.z - l.loz)*l.inv_cell);
        int x0 = max(cx - 1, 0), x1 = min(cx + 1, l.gx - 1);
        int y0 = max(cy - 1, 0), y1 = min(cy + 1, l.gy - 1);
        int z0 = max(cz - 1, 0), z1 = min(cz + 1, l.gz - 1);
        if(x0 > x1 || y0 > y1 || z0 > z1) continue;
        for(int z = z0; z <= z1; z++){
            for(int y = y0; y <= y1; y++){
                int base = l.off + l.gx*(y + l.gy*z);
                int a = cell_start[base + x0], b = cell_start[base + x1 + 1];
                for(int m = a; m < b; m++){
                    float3 pj = spos[m];
                    float rx = pj.x - pi.x, ry = pj.y - pi.y, rz = pj.z - pi.z;
                    float r = srad[m];
                    float d2 = rx*rx + ry*ry + rz*rz;
                    if(d2 < r*r){
                        if(density) sum += densW(d2, r);
                        else k++;
                    }
                }
            }
        }
    }
    out[perm[s]] = density ? __float_as_int(sum) : k;
}

__global__ void statsKernel(const int* counts, int n, unsigned long long* sum, int* mx){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;
    atomicAdd(sum, (unsigned long long)counts[i]);
    atomicMax(mx, counts[i]);
}

struct Args {
    std::string dist = "multires", rspec = "";
    int n = 50000, seed = 1;
    double box = 100.0, budget_s = 10.0, ratio = 1.0, alpha = 2.0, core = 0.01, nbrs = 80.0, single_cap_s = 5.0;
    bool density = false, skip_single = false;
};

static void scene(const Args& a, std::vector<float3>& pos, std::vector<float>& rad, double& rmin, double& rmax){
    std::mt19937_64 rng((unsigned long long)a.seed*7919ull + (unsigned long long)a.n);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    double L = a.box;
    pos.resize(a.n);
    rad.resize(a.n);
    if(a.dist == "multires"){
        double R3 = a.ratio*a.ratio*a.ratio;
        double dxc = L*cbrt((7.0/8.0 + R3/8.0)/a.n);
        double rc = 2.673*dxc, rf = rc/a.ratio;
        int nc = (int)llround(a.n*(7.0/8.0)/(7.0/8.0 + R3/8.0));
        double lo = 0.25*L, hi = 0.75*L;
        for(int idx = 0; idx < a.n; idx++){
            double x, y, z;
            if(idx < nc){
                do { x = u01(rng)*L; y = u01(rng)*L; z = u01(rng)*L; } while(x >= lo && x < hi && y >= lo && y < hi && z >= lo && z < hi);
                rad[idx] = (float)rc;
            }
            else {
                x = lo + u01(rng)*(hi - lo); y = lo + u01(rng)*(hi - lo); z = lo + u01(rng)*(hi - lo);
                rad[idx] = (float)rf;
            }
            pos[idx] = make_float3((float)x, (float)y, (float)z);
        }
    }
    else if(a.dist == "randrad"){
        for(auto& p : pos) p = make_float3((float)(u01(rng)*L), (float)(u01(rng)*L), (float)(u01(rng)*L));
        std::vector<std::string> f;
        size_t st = 0, e;
        while((e = a.rspec.find(':', st)) != std::string::npos){ f.push_back(a.rspec.substr(st, e - st)); st = e + 1; }
        f.push_back(a.rspec.substr(st));
        if(f[0] == "uniform"){
            std::uniform_real_distribution<double> ud(atof(f[1].c_str()), atof(f[2].c_str()));
            for(auto& x : rad) x = (float)ud(rng);
        }
        else if(f[0] == "lognormal"){
            std::lognormal_distribution<double> ln(atof(f[1].c_str()), atof(f[2].c_str()));
            double lo = atof(f[3].c_str()), hi = atof(f[4].c_str());
            for(auto& x : rad) x = (float)std::min(hi, std::max(lo, ln(rng)));
        }
        else if(f[0] == "const"){
            for(auto& x : rad) x = (float)atof(f[1].c_str());
        }
        else { fprintf(stderr, "bad --rspec\n"); exit(1); }
    }
    else if(a.dist == "profile"){
        double R = 0.4995*L, c = a.core*R, cx = 0.5*L;
        const int T = 200000;
        std::vector<double> rr(T + 1), cdf(T + 1);
        cdf[0] = 0.0;
        for(int k = 0; k <= T; k++) rr[k] = R*k/T;
        auto g = [&](double r){ return 4.0*M_PI*r*r*pow(r*r + c*c, -0.5*a.alpha); };
        for(int k = 1; k <= T; k++) cdf[k] = cdf[k-1] + 0.5*(g(rr[k-1]) + g(rr[k]))*(rr[k] - rr[k-1]);
        double A = a.n/cdf[T];
        std::normal_distribution<double> gn(0.0, 1.0);
        for(int idx = 0; idx < a.n; idx++){
            double u = u01(rng)*cdf[T];
            int k = (int)(std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
            k = std::min(std::max(k, 1), T);
            double f = (u - cdf[k-1])/std::max(1e-300, cdf[k] - cdf[k-1]);
            double r = rr[k-1] + f*(rr[k] - rr[k-1]);
            double vx, vy, vz, vn;
            do { vx = gn(rng); vy = gn(rng); vz = gn(rng); vn = sqrt(vx*vx + vy*vy + vz*vz); } while(vn < 1e-12);
            pos[idx] = make_float3((float)(cx + r*vx/vn), (float)(cx + r*vy/vn), (float)(cx + r*vz/vn));
            double nd = A*pow(r*r + c*c, -0.5*a.alpha);
            rad[idx] = (float)cbrt(a.nbrs/(nd*4.0/3.0*M_PI));
        }
    }
    else { fprintf(stderr, "unknown --dist %s\n", a.dist.c_str()); exit(1); }
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
        else if(k == "--n") a.n = atoi(val().c_str());
        else if(k == "--seed") a.seed = atoi(val().c_str());
        else if(k == "--box") a.box = atof(val().c_str());
        else if(k == "--budget-s") a.budget_s = atof(val().c_str());
        else if(k == "--ratio") a.ratio = atof(val().c_str());
        else if(k == "--alpha") a.alpha = atof(val().c_str());
        else if(k == "--core") a.core = atof(val().c_str());
        else if(k == "--nbrs") a.nbrs = atof(val().c_str());
        else if(k == "--single-cap-s") a.single_cap_s = atof(val().c_str());
        else if(k == "--density") a.density = true;
        else if(k == "--skip-single") a.skip_single = true;
        else if(k == "--rspec") a.rspec = val();
        else { fprintf(stderr, "unknown option %s\n", k.c_str()); return 1; }
    }
    const int n = a.n;
    const int dens = a.density ? 1 : 0;

    std::vector<float3> h_pos;
    std::vector<float> h_rad;
    double rmin, rmax;
    scene(a, h_pos, h_rad, rmin, rmax);

    float3 *d_pos0, *d_pos, *d_pos_rt, *d_pos_rt2, *d_mlpos;
    float *d_rad0, *d_rad, *d_rad_rt, *d_rad_rt2, *d_mlrad;
    unsigned int *d_mk, *d_mk_sorted;
    int *d_iota, *d_perm, *d_out_single, *d_out_rt, *d_out_ml, *d_mx;
    int *d_level, *d_mlkey, *d_mlkey_sorted, *d_mlperm, *d_bbox;
    unsigned long long* d_sum;
    CK(cudaMalloc(&d_pos0, n*sizeof(float3))); CK(cudaMalloc(&d_pos, n*sizeof(float3)));
    CK(cudaMalloc(&d_pos_rt, n*sizeof(float3))); CK(cudaMalloc(&d_pos_rt2, n*sizeof(float3)));
    CK(cudaMalloc(&d_mlpos, n*sizeof(float3)));
    CK(cudaMalloc(&d_rad0, n*sizeof(float))); CK(cudaMalloc(&d_rad, n*sizeof(float)));
    CK(cudaMalloc(&d_rad_rt, n*sizeof(float))); CK(cudaMalloc(&d_rad_rt2, n*sizeof(float)));
    CK(cudaMalloc(&d_mlrad, n*sizeof(float)));
    CK(cudaMalloc(&d_mk, n*sizeof(unsigned))); CK(cudaMalloc(&d_mk_sorted, n*sizeof(unsigned)));
    CK(cudaMalloc(&d_iota, n*sizeof(int))); CK(cudaMalloc(&d_perm, n*sizeof(int)));
    CK(cudaMalloc(&d_out_single, n*sizeof(int))); CK(cudaMalloc(&d_out_rt, n*sizeof(int))); CK(cudaMalloc(&d_out_ml, n*sizeof(int)));
    CK(cudaMalloc(&d_level, n*sizeof(int))); CK(cudaMalloc(&d_mlkey, n*sizeof(int)));
    CK(cudaMalloc(&d_mlkey_sorted, n*sizeof(int))); CK(cudaMalloc(&d_mlperm, n*sizeof(int)));
    CK(cudaMalloc(&d_bbox, 7*MAXL*sizeof(int)));
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

    int nl = std::min(MAXL, (int)ceil(log2(rmax/rmin) + 1e-9) + 1);
    size_t mlsort_bytes = 0;
    void* d_mlsort = nullptr;
    CK(cub::DeviceRadixSort::SortPairs(nullptr, mlsort_bytes, d_mlkey, d_mlkey_sorted, d_iota, d_mlperm, n, 0, 31));
    CK(cudaMalloc(&d_mlsort, mlsort_bytes));
    int* d_mlcell_start = nullptr;
    long long ml_alloc_cells = 0;
    MLParams P = {};
    long long ml_cells = 0;
    int ml_used_levels = 0;

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

    long long chunk_single = std::max(1024, n/1024), chunk_rt = std::max(1024, n/1024), chunk_ml = std::max(1024, n/1024);

    auto single_call = [&](double* t){
        Timer tm;
        tm.start();
        cellIdKernel<<<blocksFor(n), BLOCK>>>(d_pos, d_cell_id, n, inv_cell, gdim);
        CK(cub::DeviceRadixSort::SortPairs(d_gsort, gsort_bytes, d_cell_id, d_sorted_keys, d_iota, d_cell_particles, n, 0, cbits));
        cellStartKernel<<<blocksFor(num_cells + 1), BLOCK>>>(d_sorted_keys, d_cell_start, n, num_cells);
        t[0] = tm.stop();
        t[1] = chunked(n, chunk_single, [&](int off, int cnt){
            gridKernel<<<blocksFor(cnt), BLOCK>>>(d_pos, d_rad, d_cell_id, d_cell_start, d_cell_particles, gdim, off, cnt, dens, d_out_single);
        });
        CK(cudaGetLastError());
    };

    auto ml_call = [&](double* t){
        Timer tm;
        tm.start();
        mlInitKernel<<<1, 7*MAXL>>>(d_bbox);
        mlLevelKernel<<<blocksFor(n), BLOCK>>>(d_pos, d_rad, d_level, d_bbox, n, (float)rmax, nl);
        int hb[7*MAXL];
        CK(cudaMemcpy(hb, d_bbox, sizeof(hb), cudaMemcpyDeviceToHost));
        P.nl = nl;
        P.rmax = (float)rmax;
        long long off = 0;
        ml_used_levels = 0;
        for(int L = 0; L < MAXL; L++){
            MLLevel& l = P.lv[L];
            l = MLLevel{0, 0, 0, 0, 0, 0, 0, (int)off};
            if(L >= nl || hb[6*MAXL + L] == 0) continue;
            float lo[3], hi[3];
            for(int c = 0; c < 3; c++){ memcpy(&lo[c], &hb[3*L + c], 4); memcpy(&hi[c], &hb[3*MAXL + 3*L + c], 4); }
            double cl = rmax*pow(2.0, -L);
            double ext = std::max({(double)hi[0] - lo[0], (double)hi[1] - lo[1], (double)hi[2] - lo[2]});
            cl = std::max(cl, ext/1000.0);
            int g[3];
            for(int c = 0; c < 3; c++) g[c] = (int)floor(((double)hi[c] - lo[c])/cl) + 1;
            l.lox = lo[0]; l.loy = lo[1]; l.loz = lo[2];
            l.inv_cell = (float)(1.0/cl);
            l.gx = g[0]; l.gy = g[1]; l.gz = g[2];
            l.off = (int)off;
            off += (long long)g[0]*g[1]*g[2];
            ml_used_levels++;
        }
        if(off > (1ll << 30)){ fprintf(stderr, "multi-level grid too large: %lld cells\n", off); exit(1); }
        ml_cells = off;
        if(off + 1 > ml_alloc_cells){
            if(d_mlcell_start) CK(cudaFree(d_mlcell_start));
            ml_alloc_cells = off + 1;
            CK(cudaMalloc(&d_mlcell_start, ml_alloc_cells*sizeof(int)));
        }
        int bits = 1;
        while((1ll << bits) < off + 1) bits++;
        mlKeyKernel<<<blocksFor(n), BLOCK>>>(P, d_pos, d_level, d_mlkey, n);
        CK(cub::DeviceRadixSort::SortPairs(d_mlsort, mlsort_bytes, d_mlkey, d_mlkey_sorted, d_iota, d_mlperm, n, 0, bits));
        gatherKernel<<<blocksFor(n), BLOCK>>>(d_mlperm, d_pos, d_rad, d_mlpos, d_mlrad, n);
        cellStartKernel<<<blocksFor(off + 1), BLOCK>>>(d_mlkey_sorted, d_mlcell_start, n, (int)off);
        t[0] = tm.stop();
        t[1] = chunked(n, chunk_ml, [&](int o, int cnt){
            mlQueryKernel<<<blocksFor(cnt), BLOCK>>>(P, d_mlpos, d_mlrad, d_mlcell_start, d_mlperm, o, cnt, dens, d_out_ml);
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
            FrnnParams p;
            p.handle = handle; p.pos = d_pos_rt; p.radius = d_rad_rt; p.counts = d_out_rt; p.nbr = nullptr;
            p.n = n; p.cap = 0; p.offset = off; p.mode = dens ? 2 : 0;
            CK(cudaMemcpyAsync(d_params, &p, sizeof(p), cudaMemcpyHostToDevice, 0));
            OK(optixLaunch(pipeline, 0, (CUdeviceptr)d_params, sizeof(FrnnParams), &sbt, (unsigned)cnt, 1, 1));
        });
    };

    auto fetch_rt_in_base_order = [&](std::vector<int>& out){
        std::vector<int> tmp(n);
        CK(cudaMemcpy(tmp.data(), d_out_rt, n*sizeof(int), cudaMemcpyDeviceToHost));
        out = tmp;
    };

    double ts[2] = {0, 0}, tml[2], trt[3];
    bool run_single = !a.skip_single;
    ml_call(tml);
    rt_call(trt);
    if(run_single){
        single_call(ts);
        if(ts[0] + ts[1] > a.single_cap_s*1000.0) run_single = false;
    }
    bool single_measured = ts[0] + ts[1] > 0;

    std::vector<int> cm(n), cr(n), cs;
    CK(cudaMemcpy(cm.data(), d_out_ml, n*sizeof(int), cudaMemcpyDeviceToHost));
    fetch_rt_in_base_order(cr);
    if(single_measured){
        cs.resize(n);
        CK(cudaMemcpy(cs.data(), d_out_single, n*sizeof(int), cudaMemcpyDeviceToHost));
    }

    auto compare = [&](const std::vector<int>& ref, const std::vector<int>& x, int& mism, double& worst){
        mism = 0; worst = 0;
        for(int i = 0; i < n; i++){
            if(dens){
                float g, r;
                memcpy(&g, &ref[i], 4); memcpy(&r, &x[i], 4);
                double rel = fabs((double)r - g)/std::max(1e-12, fabs((double)g));
                worst = std::max(worst, rel);
                if(rel > 1e-3) mism++;
            }
            else {
                int dd = abs(ref[i] - x[i]);
                worst = std::max(worst, (double)dd);
                if(dd) mism++;
            }
        }
    };
    int mism_rt, mism_single = -1;
    double worst_rt, worst_single = -1;
    compare(cm, cr, mism_rt, worst_rt);
    if(single_measured) compare(cm, cs, mism_single, worst_single);
    auto ok = [&](int mism, double worst){ return dens ? mism <= std::max(1, n/1000) : (worst <= 1 && mism <= std::max(1, n/1000)); };
    bool valid = ok(mism_rt, worst_rt) && (!single_measured || ok(mism_single, worst_single));

    double mean_q = 0;
    int max_q = 0;
    double rho_sum = 0;
    if(!dens){
        unsigned long long s = 0;
        for(int i = 0; i < n; i++){ s += cm[i]; max_q = std::max(max_q, cm[i]); }
        mean_q = (double)s/n;
    }
    else {
        for(int i = 0; i < n; i++){ float r; memcpy(&r, &cm[i], 4); rho_sum += r; }
    }

    std::vector<double> S0, S1, M0, M1, R0, R1, R2;
    if(single_measured && !run_single){ S0.push_back(ts[0]); S1.push_back(ts[1]); }
    double spent = ts[0] + ts[1] + tml[0] + tml[1] + trt[0] + trt[1] + trt[2];
    int reps = 0;
    while(reps < 3 || (spent < a.budget_s*1000.0 && reps < 10)){
        ml_call(tml);
        rt_call(trt);
        M0.push_back(tml[0]); M1.push_back(tml[1]);
        R0.push_back(trt[0]); R1.push_back(trt[1]); R2.push_back(trt[2]);
        spent += tml[0] + tml[1] + trt[0] + trt[1] + trt[2];
        if(run_single){
            single_call(ts);
            S0.push_back(ts[0]); S1.push_back(ts[1]);
            spent += ts[0] + ts[1];
        }
        reps++;
        if(spent > 4.0*a.budget_s*1000.0) break;
    }

    auto mean = [](const std::vector<double>& v){ double s = 0; for(double x : v) s += x; return v.empty() ? -1.0 : s/v.size(); };
    double s_tot = single_measured ? mean(S0) + mean(S1) : -1.0;
    double m_tot = mean(M0) + mean(M1);
    double r_tot = mean(R0) + mean(R1) + mean(R2);

    printf("FRNNML mode=%s dist=%s n=%d box=%.1f ratio=%.2f alpha=%.2f core=%.4f nbrs_target=%.1f rmin=%.5f rmax=%.5f hrange=%.2f "
           "valid=%d mism_rt=%d worst_rt=%.3e mism_single=%d worst_single=%.3e mean_nbrs=%.3f max_nbrs=%d mean_density=%.5f "
           "gdim_single=%d ml_levels=%d ml_cells=%lld reps=%zu single_reps=%zu "
           "single_build_ms=%.4f single_pass_ms=%.4f ml_build_ms=%.4f ml_pass_ms=%.4f rt_sched_ms=%.4f rt_bvh_ms=%.4f rt_trace_ms=%.4f "
           "single_total_ms=%.4f ml_total_ms=%.4f rt_total_ms=%.4f\n",
           dens ? "density" : "count", a.dist.c_str(), n, a.box, a.ratio, a.alpha, a.core, a.nbrs, rmin, rmax, rmax/rmin,
           valid ? 1 : 0, mism_rt, worst_rt, mism_single, worst_single, mean_q, max_q, dens ? rho_sum/n : -1.0,
           gdim, ml_used_levels, ml_cells, M0.size(), S0.size(),
           single_measured ? mean(S0) : -1.0, single_measured ? mean(S1) : -1.0, mean(M0), mean(M1), mean(R0), mean(R1), mean(R2),
           s_tot, m_tot, r_tot);
    return valid ? 0 : 2;
}
