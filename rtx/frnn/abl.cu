#define NOMINMAX
#include "ablShared.h"

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

#ifdef ABL_INSTR
#include "abl_m0_instr.h"
#include "abl_m1_instr.h"
#include "abl_m2_instr.h"
#define BLOB0 abl_m0_instr
#define BLOB1 abl_m1_instr
#define BLOB2 abl_m2_instr
#define NINSTR 2
#else
#include "abl_m0.h"
#include "abl_m1.h"
#include "abl_m2.h"
#define BLOB0 abl_m0
#define BLOB1 abl_m1
#define BLOB2 abl_m2
#define NINSTR 0
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

__global__ void gatherKernel(const int* perm, const float3* p_in, const float* r_in, const float3* v_in, const float* rho_in, const float* a_in,
                             float3* p_out, float* r_out, float3* v_out, float* rho_out, float* a_out, int n){
    int s = blockIdx.x*blockDim.x + threadIdx.x;
    if(s >= n) return;
    int o = perm[s];
    p_out[s] = p_in[o];
    r_out[s] = r_in[o];
    v_out[s] = v_in[o];
    rho_out[s] = rho_in[o];
    a_out[s] = a_in[o];
}

__global__ void aabbKernel(const float3* pos, const float* rad, OptixAabb* aabb, int n, float scale){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;
    float h = rad[i]*scale;
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

template<int MODE>
__global__ void gridKernel(const float3* pos, const float* rad, const float3* vel, const float* rho, const float* A,
                           const int* cell_id, const int* cell_start, const int* cell_particles,
                           int gdim, int S, int offset, int count, int* counts, float* out, int* cand){
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if(t >= count) return;
    int i = offset + t;
    int cid = cell_id[i];
    int cx = cid % gdim, cy = (cid / gdim) % gdim, cz = cid / (gdim*gdim);
    float3 xi = pos[i];
    int k = 0;
    float s0 = 0.f, ax = 0.f, ay = 0.f, az = 0.f;
    float3 vi = make_float3(0.f, 0.f, 0.f);
    float Ai = 0.f, rhoi = 0.f;
    if(MODE == 2){ vi = vel[i]; Ai = A[i]; rhoi = rho[i]; }
#ifdef ABL_INSTR
    int tests = 0;
#endif
    for(int dz = -S; dz <= S; dz++){
        int nz = cz + dz;
        if(nz < 0 || nz >= gdim) continue;
        for(int dy = -S; dy <= S; dy++){
            int ny = cy + dy;
            if(ny < 0 || ny >= gdim) continue;
            for(int dx = -S; dx <= S; dx++){
                int nx = cx + dx;
                if(nx < 0 || nx >= gdim) continue;
                int c = nx + ny*gdim + nz*gdim*gdim;
                int e = cell_start[c+1];
#ifdef ABL_INSTR
                tests += e - cell_start[c];
#endif
                for(int m = cell_start[c]; m < e; m++){
                    int j = cell_particles[m];
                    float3 p = pos[j];
                    float rx = p.x - xi.x, ry = p.y - xi.y, rz = p.z - xi.z;
                    float r = rad[j];
                    float d2 = rx*rx + ry*ry + rz*rz;
                    if(!(d2 < r*r)) continue;
                    if(MODE == 0) k++;
                    else if(MODE == 1) s0 += ablW(d2, r);
                    else ablForce(xi, vi, Ai, rhoi, p, vel[j], A[j], rho[j], r, d2, ax, ay, az);
                }
            }
        }
    }
    if(MODE == 0) counts[i] = k;
    else if(MODE == 1) out[i] = s0;
    else { out[3*(size_t)i] = ax; out[3*(size_t)i + 1] = ay; out[3*(size_t)i + 2] = az; }
#ifdef ABL_INSTR
    cand[i] = tests;
#endif
}

__device__ __forceinline__ int lowerBound(const int* a, int n, int v){
    int lo = 0, hi = n;
    while(lo < hi){
        int mid = (lo + hi) >> 1;
        if(a[mid] < v) lo = mid + 1; else hi = mid;
    }
    return lo;
}

template<int MODE>
__global__ void gridCompactKernel(const float3* pos, const float* rad, const float3* vel, const float* rho, const float* A,
                                  const int* cell_id, const int* sorted_keys, const int* cell_particles, int ntot,
                                  int gdim, int S, int offset, int count, int* counts, float* out, int* cand){
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if(t >= count) return;
    int i = offset + t;
    int cid = cell_id[i];
    int cx = cid % gdim, cy = (cid / gdim) % gdim, cz = cid / (gdim*gdim);
    float3 xi = pos[i];
    int k = 0;
    float s0 = 0.f, ax = 0.f, ay = 0.f, az = 0.f;
    float3 vi = make_float3(0.f, 0.f, 0.f);
    float Ai = 0.f, rhoi = 0.f;
    if(MODE == 2){ vi = vel[i]; Ai = A[i]; rhoi = rho[i]; }
#ifdef ABL_INSTR
    int tests = 0;
#endif
    int x0 = max(cx - S, 0), x1 = min(cx + S, gdim - 1);
    for(int dz = -S; dz <= S; dz++){
        int nz = cz + dz;
        if(nz < 0 || nz >= gdim) continue;
        for(int dy = -S; dy <= S; dy++){
            int ny = cy + dy;
            if(ny < 0 || ny >= gdim) continue;
            int row = ny*gdim + nz*gdim*gdim;
            int b = lowerBound(sorted_keys, ntot, row + x0);
            int e = lowerBound(sorted_keys, ntot, row + x1 + 1);
#ifdef ABL_INSTR
            tests += e - b;
#endif
            for(int m = b; m < e; m++){
                int j = cell_particles[m];
                float3 p = pos[j];
                float rx = p.x - xi.x, ry = p.y - xi.y, rz = p.z - xi.z;
                float r = rad[j];
                float d2 = rx*rx + ry*ry + rz*rz;
                if(!(d2 < r*r)) continue;
                if(MODE == 0) k++;
                else if(MODE == 1) s0 += ablW(d2, r);
                else ablForce(xi, vi, Ai, rhoi, p, vel[j], A[j], rho[j], r, d2, ax, ay, az);
            }
        }
    }
    if(MODE == 0) counts[i] = k;
    else if(MODE == 1) out[i] = s0;
    else { out[3*(size_t)i] = ax; out[3*(size_t)i + 1] = ay; out[3*(size_t)i + 2] = az; }
#ifdef ABL_INSTR
    cand[i] = tests;
#endif
}

__global__ void gridCountD(const float3* pos, const float* rad, const int* cell_id, const int* cell_start, const int* cell_particles,
                           int gdim, int S, int offset, int count, int* counts){
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if(t >= count) return;
    int i = offset + t;
    int cid = cell_id[i];
    int cx = cid % gdim, cy = (cid / gdim) % gdim, cz = cid / (gdim*gdim);
    double xi = pos[i].x, yi = pos[i].y, zi = pos[i].z;
    int k = 0;
    for(int dz = -S; dz <= S; dz++){
        int nz = cz + dz;
        if(nz < 0 || nz >= gdim) continue;
        for(int dy = -S; dy <= S; dy++){
            int ny = cy + dy;
            if(ny < 0 || ny >= gdim) continue;
            for(int dx = -S; dx <= S; dx++){
                int nx = cx + dx;
                if(nx < 0 || nx >= gdim) continue;
                int c = nx + ny*gdim + nz*gdim*gdim;
                int e = cell_start[c+1];
                for(int m = cell_start[c]; m < e; m++){
                    int j = cell_particles[m];
                    double rx = (double)pos[j].x - xi, ry = (double)pos[j].y - yi, rz = (double)pos[j].z - zi;
                    double r = rad[j];
                    if(rx*rx + ry*ry + rz*rz < r*r) k++;
                }
            }
        }
    }
    counts[i] = k;
}

template<typename T>
__global__ void gridLJ(const float3* pos, const float* rad, const int* cell_id, const int* cell_start, const int* cell_particles,
                       int gdim, int S, int offset, int count, int* counts, float* out){
    int t = blockIdx.x*blockDim.x + threadIdx.x;
    if(t >= count) return;
    int i = offset + t;
    int cid = cell_id[i];
    int cx = cid % gdim, cy = (cid / gdim) % gdim, cz = cid / (gdim*gdim);
    T xi = pos[i].x, yi = pos[i].y, zi = pos[i].z;
    T fx = 0, fy = 0, fz = 0;
    int k = 0;
    for(int dz = -S; dz <= S; dz++){
        int nz = cz + dz;
        if(nz < 0 || nz >= gdim) continue;
        for(int dy = -S; dy <= S; dy++){
            int ny = cy + dy;
            if(ny < 0 || ny >= gdim) continue;
            for(int dx = -S; dx <= S; dx++){
                int nx = cx + dx;
                if(nx < 0 || nx >= gdim) continue;
                int c = nx + ny*gdim + nz*gdim*gdim;
                int e = cell_start[c+1];
                for(int m = cell_start[c]; m < e; m++){
                    int j = cell_particles[m];
                    T rx = (T)pos[j].x - xi, ry = (T)pos[j].y - yi, rz = (T)pos[j].z - zi;
                    T r = rad[j];
                    T d2 = rx*rx + ry*ry + rz*rz;
                    if(!(d2 < r*r)) continue;
                    k++;
                    if(d2 <= (T)0) continue;
                    T sg = r/(T)2.5;
                    T s2 = sg*sg/d2;
                    T s6 = s2*s2*s2;
                    T f = (T)24*s6*((T)2*s6 - (T)1)/d2;
                    fx -= f*rx; fy -= f*ry; fz -= f*rz;
                }
            }
        }
    }
    counts[i] = k;
    out[i] = (float)sqrt((double)(fx*fx + fy*fy + fz*fz));
}

__global__ void sumKernel(const int* v, int n, unsigned long long* sum, int* mx){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;
    atomicAdd(sum, (unsigned long long)v[i]);
    atomicMax(mx, v[i]);
}

struct Args {
    std::string dist = "disordered", radius = "", variant = "base", mode = "count", file = "";
    int n = 1000000, seed = 1, clusters = 4, reps = 7, div = 1;
    double box = 103.0, sigma = 25.0, k = -1.0, warm_s = 1.0, alpha = 3.0, core = 0.01;
    bool no_grid = false, compact = false, fp64 = false, xfer = false, lj = false;
    int kcap = 50;
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
    else if(a.dist == "surface"){
        std::normal_distribution<double> gn(0.0, 1.0);
        auto unit = [&](){ double x, y, z, m; do { x = gn(rng); y = gn(rng); z = gn(rng); m = sqrt(x*x + y*y + z*z); } while(m < 1e-9); return make_double3(x/m, y/m, z/m); };
        const int NP = 8, NS = 4;
        std::vector<double3> pc(NP), pn(NP), su(NP), sv(NP), sc(NS);
        std::vector<double> sr(NS);
        for(int q = 0; q < NP; q++){
            pc[q] = make_double3(L*(0.2 + 0.6*u01(rng)), L*(0.2 + 0.6*u01(rng)), L*(0.2 + 0.6*u01(rng)));
            pn[q] = unit();
            double3 t = fabs(pn[q].x) < 0.9 ? make_double3(1, 0, 0) : make_double3(0, 1, 0);
            double3 u = make_double3(pn[q].y*t.z - pn[q].z*t.y, pn[q].z*t.x - pn[q].x*t.z, pn[q].x*t.y - pn[q].y*t.x);
            double um = sqrt(u.x*u.x + u.y*u.y + u.z*u.z);
            su[q] = make_double3(u.x/um, u.y/um, u.z/um);
            sv[q] = make_double3(pn[q].y*su[q].z - pn[q].z*su[q].y, pn[q].z*su[q].x - pn[q].x*su[q].z, pn[q].x*su[q].y - pn[q].y*su[q].x);
        }
        for(int q = 0; q < NS; q++){
            sc[q] = make_double3(L*(0.3 + 0.4*u01(rng)), L*(0.3 + 0.4*u01(rng)), L*(0.3 + 0.4*u01(rng)));
            sr[q] = L*(0.1 + 0.15*u01(rng));
        }
        for(int idx = 0; idx < a.n; idx++){
            double x, y, z;
            do {
                if(idx % 2 == 0){
                    int q = (int)(u01(rng)*NP) % NP;
                    double rr = 0.4*L*sqrt(u01(rng)), th = 2.0*3.14159265358979*u01(rng);
                    double a1 = rr*cos(th), a2 = rr*sin(th), e = 0.002*gn(rng);
                    x = pc[q].x + a1*su[q].x + a2*sv[q].x + e*pn[q].x;
                    y = pc[q].y + a1*su[q].y + a2*sv[q].y + e*pn[q].y;
                    z = pc[q].z + a1*su[q].z + a2*sv[q].z + e*pn[q].z;
                }
                else {
                    int q = (int)(u01(rng)*NS) % NS;
                    double3 d = unit();
                    double rr = sr[q]*(1.0 + 0.0001*gn(rng));
                    x = sc[q].x + rr*d.x; y = sc[q].y + rr*d.y; z = sc[q].z + rr*d.z;
                }
            } while(x < 0 || x >= L || y < 0 || y >= L || z < 0 || z >= L);
            pos[idx] = make_float3((float)x, (float)y, (float)z);
        }
    }
    else if(a.dist == "file"){
        FILE* fp = fopen(a.file.c_str(), "r");
        if(!fp){ fprintf(stderr, "cannot open %s\n", a.file.c_str()); exit(1); }
        pos.clear();
        double x, y, z;
        while(fscanf(fp, "%lf,%lf,%lf", &x, &y, &z) == 3) pos.push_back(make_float3((float)x, (float)y, (float)z));
        fclose(fp);
        if((int)pos.size() != a.n){ fprintf(stderr, "file has %zu points, --n says %d\n", pos.size(), a.n); exit(1); }
    }
    else if(a.dist == "profile"){
        double R = 0.4995*L, c = a.core*R, cx = 0.5*L;
        const int T = 200000;
        std::vector<double> rr(T + 1), cdf(T + 1);
        cdf[0] = 0.0;
        for(int q = 0; q <= T; q++) rr[q] = R*q/T;
        auto g = [&](double r){ return 4.0*3.14159265358979*r*r*pow(r*r + c*c, -0.5*a.alpha); };
        for(int q = 1; q <= T; q++) cdf[q] = cdf[q-1] + 0.5*(g(rr[q-1]) + g(rr[q]))*(rr[q] - rr[q-1]);
        double A = a.n/cdf[T];
        std::normal_distribution<double> gn(0.0, 1.0);
        double nb = a.k > 0 ? a.k : 80.0;
        for(int idx = 0; idx < a.n; idx++){
            double u = u01(rng)*cdf[T];
            int q = (int)(std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
            q = std::min(std::max(q, 1), T);
            double f = (u - cdf[q-1])/std::max(1e-300, cdf[q] - cdf[q-1]);
            double r = rr[q-1] + f*(rr[q] - rr[q-1]);
            double vx, vy, vz, vn;
            do { vx = gn(rng); vy = gn(rng); vz = gn(rng); vn = sqrt(vx*vx + vy*vy + vz*vz); } while(vn < 1e-12);
            pos[idx] = make_float3((float)(cx + r*vx/vn), (float)(cx + r*vy/vn), (float)(cx + r*vz/vn));
            double nd = A*pow(r*r + c*c, -0.5*a.alpha);
            rad[idx] = (float)cbrt(nb/(nd*4.0/3.0*3.14159265358979));
        }
        rmin = 1e30; rmax = 0;
        for(float x : rad){ rmin = std::min(rmin, (double)x); rmax = std::max(rmax, (double)x); }
        return;
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
        else if(k == "--warm-s") a.warm_s = atof(val().c_str());
        else if(k == "--variant") a.variant = val();
        else if(k == "--mode") a.mode = val();
        else if(k == "--grid-div") a.div = atoi(val().c_str());
        else if(k == "--alpha") a.alpha = atof(val().c_str());
        else if(k == "--core") a.core = atof(val().c_str());
        else if(k == "--no-grid") a.no_grid = true;
        else if(k == "--file") a.file = val();
        else if(k == "--grid-fp64") a.fp64 = true;
        else if(k == "--grid-lj") a.lj = true;
        else if(k == "--xfer") a.xfer = true;
        else if(k == "--kcap") a.kcap = atoi(val().c_str());
        else if(k == "--grid-compact") a.compact = true;
        else { fprintf(stderr, "unknown option %s\n", k.c_str()); return 1; }
    }
    const int n = a.n;
    const int mode = a.mode == "count" ? 0 : a.mode == "density" ? 1 : a.mode == "force" ? 2 : -1;
    if(mode < 0){ fprintf(stderr, "unknown --mode\n"); return 1; }
    const std::string& V = a.variant;
    const bool is_sphere = V == "sphere" || V == "sphereany" || V == "spherefront";
    const bool uses_ah = V == "report" || V == "kcap" || is_sphere;
    const bool checkable = !(V == "empty" || V == "tiny" || V == "kcap" || a.no_grid);
    if(!(V == "base" || V == "empty" || V == "report" || V == "fasttrace" || V == "unsorted" || V == "tiny" || V == "sphere" || V == "sphereany" || V == "spherefront" || V == "aabbseg" || V == "refit" || V == "kcap")){
        fprintf(stderr, "unknown --variant\n"); return 1;
    }
    if(V == "empty" && mode != 0){ fprintf(stderr, "empty variant is count-only\n"); return 1; }

    std::vector<float3> h_pos;
    std::vector<float> h_rad;
    double rmin, rmax;
    scene(a, h_pos, h_rad, rmin, rmax);
    std::vector<float3> h_vel(n);
    std::vector<float> h_rho(n), h_A(n);
    {
        std::mt19937_64 rng(12345ull + (unsigned long long)n);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        for(int i = 0; i < n; i++){
            h_vel[i] = make_float3((float)u(rng), (float)u(rng), (float)u(rng));
            double rho = 1.0 + 0.05*u(rng);
            h_rho[i] = (float)rho;
            h_A[i] = (float)(100.0*(rho - 1.0)/(rho*rho));
        }
    }

    float3 *d_p0, *d_v0, *d_pos, *d_vel, *d_pos2, *d_vel2;
    float *d_r0, *d_rho0, *d_A0, *d_rad, *d_rho, *d_A, *d_rad2, *d_rho2, *d_A2;
    unsigned int *d_mk, *d_mk_sorted;
    int *d_iota, *d_perm, *d_qidx = nullptr, *d_counts_g, *d_counts_r, *d_cand, *d_calls, *d_calls2, *d_mx;
    float *d_out_g, *d_out_r;
    unsigned long long* d_sum;
    size_t outn = (size_t)n*(mode == 2 ? 3 : 1);
    CK(cudaMalloc(&d_p0, n*sizeof(float3))); CK(cudaMalloc(&d_v0, n*sizeof(float3)));
    CK(cudaMalloc(&d_pos, n*sizeof(float3))); CK(cudaMalloc(&d_vel, n*sizeof(float3)));
    CK(cudaMalloc(&d_pos2, n*sizeof(float3))); CK(cudaMalloc(&d_vel2, n*sizeof(float3)));
    CK(cudaMalloc(&d_r0, n*sizeof(float))); CK(cudaMalloc(&d_rho0, n*sizeof(float))); CK(cudaMalloc(&d_A0, n*sizeof(float)));
    CK(cudaMalloc(&d_rad, n*sizeof(float))); CK(cudaMalloc(&d_rho, n*sizeof(float))); CK(cudaMalloc(&d_A, n*sizeof(float)));
    CK(cudaMalloc(&d_rad2, n*sizeof(float))); CK(cudaMalloc(&d_rho2, n*sizeof(float))); CK(cudaMalloc(&d_A2, n*sizeof(float)));
    CK(cudaMalloc(&d_mk, n*sizeof(unsigned))); CK(cudaMalloc(&d_mk_sorted, n*sizeof(unsigned)));
    CK(cudaMalloc(&d_iota, n*sizeof(int))); CK(cudaMalloc(&d_perm, n*sizeof(int)));
    CK(cudaMalloc(&d_counts_g, n*sizeof(int))); CK(cudaMalloc(&d_counts_r, n*sizeof(int)));
    CK(cudaMalloc(&d_cand, n*sizeof(int))); CK(cudaMalloc(&d_calls, n*sizeof(int))); CK(cudaMalloc(&d_calls2, n*sizeof(int)));
    CK(cudaMalloc(&d_out_g, outn*sizeof(float))); CK(cudaMalloc(&d_out_r, outn*sizeof(float)));
    CK(cudaMalloc(&d_mx, sizeof(int))); CK(cudaMalloc(&d_sum, sizeof(unsigned long long)));
    CK(cudaMemcpy(d_p0, h_pos.data(), n*sizeof(float3), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_r0, h_rad.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_v0, h_vel.data(), n*sizeof(float3), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_rho0, h_rho.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_A0, h_A.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemset(d_cand, 0, n*sizeof(int))); CK(cudaMemset(d_calls, 0, n*sizeof(int))); CK(cudaMemset(d_calls2, 0, n*sizeof(int)));
    CK(cudaMemset(d_counts_g, 0, n*sizeof(int))); CK(cudaMemset(d_counts_r, 0, n*sizeof(int)));
    CK(cudaMemset(d_out_g, 0, outn*sizeof(float))); CK(cudaMemset(d_out_r, 0, outn*sizeof(float)));
    iotaKernel<<<blocksFor(n), BLOCK>>>(d_iota, n);

    float inv_mcell = (float)(1024.0/a.box);
    size_t msort_bytes = 0;
    void* d_msort = nullptr;
    CK(cub::DeviceRadixSort::SortPairs(nullptr, msort_bytes, d_mk, d_mk_sorted, d_iota, d_perm, n, 0, 30));
    CK(cudaMalloc(&d_msort, msort_bytes));
    mortonKernel<<<blocksFor(n), BLOCK>>>(d_p0, d_mk, n, inv_mcell);
    CK(cub::DeviceRadixSort::SortPairs(d_msort, msort_bytes, d_mk, d_mk_sorted, d_iota, d_perm, n, 0, 30));
    gatherKernel<<<blocksFor(n), BLOCK>>>(d_perm, d_p0, d_r0, d_v0, d_rho0, d_A0, d_pos, d_rad, d_vel, d_rho, d_A, n);
    CK(cudaDeviceSynchronize());

    if(V == "unsorted"){
        std::vector<int> q(n);
        for(int i = 0; i < n; i++) q[i] = i;
        std::mt19937_64 rng(777ull);
        std::shuffle(q.begin(), q.end(), rng);
        CK(cudaMalloc(&d_qidx, n*sizeof(int)));
        CK(cudaMemcpy(d_qidx, q.data(), n*sizeof(int), cudaMemcpyHostToDevice));
    }

    double cell = rmax/a.div;
    int gdim = std::max(1, (int)floor(a.box/cell));
    if(gdim > 1024) gdim = 1024;
    float inv_cell = (float)(gdim/a.box);
    int S = (int)ceil(rmax*gdim/a.box - 1e-9);
    long long num_cells = (long long)gdim*gdim*gdim;
    int cbits = 1;
    while((1ll << cbits) < num_cells) cbits++;
    int *d_cell_id, *d_sorted_keys, *d_cell_particles, *d_cell_start;
    CK(cudaMalloc(&d_cell_id, n*sizeof(int))); CK(cudaMalloc(&d_sorted_keys, n*sizeof(int)));
    CK(cudaMalloc(&d_cell_particles, n*sizeof(int))); CK(cudaMalloc(&d_cell_start, (a.compact ? 1 : num_cells + 1)*sizeof(int)));
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
    pco.numPayloadValues = (mode == 2 ? 8 : 1) + NINSTR;
    pco.numAttributeValues = 2;
    pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pco.pipelineLaunchParamsVariableName = "params";
    pco.usesPrimitiveTypeFlags = is_sphere ? OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE : OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
    const unsigned char* blob = mode == 0 ? BLOB0 : mode == 1 ? BLOB1 : BLOB2;
    size_t blob_size = mode == 0 ? sizeof(BLOB0) : mode == 1 ? sizeof(BLOB1) : sizeof(BLOB2);
    OptixModule module;
    OK(optixModuleCreate(ctx, &mco, &pco, (const char*)blob, blob_size, g_log, &g_log_size, &module));

    OptixAccelBuildOptions ao = {};
    ao.buildFlags = V == "fasttrace" ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE : OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
    if(V == "refit") ao.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    bool built_once = false;
    ao.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixModule sphere_module = nullptr;
    if(is_sphere){
        OptixBuiltinISOptions bo = {};
        bo.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
        bo.buildFlags = ao.buildFlags;
        OK(optixBuiltinISModuleGet(ctx, &mco, &pco, &bo, &sphere_module));
    }

    OptixProgramGroupOptions pgo = {};
    OptixProgramGroup pg[3];
    OptixProgramGroupDesc d = {};
    d.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d.raygen.module = module; d.raygen.entryFunctionName = "__raygen__q";
    OK(optixProgramGroupCreate(ctx, &d, 1, &pgo, g_log, &g_log_size, &pg[0]));
    d = {}; d.kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d.miss.module = module; d.miss.entryFunctionName = "__miss__q";
    OK(optixProgramGroupCreate(ctx, &d, 1, &pgo, g_log, &g_log_size, &pg[1]));
    d = {}; d.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    if(is_sphere){
        d.hitgroup.moduleIS = sphere_module; d.hitgroup.entryFunctionNameIS = nullptr;
        d.hitgroup.moduleAH = module; d.hitgroup.entryFunctionNameAH = V == "sphere" ? "__anyhit__sphere" : V == "sphereany" ? "__anyhit__sphereany" : "__anyhit__spherefront";
    }
    else if(V == "report"){
        d.hitgroup.moduleIS = module; d.hitgroup.entryFunctionNameIS = "__intersection__report";
        d.hitgroup.moduleAH = module; d.hitgroup.entryFunctionNameAH = "__anyhit__report";
    }
    else if(V == "empty"){
        d.hitgroup.moduleIS = module; d.hitgroup.entryFunctionNameIS = "__intersection__empty";
    }
    else if(V == "aabbseg"){
        d.hitgroup.moduleIS = module; d.hitgroup.entryFunctionNameIS = "__intersection__seg";
    }
    else if(V == "kcap"){
        d.hitgroup.moduleIS = module; d.hitgroup.entryFunctionNameIS = "__intersection__kcap";
        d.hitgroup.moduleAH = module; d.hitgroup.entryFunctionNameAH = "__anyhit__term";
    }
    else {
        d.hitgroup.moduleIS = module; d.hitgroup.entryFunctionNameIS = "__intersection__base";
    }
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
    CUdeviceptr vert_ptr = (CUdeviceptr)d_pos, rad_ptr = (CUdeviceptr)d_rad;
    unsigned int gflags[1] = { V == "report" ? (unsigned)OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL : (unsigned)OPTIX_GEOMETRY_FLAG_NONE };
    OptixBuildInput bi = {};
    if(is_sphere){
        bi.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
        bi.sphereArray.vertexBuffers = &vert_ptr;
        bi.sphereArray.vertexStrideInBytes = sizeof(float3);
        bi.sphereArray.numVertices = (unsigned)n;
        bi.sphereArray.radiusBuffers = &rad_ptr;
        bi.sphereArray.radiusStrideInBytes = sizeof(float);
        bi.sphereArray.singleRadius = 0;
        bi.sphereArray.flags = gflags;
        bi.sphereArray.numSbtRecords = 1;
    }
    else {
        bi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        bi.customPrimitiveArray.aabbBuffers = &aabb_ptr;
        bi.customPrimitiveArray.numPrimitives = (unsigned)n;
        bi.customPrimitiveArray.strideInBytes = sizeof(OptixAabb);
        bi.customPrimitiveArray.flags = gflags;
        bi.customPrimitiveArray.numSbtRecords = 1;
    }
    OptixAccelBufferSizes bs;
    OK(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs));
    void *d_gtemp, *d_gout, *d_params;
    size_t gtemp_bytes = std::max(bs.tempSizeInBytes, bs.tempUpdateSizeInBytes);
    CK(cudaMalloc(&d_gtemp, gtemp_bytes));
    CK(cudaMalloc(&d_gout, bs.outputSizeInBytes));
    CK(cudaMalloc(&d_params, sizeof(AblParams)));
    const float aabb_scale = V == "tiny" ? 1e-4f : 1.0005f;
    const float tmax = (is_sphere || V == "aabbseg") ? (float)(2.0*rmax*1.001) : 1e-16f;
    const unsigned rayflags = OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | (uses_ah ? 0u : (unsigned)OPTIX_RAY_FLAG_DISABLE_ANYHIT);

    long long chunk_grid = std::max(1024, n/1024), chunk_rt = std::max(1024, n/1024);

    auto grid_call = [&](double* t){
        if(a.no_grid){ t[0] = t[1] = t[2] = 0.0; return; }
        Timer tm;
        tm.start();
        cellIdKernel<<<blocksFor(n), BLOCK>>>(d_pos, d_cell_id, n, inv_cell, gdim);
        CK(cub::DeviceRadixSort::SortPairs(d_gsort, gsort_bytes, d_cell_id, d_sorted_keys, d_iota, d_cell_particles, n, 0, cbits));
        t[0] = tm.stop();
        tm.start();
        if(!a.compact) cellStartKernel<<<blocksFor(num_cells + 1), BLOCK>>>(d_sorted_keys, d_cell_start, n, num_cells);
        t[1] = tm.stop();
        CK(cudaGetLastError());
        t[2] = chunked(n, chunk_grid, [&](int off, int cnt){
            if(a.compact){
                if(mode == 0) gridCompactKernel<0><<<blocksFor(cnt), BLOCK>>>(d_pos, d_rad, d_vel, d_rho, d_A, d_cell_id, d_sorted_keys, d_cell_particles, n, gdim, S, off, cnt, d_counts_g, d_out_g, d_cand);
                else if(mode == 1) gridCompactKernel<1><<<blocksFor(cnt), BLOCK>>>(d_pos, d_rad, d_vel, d_rho, d_A, d_cell_id, d_sorted_keys, d_cell_particles, n, gdim, S, off, cnt, d_counts_g, d_out_g, d_cand);
                else gridCompactKernel<2><<<blocksFor(cnt), BLOCK>>>(d_pos, d_rad, d_vel, d_rho, d_A, d_cell_id, d_sorted_keys, d_cell_particles, n, gdim, S, off, cnt, d_counts_g, d_out_g, d_cand);
                return;
            }
            if(a.lj && mode == 0){
                if(a.fp64) gridLJ<double><<<blocksFor(cnt), BLOCK>>>(d_pos, d_rad, d_cell_id, d_cell_start, d_cell_particles, gdim, S, off, cnt, d_counts_g, d_out_g);
                else gridLJ<float><<<blocksFor(cnt), BLOCK>>>(d_pos, d_rad, d_cell_id, d_cell_start, d_cell_particles, gdim, S, off, cnt, d_counts_g, d_out_g);
                return;
            }
            if(a.fp64 && mode == 0){
                gridCountD<<<blocksFor(cnt), BLOCK>>>(d_pos, d_rad, d_cell_id, d_cell_start, d_cell_particles, gdim, S, off, cnt, d_counts_g);
                return;
            }
            if(mode == 0) gridKernel<0><<<blocksFor(cnt), BLOCK>>>(d_pos, d_rad, d_vel, d_rho, d_A, d_cell_id, d_cell_start, d_cell_particles, gdim, S, off, cnt, d_counts_g, d_out_g, d_cand);
            else if(mode == 1) gridKernel<1><<<blocksFor(cnt), BLOCK>>>(d_pos, d_rad, d_vel, d_rho, d_A, d_cell_id, d_cell_start, d_cell_particles, gdim, S, off, cnt, d_counts_g, d_out_g, d_cand);
            else gridKernel<2><<<blocksFor(cnt), BLOCK>>>(d_pos, d_rad, d_vel, d_rho, d_A, d_cell_id, d_cell_start, d_cell_particles, gdim, S, off, cnt, d_counts_g, d_out_g, d_cand);
        });
        CK(cudaGetLastError());
    };

    OptixTraversableHandle handle = 0;
    auto rt_call = [&](double* t){
        Timer tm;
        tm.start();
        mortonKernel<<<blocksFor(n), BLOCK>>>(d_pos, d_mk, n, inv_mcell);
        CK(cub::DeviceRadixSort::SortPairs(d_msort, msort_bytes, d_mk, d_mk_sorted, d_iota, d_perm, n, 0, 30));
        gatherKernel<<<blocksFor(n), BLOCK>>>(d_perm, d_pos, d_rad, d_vel, d_rho, d_A, d_pos2, d_rad2, d_vel2, d_rho2, d_A2, n);
        std::swap(d_pos, d_pos2); std::swap(d_rad, d_rad2); std::swap(d_vel, d_vel2); std::swap(d_rho, d_rho2); std::swap(d_A, d_A2);
        t[0] = tm.stop();
        tm.start();
        if(is_sphere){ vert_ptr = (CUdeviceptr)d_pos; rad_ptr = (CUdeviceptr)d_rad; }
        else aabbKernel<<<blocksFor(n), BLOCK>>>(d_pos, d_rad, d_aabb, n, aabb_scale);
        ao.operation = (V == "refit" && built_once) ? OPTIX_BUILD_OPERATION_UPDATE : OPTIX_BUILD_OPERATION_BUILD;
        OK(optixAccelBuild(ctx, 0, &ao, &bi, 1, (CUdeviceptr)d_gtemp, gtemp_bytes, (CUdeviceptr)d_gout, bs.outputSizeInBytes, &handle, nullptr, 0));
        built_once = true;
        t[1] = tm.stop();
        t[2] = chunked(n, chunk_rt, [&](int off, int cnt){
            AblParams p;
            p.handle = handle; p.pos = d_pos; p.rad = d_rad; p.vel = d_vel; p.rho = d_rho; p.A = d_A; p.qidx = d_qidx;
            p.counts = d_counts_r; p.out = d_out_r; p.calls = d_calls; p.calls2 = d_calls2;
            p.n = n; p.offset = off; p.tmax = tmax; p.rayflags = rayflags; p.shift = (V == "spherefront" || V == "aabbseg") ? tmax : 0.f; p.kcap = (unsigned)a.kcap;
            CK(cudaMemcpyAsync(d_params, &p, sizeof(p), cudaMemcpyHostToDevice, 0));
            OK(optixLaunch(pipeline, 0, (CUdeviceptr)d_params, sizeof(AblParams), &sbt, (unsigned)cnt, 1, 1));
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

    unsigned long long sum_g = 0, sum_r = 0, sum_cand = 0, sum_calls = 0, sum_calls2 = 0;
    int max_g = 0, max_r = 0, max_cand = 0, max_calls = 0, max_calls2 = 0;
    if(mode == 0){ stats(d_counts_g, sum_g, max_g); stats(d_counts_r, sum_r, max_r); }
#ifdef ABL_INSTR
    stats(d_cand, sum_cand, max_cand);
    stats(d_calls, sum_calls, max_calls);
    stats(d_calls2, sum_calls2, max_calls2);
#endif

    int mismatched = 0;
    double worst = 0.0;
    if(mode == 0){
        std::vector<int> cg(n), cr(n);
        CK(cudaMemcpy(cg.data(), d_counts_g, n*sizeof(int), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(cr.data(), d_counts_r, n*sizeof(int), cudaMemcpyDeviceToHost));
        for(int i = 0; i < n; i++){
            if(cg[i] != cr[i]) mismatched++;
            worst = std::max(worst, (double)abs(cg[i] - cr[i]));
        }
    }
    else {
        std::vector<float> og(outn), orr(outn);
        CK(cudaMemcpy(og.data(), d_out_g, outn*sizeof(float), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(orr.data(), d_out_r, outn*sizeof(float), cudaMemcpyDeviceToHost));
        int c = mode == 2 ? 3 : 1;
        double scale = 0;
        for(int i = 0; i < n; i++){
            double m2 = 0;
            for(int q = 0; q < c; q++) m2 += (double)og[(size_t)c*i + q]*og[(size_t)c*i + q];
            scale += sqrt(m2);
        }
        scale /= n;
        for(int i = 0; i < n; i++){
            double e2 = 0, m2 = 0;
            for(int q = 0; q < c; q++){
                double dd = (double)og[(size_t)c*i + q] - orr[(size_t)c*i + q];
                e2 += dd*dd;
                m2 += (double)og[(size_t)c*i + q]*og[(size_t)c*i + q];
            }
            double rel = sqrt(e2)/(sqrt(m2) + 0.01*scale + 1e-30);
            worst = std::max(worst, rel);
            if(rel > 1e-3) mismatched++;
        }
    }
    int valid = checkable ? ((mode == 0 ? (worst <= 1 && mismatched <= std::max(1, n/1000)) : (mismatched <= std::max(1, n/1000))) ? 1 : 0) : -1;

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
    std::vector<double> RS, GS;
    for(size_t r = 0; r < G2.size(); r++){ RS.push_back(R1[r] + R2[r]); GS.push_back(G0[r] + G1[r] + G2[r]); }
    auto cv = [](const std::vector<double>& v){
        double m = 0, s = 0;
        for(double x : v) m += x;
        m /= v.size();
        for(double x : v) s += (x - m)*(x - m);
        return v.size() > 1 ? sqrt(s/(v.size() - 1))/m : 0.0;
    };

    double xfer_d2h = -1, xfer_h2d = -1;
    if(a.xfer){
        size_t nb = (size_t)sum_g*4 + (size_t)n*8;
        size_t pb = (size_t)n*3*(a.fp64 ? 8 : 4);
        std::vector<char> hb(nb), hp(pb);
        void *db1, *db2;
        CK(cudaMalloc(&db1, nb)); CK(cudaMalloc(&db2, pb));
        std::vector<double> X1, X2;
        for(int r = 0; r < 5; r++){
            Timer tm;
            tm.start(); CK(cudaMemcpy(hb.data(), db1, nb, cudaMemcpyDeviceToHost)); X1.push_back(tm.stop());
            tm.start(); CK(cudaMemcpy(db2, hp.data(), pb, cudaMemcpyHostToDevice)); X2.push_back(tm.stop());
        }
        xfer_d2h = median(X1); xfer_h2d = median(X2);
        CK(cudaFree(db1)); CK(cudaFree(db2));
    }
    cudaDeviceProp prop;
    int dev = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaGetDeviceProperties(&prop, dev));
    std::string gpu = prop.name;
    for(auto& ch : gpu) if(ch == ' ') ch = '_';
#ifdef ABL_INSTR
    const char* build = "instr";
#else
    const char* build = "timing";
#endif
    printf("ABL build=%s gpu=%s variant=%s mode=%s div=%d compact=%d fp64=%d lj=%d kcap=%d xfer_d2h_ms=%.4f xfer_h2d_ms=%.4f dist=%s alpha=%.2f core=%.5f radius=%s k=%.3f n=%d box=%.3f rmin=%.6f rmax=%.6f gdim=%d S=%d reps=%d valid=%d mismatched=%d worst=%.3e "
           "sum_grid_nbr=%llu sum_rt_nbr=%llu sum_cand=%llu sum_calls=%llu sum_calls2=%llu max_calls=%d "
           "grid_sort_ms=%.5f grid_cellstart_ms=%.5f grid_pass_ms=%.5f rt_sched_ms=%.5f rt_bvh_ms=%.5f rt_trace_ms=%.5f "
           "grid_pass_min=%.5f rt_bvh_min=%.5f rt_trace_min=%.5f grid_total_cv=%.4f rt_bt_cv=%.4f warm_iters=%d\n",
           build, gpu.c_str(), V.c_str(), a.mode.c_str(), a.div, a.compact ? 1 : 0, a.fp64 ? 1 : 0, a.lj ? 1 : 0, V == "kcap" ? a.kcap : 0, xfer_d2h, xfer_h2d, a.dist.c_str(), a.alpha, a.core, a.k > 0 ? "k" : a.radius.c_str(), a.k, n, a.box, rmin, rmax, gdim, S, a.reps,
           valid, mismatched, worst, sum_g, sum_r, sum_cand, sum_calls, sum_calls2, max_calls,
           median(G0), median(G1), median(G2), median(R0), median(R1), median(R2),
           vmin(G2), vmin(R1), vmin(R2), cv(GS), cv(RS), warm_iters);
    return valid == 0 ? 2 : 0;
}
