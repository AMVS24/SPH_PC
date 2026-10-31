#define BUILDING_CUDAPHYSICS
#include "cudaPhysics.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#ifdef GPU_DOUBLE
typedef double real;
#else
typedef float real;
#endif
#define R(x) ((real)(x))

#define CK(call) do{ cudaError_t e_ = (call); if(e_ != cudaSuccess){ fprintf(stderr, "CUDA error: %s (%s:%d): %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } }while(0)

struct rvec3 { real x, y, z; };

struct KParams {
    real H, particle_mass, target_density, pressure_multiplier;
    real gravity, viscosity_alpha, wall_restitution, cell_size;
    real kernel_norm, grad_norm, c0, cutoff2;
    real grid_min_x, grid_min_y, grid_min_z;
    real min_x, max_x, min_y, max_y, min_z, max_z;
    int grid_w, grid_h, grid_d;
};

static const int BLOCK = 256;
static const double PI_D = 3.14159265358979323846;

static GpuParams g_p;
static KParams g_kp;
static int g_n = 0, g_n_fluid = 0, g_num_cells = 0, g_sort_bits = 1;
static rvec3 *d_pos = nullptr, *d_vel = nullptr, *d_acc = nullptr;
static real *d_density = nullptr, *d_pressure = nullptr;
static int *d_cell_id = nullptr, *d_sorted_keys = nullptr, *d_cell_particles = nullptr, *d_cell_start = nullptr, *d_iota = nullptr;
static int* d_scratch = nullptr;
static void* d_sort_temp = nullptr;
static size_t g_sort_temp_bytes = 0;
static size_t g_alloc_bytes = 0;

#define ALLOC(ptr, bytes) do{ CK(cudaMalloc(&(ptr), (bytes))); g_alloc_bytes += (bytes); }while(0)

static int blocksFor(int count){ return (count + BLOCK - 1) / BLOCK; }

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

__global__ void cellIdKernel(KParams p, const rvec3* pos, int* cell_id, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;
    int cx = (int)((pos[i].x - p.grid_min_x)/p.cell_size);
    int cy = (int)((pos[i].y - p.grid_min_y)/p.cell_size);
    int cz = (int)((pos[i].z - p.grid_min_z)/p.cell_size);
    if(cx < 0) cx = 0; else if(cx >= p.grid_w) cx = p.grid_w-1;
    if(cy < 0) cy = 0; else if(cy >= p.grid_h) cy = p.grid_h-1;
    if(cz < 0) cz = 0; else if(cz >= p.grid_d) cz = p.grid_d-1;
    cell_id[i] = cx + cy*p.grid_w + cz*p.grid_w*p.grid_h;
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

__global__ void densityKernel(KParams p, const rvec3* pos, const int* cell_id, const int* cell_start, const int* cell_particles, real* density, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;

    int cx = cell_id[i] % p.grid_w;
    int cy = (cell_id[i] / p.grid_w) % p.grid_h;
    int cz = cell_id[i] / (p.grid_w*p.grid_h);
    real xi = pos[i].x, yi = pos[i].y, zi = pos[i].z;
    real sum = R(0);

    for(int dz = -1; dz <= 1; dz++){
        int nz = cz + dz;
        if(nz < 0 || nz >= p.grid_d) continue;
        for(int dy = -1; dy <= 1; dy++){
            int ny = cy + dy;
            if(ny < 0 || ny >= p.grid_h) continue;
            for(int dx = -1; dx <= 1; dx++){
                int nx = cx + dx;
                if(nx < 0 || nx >= p.grid_w) continue;
                int c = nx + ny*p.grid_w + nz*p.grid_w*p.grid_h;

                for(int k = cell_start[c]; k < cell_start[c+1]; k++){
                    int j = cell_particles[k];
                    real rx = pos[j].x - xi;
                    real ry = pos[j].y - yi;
                    real rz = pos[j].z - zi;
                    real r2 = rx*rx + ry*ry + rz*rz;
                    if(r2 < p.cutoff2) sum += p.particle_mass*kernelW(sqrt(r2)/p.H, p.kernel_norm);
                }
            }
        }
    }
    density[i] = sum;
}

__global__ void pressureKernel(KParams p, const real* density, real* pressure, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;
    pressure[i] = (density[i] - p.target_density)*p.pressure_multiplier;
}

__global__ void forceKernel(KParams p, const rvec3* pos, const rvec3* vel, const real* density, const real* pressure,
                            const int* cell_id, const int* cell_start, const int* cell_particles, rvec3* acc, int n_fluid){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n_fluid) return;

    int cx = cell_id[i] % p.grid_w;
    int cy = (cell_id[i] / p.grid_w) % p.grid_h;
    int cz = cell_id[i] / (p.grid_w*p.grid_h);
    real xi = pos[i].x, yi = pos[i].y, zi = pos[i].z;
    real vxi = vel[i].x, vyi = vel[i].y, vzi = vel[i].z;
    real rho_i = density[i], p_i = pressure[i];
    real ax = R(0), ay = R(0), az = R(0);

    for(int dz = -1; dz <= 1; dz++){
        int nz = cz + dz;
        if(nz < 0 || nz >= p.grid_d) continue;
        for(int dy = -1; dy <= 1; dy++){
            int ny = cy + dy;
            if(ny < 0 || ny >= p.grid_h) continue;
            for(int dx = -1; dx <= 1; dx++){
                int nx = cx + dx;
                if(nx < 0 || nx >= p.grid_w) continue;
                int c = nx + ny*p.grid_w + nz*p.grid_w*p.grid_h;

                for(int k = cell_start[c]; k < cell_start[c+1]; k++){
                    int j = cell_particles[k];
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
            }
        }
    }
    ay -= p.gravity;
    acc[i].x = ax;
    acc[i].y = ay;
    acc[i].z = az;
}

__global__ void densitySearchKernel(KParams p, const rvec3* pos, const int* cell_id, const int* cell_start, const int* cell_particles, int* scratch, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;
    int cx = cell_id[i] % p.grid_w;
    int cy = (cell_id[i] / p.grid_w) % p.grid_h;
    int cz = cell_id[i] / (p.grid_w*p.grid_h);
    real xi = pos[i].x, yi = pos[i].y, zi = pos[i].z;
    int count = 0;
    for(int dz = -1; dz <= 1; dz++){
        int nz = cz + dz;
        if(nz < 0 || nz >= p.grid_d) continue;
        for(int dy = -1; dy <= 1; dy++){
            int ny = cy + dy;
            if(ny < 0 || ny >= p.grid_h) continue;
            for(int dx = -1; dx <= 1; dx++){
                int nx = cx + dx;
                if(nx < 0 || nx >= p.grid_w) continue;
                int c = nx + ny*p.grid_w + nz*p.grid_w*p.grid_h;
                for(int k = cell_start[c]; k < cell_start[c+1]; k++){
                    int j = cell_particles[k];
                    real rx = pos[j].x - xi;
                    real ry = pos[j].y - yi;
                    real rz = pos[j].z - zi;
                    real r2 = rx*rx + ry*ry + rz*rz;
                    if(r2 < p.cutoff2) count++;
                }
            }
        }
    }
    scratch[i] = count;
}

__global__ void forceSearchKernel(KParams p, const rvec3* pos, const real* density, const int* cell_id, const int* cell_start, const int* cell_particles, int* scratch, int n_fluid){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n_fluid) return;
    int cx = cell_id[i] % p.grid_w;
    int cy = (cell_id[i] / p.grid_w) % p.grid_h;
    int cz = cell_id[i] / (p.grid_w*p.grid_h);
    real xi = pos[i].x, yi = pos[i].y, zi = pos[i].z;
    int count = 0;
    for(int dz = -1; dz <= 1; dz++){
        int nz = cz + dz;
        if(nz < 0 || nz >= p.grid_d) continue;
        for(int dy = -1; dy <= 1; dy++){
            int ny = cy + dy;
            if(ny < 0 || ny >= p.grid_h) continue;
            for(int dx = -1; dx <= 1; dx++){
                int nx = cx + dx;
                if(nx < 0 || nx >= p.grid_w) continue;
                int c = nx + ny*p.grid_w + nz*p.grid_w*p.grid_h;
                for(int k = cell_start[c]; k < cell_start[c+1]; k++){
                    int j = cell_particles[k];
                    if(i == j) continue;
                    real rx = pos[j].x - xi;
                    real ry = pos[j].y - yi;
                    real rz = pos[j].z - zi;
                    real r2 = rx*rx + ry*ry + rz*rz;
                    if(r2 == R(0) || r2 >= p.cutoff2) continue;
                    if(density[j] == R(0)) continue;
                    count++;
                }
            }
        }
    }
    scratch[i] = count;
}

__global__ void kickKernel(rvec3* vel, const rvec3* acc, int n_fluid, real dt){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n_fluid) return;
    vel[i].x += acc[i].x*dt;
    vel[i].y += acc[i].y*dt;
    vel[i].z += acc[i].z*dt;
}

static std::vector<rvec3> toReal(const vec3* a, int n){
    std::vector<rvec3> out(n);
    for(int i = 0; i < n; i++) out[i] = rvec3{R(a[i].x), R(a[i].y), R(a[i].z)};
    return out;
}

static void fromReal(const rvec3* d_src, vec3* out, int n){
    std::vector<rvec3> tmp(n);
    CK(cudaMemcpy(tmp.data(), d_src, n*sizeof(rvec3), cudaMemcpyDeviceToHost));
    for(int i = 0; i < n; i++) out[i] = vec3{(double)tmp[i].x, (double)tmp[i].y, (double)tmp[i].z};
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
    cudaFree(d_pos); cudaFree(d_vel); cudaFree(d_acc);
    cudaFree(d_density); cudaFree(d_pressure);
    cudaFree(d_cell_id); cudaFree(d_sorted_keys); cudaFree(d_cell_particles); cudaFree(d_cell_start); cudaFree(d_iota);
    cudaFree(d_sort_temp); cudaFree(d_scratch);
    d_pos = d_vel = d_acc = nullptr;
    d_density = d_pressure = nullptr;
    d_cell_id = d_sorted_keys = d_cell_particles = d_cell_start = d_iota = nullptr;
    d_sort_temp = nullptr; d_scratch = nullptr;
    g_sort_temp_bytes = 0;
    g_alloc_bytes = 0;
}

void gpuInit(int n, int n_fluid, const vec3* position_array, const vec3* velocity_array, const GpuParams* params){
    gpuFree();
    g_n = n;
    g_n_fluid = n_fluid;
    gpuSetParams(params);
    g_num_cells = params->grid_w*params->grid_h*params->grid_d;

    g_sort_bits = 1;
    while((1 << g_sort_bits) < g_num_cells) g_sort_bits++;

    ALLOC(d_pos, n*sizeof(rvec3));
    ALLOC(d_vel, n*sizeof(rvec3));
    ALLOC(d_acc, n*sizeof(rvec3));
    ALLOC(d_density, n*sizeof(real));
    ALLOC(d_pressure, n*sizeof(real));
    ALLOC(d_cell_id, n*sizeof(int));
    ALLOC(d_sorted_keys, n*sizeof(int));
    ALLOC(d_cell_particles, n*sizeof(int));
    ALLOC(d_iota, n*sizeof(int));
    ALLOC(d_cell_start, (g_num_cells + 1)*sizeof(int));

    std::vector<rvec3> pos_r = toReal(position_array, n);
    std::vector<rvec3> vel_r = toReal(velocity_array, n);
    CK(cudaMemcpy(d_pos, pos_r.data(), n*sizeof(rvec3), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_vel, vel_r.data(), n*sizeof(rvec3), cudaMemcpyHostToDevice));
    ALLOC(d_scratch, n*sizeof(int));
    CK(cudaMemset(d_acc, 0, n*sizeof(rvec3)));
    CK(cudaMemset(d_density, 0, n*sizeof(real)));
    CK(cudaMemset(d_pressure, 0, n*sizeof(real)));

    iotaKernel<<<blocksFor(n), BLOCK>>>(d_iota, n);
    CK(cudaGetLastError());

    CK(cub::DeviceRadixSort::SortPairs(nullptr, g_sort_temp_bytes, d_cell_id, d_sorted_keys, d_iota, d_cell_particles, n, 0, g_sort_bits));
    ALLOC(d_sort_temp, g_sort_temp_bytes);

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
    cellIdKernel<<<blocksFor(g_n), BLOCK>>>(g_kp, d_pos, d_cell_id, g_n);
    CK(cudaGetLastError());

    CK(cub::DeviceRadixSort::SortPairs(d_sort_temp, g_sort_temp_bytes, d_cell_id, d_sorted_keys, d_iota, d_cell_particles, g_n, 0, g_sort_bits));

    cellStartKernel<<<blocksFor(g_num_cells + 1), BLOCK>>>(d_sorted_keys, d_cell_start, g_n, g_num_cells);
    CK(cudaGetLastError());
}

void gpuDensity(){
    densityKernel<<<blocksFor(g_n), BLOCK>>>(g_kp, d_pos, d_cell_id, d_cell_start, d_cell_particles, d_density, g_n);
    CK(cudaGetLastError());
}

void gpuPressure(){
    pressureKernel<<<blocksFor(g_n), BLOCK>>>(g_kp, d_density, d_pressure, g_n);
    CK(cudaGetLastError());
}

void gpuForces(){
    forceKernel<<<blocksFor(g_n_fluid), BLOCK>>>(g_kp, d_pos, d_vel, d_density, d_pressure, d_cell_id, d_cell_start, d_cell_particles, d_acc, g_n_fluid);
    CK(cudaGetLastError());
}

void gpuKick(double dt){
    kickKernel<<<blocksFor(g_n_fluid), BLOCK>>>(d_vel, d_acc, g_n_fluid, R(dt));
    CK(cudaGetLastError());
}

void gpuDensitySearchOnly(){
    densitySearchKernel<<<blocksFor(g_n), BLOCK>>>(g_kp, d_pos, d_cell_id, d_cell_start, d_cell_particles, d_scratch, g_n);
    CK(cudaGetLastError());
}

void gpuForcesSearchOnly(){
    forceSearchKernel<<<blocksFor(g_n_fluid), BLOCK>>>(g_kp, d_pos, d_density, d_cell_id, d_cell_start, d_cell_particles, d_scratch, g_n_fluid);
    CK(cudaGetLastError());
}

size_t gpuAllocatedBytes(){
    return g_alloc_bytes;
}

void gpuSync(){
    CK(cudaDeviceSynchronize());
}

void gpuDownloadPositions(vec3* position_array){
    fromReal(d_pos, position_array, g_n);
}

void gpuDownloadVelocities(vec3* velocity_array){
    fromReal(d_vel, velocity_array, g_n);
}

void gpuDownloadDensity(double* density_array){
    std::vector<real> tmp(g_n);
    CK(cudaMemcpy(tmp.data(), d_density, g_n*sizeof(real), cudaMemcpyDeviceToHost));
    for(int i = 0; i < g_n; i++) density_array[i] = (double)tmp[i];
}
