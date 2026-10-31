#define BUILDING_CUDAPHYSICS
#include "cudaPhysics.h"

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CK(call) do{ cudaError_t e_ = (call); if(e_ != cudaSuccess){ fprintf(stderr, "CUDA error: %s (%s:%d): %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } }while(0)

static const int BLOCK = 256;
static const double PI_D = 3.14159265358979323846;

static GpuParams g_p;
static int g_n = 0, g_n_fluid = 0, g_num_cells = 0, g_sort_bits = 1;
static vec3 *d_pos = nullptr, *d_vel = nullptr, *d_acc = nullptr;
static double *d_density = nullptr, *d_pressure = nullptr;
static int *d_cell_id = nullptr, *d_sorted_keys = nullptr, *d_cell_particles = nullptr, *d_cell_start = nullptr, *d_iota = nullptr;
static void* d_sort_temp = nullptr;
static size_t g_sort_temp_bytes = 0;

static int blocksFor(int count){ return (count + BLOCK - 1) / BLOCK; }

__global__ void iotaKernel(int* out, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) out[i] = i;
}

__global__ void driftKernel(vec3* pos, const vec3* vel, int n_fluid, double dt){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n_fluid) return;
    pos[i].x += vel[i].x*dt;
    pos[i].y += vel[i].y*dt;
    pos[i].z += vel[i].z*dt;
}

__device__ void bounceAxis(double& x, double& v, double lo, double hi, double restitution){
    if(x < lo){
        x = lo;
        v = -v*restitution;
    }
    else if(x > hi){
        x = hi;
        v = -v*restitution;
    }
}

__global__ void boundaryKernel(GpuParams p, vec3* pos, vec3* vel, int n_fluid){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n_fluid) return;
    bounceAxis(pos[i].x, vel[i].x, p.min_x, p.max_x, p.wall_restitution);
    bounceAxis(pos[i].y, vel[i].y, p.min_y, p.max_y, p.wall_restitution);
    bounceAxis(pos[i].z, vel[i].z, p.min_z, p.max_z, p.wall_restitution);
}

__global__ void cellIdKernel(GpuParams p, const vec3* pos, int* cell_id, int n){
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

__device__ double kernelW(double q, double norm){
    if(q < 1){
        double q2 = q*q;
        return norm * (1 - 1.5 * q2 + 0.75 * q2*q);
    }
    else if(q < 2){
        return norm * (0.25 * (2 - q) * (2 - q) * (2 - q));
    }
    return 0;
}

__global__ void densityKernel(GpuParams p, const vec3* pos, const int* cell_id, const int* cell_start, const int* cell_particles, double* density, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;

    int cx = cell_id[i] % p.grid_w;
    int cy = (cell_id[i] / p.grid_w) % p.grid_h;
    int cz = cell_id[i] / (p.grid_w*p.grid_h);
    double xi = pos[i].x, yi = pos[i].y, zi = pos[i].z;
    double sum = 0;

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
                    double rx = pos[j].x - xi;
                    double ry = pos[j].y - yi;
                    double rz = pos[j].z - zi;
                    double q = sqrt(rx*rx + ry*ry + rz*rz)/p.H;
                    if(q < 2) sum += p.particle_mass*kernelW(q, p.kernel_norm);
                }
            }
        }
    }
    density[i] = sum;
}

__global__ void pressureKernel(GpuParams p, const double* density, double* pressure, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;
    pressure[i] = (density[i] - p.target_density)*p.pressure_multiplier;
}

__global__ void forceKernel(GpuParams p, const vec3* pos, const vec3* vel, const double* density, const double* pressure,
                            const int* cell_id, const int* cell_start, const int* cell_particles, vec3* acc, int n_fluid){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n_fluid) return;

    int cx = cell_id[i] % p.grid_w;
    int cy = (cell_id[i] / p.grid_w) % p.grid_h;
    int cz = cell_id[i] / (p.grid_w*p.grid_h);
    double xi = pos[i].x, yi = pos[i].y, zi = pos[i].z;
    double vxi = vel[i].x, vyi = vel[i].y, vzi = vel[i].z;
    double rho_i = density[i], p_i = pressure[i];
    double ax = 0, ay = 0, az = 0;

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
                    double rho_j = density[j];
                    if(i == j || rho_j == 0) continue;

                    double rx = pos[j].x - xi;
                    double ry = pos[j].y - yi;
                    double rz = pos[j].z - zi;
                    double norm = sqrt(rx*rx + ry*ry + rz*rz);
                    if(norm == 0 || norm >= 2*p.H) continue;

                    double q = norm/p.H;
                    double coef;
                    if(q < 1) coef = p.grad_norm * (-3 * q + 2.25 * q * q);
                    else      coef = p.grad_norm * (-3 * (2 - q) * (2 - q) / 4);
                    double gx = -(rx/norm)*coef;
                    double gy = -(ry/norm)*coef;
                    double gz = -(rz/norm)*coef;

                    double pterm = p_i/(rho_i*rho_i) + pressure[j]/(rho_j*rho_j);
                    ax -= pterm*gx;
                    ay -= pterm*gy;
                    az -= pterm*gz;

                    double dot_prod = rx*(vel[j].x - vxi) + ry*(vel[j].y - vyi) + rz*(vel[j].z - vzi);
                    if(dot_prod < 0){
                        double mu = p.H*dot_prod/(norm*norm + 0.0001*p.H*p.H);
                        double avg_density = (rho_i + rho_j)*0.5;
                        double visc = (-p.viscosity_alpha*p.c0*mu)/(avg_density + 1e-9);
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

__global__ void kickKernel(vec3* vel, const vec3* acc, int n_fluid, double dt){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n_fluid) return;
    vel[i].x += acc[i].x*dt;
    vel[i].y += acc[i].y*dt;
    vel[i].z += acc[i].z*dt;
}

void gpuSetParams(const GpuParams* params){
    g_p = *params;
    g_p.kernel_norm = 1.0/(PI_D*g_p.H*g_p.H*g_p.H);
    g_p.grad_norm = 1.0/(PI_D*g_p.H*g_p.H*g_p.H*g_p.H);
    g_p.c0 = sqrt(g_p.pressure_multiplier);
}

void gpuFree(){
    cudaFree(d_pos); cudaFree(d_vel); cudaFree(d_acc);
    cudaFree(d_density); cudaFree(d_pressure);
    cudaFree(d_cell_id); cudaFree(d_sorted_keys); cudaFree(d_cell_particles); cudaFree(d_cell_start); cudaFree(d_iota);
    cudaFree(d_sort_temp);
    d_pos = d_vel = d_acc = nullptr;
    d_density = d_pressure = nullptr;
    d_cell_id = d_sorted_keys = d_cell_particles = d_cell_start = d_iota = nullptr;
    d_sort_temp = nullptr;
    g_sort_temp_bytes = 0;
}

void gpuInit(int n, int n_fluid, const vec3* position_array, const vec3* velocity_array, const GpuParams* params){
    gpuFree();
    g_n = n;
    g_n_fluid = n_fluid;
    gpuSetParams(params);
    g_num_cells = params->grid_w*params->grid_h*params->grid_d;

    g_sort_bits = 1;
    while((1 << g_sort_bits) < g_num_cells) g_sort_bits++;

    CK(cudaMalloc(&d_pos, n*sizeof(vec3)));
    CK(cudaMalloc(&d_vel, n*sizeof(vec3)));
    CK(cudaMalloc(&d_acc, n*sizeof(vec3)));
    CK(cudaMalloc(&d_density, n*sizeof(double)));
    CK(cudaMalloc(&d_pressure, n*sizeof(double)));
    CK(cudaMalloc(&d_cell_id, n*sizeof(int)));
    CK(cudaMalloc(&d_sorted_keys, n*sizeof(int)));
    CK(cudaMalloc(&d_cell_particles, n*sizeof(int)));
    CK(cudaMalloc(&d_iota, n*sizeof(int)));
    CK(cudaMalloc(&d_cell_start, (g_num_cells + 1)*sizeof(int)));

    CK(cudaMemcpy(d_pos, position_array, n*sizeof(vec3), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_vel, velocity_array, n*sizeof(vec3), cudaMemcpyHostToDevice));
    CK(cudaMemset(d_acc, 0, n*sizeof(vec3)));
    CK(cudaMemset(d_density, 0, n*sizeof(double)));
    CK(cudaMemset(d_pressure, 0, n*sizeof(double)));

    iotaKernel<<<blocksFor(n), BLOCK>>>(d_iota, n);
    CK(cudaGetLastError());

    CK(cub::DeviceRadixSort::SortPairs(nullptr, g_sort_temp_bytes, d_cell_id, d_sorted_keys, d_iota, d_cell_particles, n, 0, g_sort_bits));
    CK(cudaMalloc(&d_sort_temp, g_sort_temp_bytes));

    CK(cudaDeviceSynchronize());
}

void gpuDrift(double dt){
    driftKernel<<<blocksFor(g_n_fluid), BLOCK>>>(d_pos, d_vel, g_n_fluid, dt);
    CK(cudaGetLastError());
}

void gpuEnforceBoundary(){
    boundaryKernel<<<blocksFor(g_n_fluid), BLOCK>>>(g_p, d_pos, d_vel, g_n_fluid);
    CK(cudaGetLastError());
}

void gpuBuildGrid(){
    cellIdKernel<<<blocksFor(g_n), BLOCK>>>(g_p, d_pos, d_cell_id, g_n);
    CK(cudaGetLastError());

    CK(cub::DeviceRadixSort::SortPairs(d_sort_temp, g_sort_temp_bytes, d_cell_id, d_sorted_keys, d_iota, d_cell_particles, g_n, 0, g_sort_bits));

    cellStartKernel<<<blocksFor(g_num_cells + 1), BLOCK>>>(d_sorted_keys, d_cell_start, g_n, g_num_cells);
    CK(cudaGetLastError());
}

void gpuDensity(){
    densityKernel<<<blocksFor(g_n), BLOCK>>>(g_p, d_pos, d_cell_id, d_cell_start, d_cell_particles, d_density, g_n);
    CK(cudaGetLastError());
}

void gpuPressure(){
    pressureKernel<<<blocksFor(g_n), BLOCK>>>(g_p, d_density, d_pressure, g_n);
    CK(cudaGetLastError());
}

void gpuForces(){
    forceKernel<<<blocksFor(g_n_fluid), BLOCK>>>(g_p, d_pos, d_vel, d_density, d_pressure, d_cell_id, d_cell_start, d_cell_particles, d_acc, g_n_fluid);
    CK(cudaGetLastError());
}

void gpuKick(double dt){
    kickKernel<<<blocksFor(g_n_fluid), BLOCK>>>(d_vel, d_acc, g_n_fluid, dt);
    CK(cudaGetLastError());
}

void gpuSync(){
    CK(cudaDeviceSynchronize());
}

void gpuDownloadPositions(vec3* position_array){
    CK(cudaMemcpy(position_array, d_pos, g_n*sizeof(vec3), cudaMemcpyDeviceToHost));
}

void gpuDownloadVelocities(vec3* velocity_array){
    CK(cudaMemcpy(velocity_array, d_vel, g_n*sizeof(vec3), cudaMemcpyDeviceToHost));
}

void gpuDownloadDensity(double* density_array){
    CK(cudaMemcpy(density_array, d_density, g_n*sizeof(double), cudaMemcpyDeviceToHost));
}
