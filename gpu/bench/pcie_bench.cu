#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>

typedef std::chrono::steady_clock clk;

static double gbs(size_t bytes, int reps, double ms){
    return (double)bytes*reps/(ms*1e-3)/1e9;
}

static double time_copy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, int reps){
    cudaMemcpy(dst, src, bytes, kind);
    clk::time_point t0 = clk::now();
    for(int r = 0; r < reps; r++) cudaMemcpy(dst, src, bytes, kind);
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

int main(){
    size_t sizes[] = {1200000, 12000000, 24000000, 120000000, 240000000};
    for(size_t bytes : sizes){
        void* dev; cudaMalloc(&dev, bytes);
        char* pageable = (char*)malloc(bytes); memset(pageable, 1, bytes);
        void* pinned; cudaMallocHost(&pinned, bytes); memset(pinned, 1, bytes);
        int reps = bytes < 50000000 ? 20 : 8;
        double h2d_pg = gbs(bytes, reps, time_copy(dev, pageable, bytes, cudaMemcpyHostToDevice, reps));
        double d2h_pg = gbs(bytes, reps, time_copy(pageable, dev, bytes, cudaMemcpyDeviceToHost, reps));
        double h2d_pn = gbs(bytes, reps, time_copy(dev, pinned, bytes, cudaMemcpyHostToDevice, reps));
        double d2h_pn = gbs(bytes, reps, time_copy(pinned, dev, bytes, cudaMemcpyDeviceToHost, reps));
        printf("PCIE bytes=%zu h2d_pageable=%.2f d2h_pageable=%.2f h2d_pinned=%.2f d2h_pinned=%.2f\n", bytes, h2d_pg, d2h_pg, h2d_pn, d2h_pn);
        cudaFree(dev); free(pageable); cudaFreeHost(pinned);
    }
    return 0;
}
