#include "cuNSearch.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>

using namespace cuNSearch;
using Real3 = std::array<Real, 3>;

int main(int argc, char** argv){
    if(argc < 4){ fprintf(stderr, "usage: %s points.txt radius reps\n", argv[0]); return 1; }
    std::ifstream f(argv[1]);
    std::vector<Real3> pos;
    std::string line;
    while(std::getline(f, line)){
        for(auto& c : line) if(c == ',') c = ' ';
        std::istringstream ss(line);
        double x, y, z;
        if(ss >> x >> y >> z) pos.push_back(Real3{(Real)x, (Real)y, (Real)z});
    }
    Real radius = (Real)atof(argv[2]);
    int reps = atoi(argv[3]);
    NeighborhoodSearch ns(radius);
    auto idx = ns.add_point_set(pos.front().data(), pos.size(), true, true);
    ns.find_neighbors();
    ns.z_sort();
    ns.point_set(idx).sort_field(pos.data());
    ns.find_neighbors();
    cudaDeviceSynchronize();
    std::vector<double> t;
    for(int r = 0; r < reps; r++){
        auto a = std::chrono::high_resolution_clock::now();
        ns.find_neighbors();
        cudaDeviceSynchronize();
        auto b = std::chrono::high_resolution_clock::now();
        t.push_back(std::chrono::duration<double, std::milli>(b - a).count());
    }
    unsigned long long total = 0;
    auto& ps = ns.point_set(idx);
    for(size_t i = 0; i < pos.size(); i++) total += ps.n_neighbors(idx, (unsigned)i);
    std::sort(t.begin(), t.end());
    printf("CUNS n=%zu radius=%.6f real_bytes=%zu reps=%d min_ms=%.4f median_ms=%.4f total_nbrs=%llu mean_nbrs=%.3f\n",
           pos.size(), (double)radius, sizeof(Real), reps, t.front(), t[t.size()/2], total, (double)total/pos.size());
    return 0;
}
