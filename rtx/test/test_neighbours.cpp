#include <rtxPhysics.h>

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

static GpuParams make_params(){
    GpuParams p = {};
    p.H = 2.0;
    p.particle_mass = 1.0;
    p.target_density = 1.0;
    p.pressure_multiplier = 1.0;
    p.gravity = 0.0;
    p.viscosity_alpha = 0.0;
    p.wall_restitution = 0.5;
    p.cell_size = 4.0;
    p.grid_w = p.grid_h = p.grid_d = 50;
    p.grid_min_x = p.grid_min_y = p.grid_min_z = -100.0;
    p.min_x = p.min_y = p.min_z = -100.0;
    p.max_x = p.max_y = p.max_z = 100.0;
    return p;
}

static std::vector<vec3> uniform_scene(int n, double half, unsigned seed){
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-half, half);
    std::vector<vec3> pts(n);
    for(auto& p : pts) p = vec3{u(rng), u(rng), u(rng)};
    return pts;
}

static std::vector<int> brute_counts(const std::vector<vec3>& pts, float cutoff2){
    int n = (int)pts.size();
    std::vector<float> x(n), y(n), z(n);
    for(int i = 0; i < n; i++){ x[i] = (float)pts[i].x; y[i] = (float)pts[i].y; z[i] = (float)pts[i].z; }
    std::vector<int> c(n, 0);
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < n; i++){
        int k = 0;
        for(int j = 0; j < n; j++){
            float dx = x[j] - x[i], dy = y[j] - y[i], dz = z[j] - z[i];
            if(dx*dx + dy*dy + dz*dz < cutoff2) k++;
        }
        c[i] = k;
    }
    return c;
}

static int run(const char* name, const std::vector<vec3>& pts, bool check_density){
    GpuParams p = make_params();
    int n = (int)pts.size();
    std::vector<vec3> vel(n, vec3{0, 0, 0});
    gpuInit(n, n, pts.data(), vel.data(), &p);
    gpuBuildGrid();
    gpuSync();

    std::vector<int> got(n);
    gpuDownloadNeighbourCounts(got.data());
    std::vector<int> ref = brute_counts(pts, (float)(4.0*p.H*p.H));

    int mismatches = 0, too_many = 0, too_few = 0;
    long long ref_total = 0, ref_overflow = 0;
    for(int i = 0; i < n; i++){
        ref_total += ref[i];
        if(ref[i] > gpuNeighbourCapacity()) ref_overflow += ref[i] - gpuNeighbourCapacity();
        if(got[i] != ref[i]){
            mismatches++;
            if(got[i] > ref[i]) too_many++; else too_few++;
        }
    }
    printf("[%s] n=%d cap=%d  count mismatches=%d (gpu>ref %d, gpu<ref %d)  total gpu=%lld ref=%lld  overflow gpu=%lld ref=%lld\n",
           name, n, gpuNeighbourCapacity(), mismatches, too_many, too_few, gpuNeighbourTotal(), ref_total, gpuNeighbourOverflow(), ref_overflow);
    int fails = mismatches > 0 || gpuNeighbourTotal() != ref_total || gpuNeighbourOverflow() != ref_overflow;

    if(check_density){
        gpuDensity();
        gpuSync();
        std::vector<double> rho(n);
        gpuDownloadDensity(rho.data());
        const double PI = 3.14159265358979323846, H = p.H, norm = 1.0/(PI*H*H*H);
        double worst = 0;
        std::vector<double> ref_rho(n, 0.0);
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < n; i++){
            double s = 0;
            for(int j = 0; j < n; j++){
                double dx = pts[j].x - pts[i].x, dy = pts[j].y - pts[i].y, dz = pts[j].z - pts[i].z;
                double r2 = dx*dx + dy*dy + dz*dz;
                if(r2 >= 4*H*H) continue;
                double q = std::sqrt(r2)/H;
                double w = q < 1 ? 1 - 1.5*q*q + 0.75*q*q*q : 0.25*(2-q)*(2-q)*(2-q);
                s += norm*w;
            }
            ref_rho[i] = s;
        }
        for(int i = 0; i < n; i++) worst = std::max(worst, std::fabs(rho[i] - ref_rho[i])/std::max(ref_rho[i], 1e-12));
        printf("[%s] density: worst relative error vs double-precision brute force = %.3e\n", name, worst);
        if(worst > 1e-4) fails = 1;
    }
    gpuFree();
    return fails;
}

int main(){
    int fails = 0;
    fails += run("uniform", uniform_scene(30000, 30.0, 1), true);
    fails += run("overcrowded", uniform_scene(30000, 12.0, 2), false);
    printf(fails ? "FAIL\n" : "PASS\n");
    return fails ? 1 : 0;
}
