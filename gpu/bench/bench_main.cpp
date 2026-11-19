#include <physics.h>
#include <cudaPhysics.h>
#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

using namespace std;
typedef chrono::steady_clock clk;

static double ms_since(clk::time_point t0){
    return chrono::duration<double, milli>(clk::now() - t0).count();
}

enum Variant { SEQ, PREHASH, HASH, GPU };

struct Sim {
    int n;
    vector<vec3> pos, vel, acc;
    vecN density, pressure;
    vector<int> cid, cc, cs, cp;
    Sim(int n_) : n(n_), pos(n_), vel(n_), acc(n_), density(n_), pressure(n_), cid(n_),
                  cc(grid_w*grid_h*grid_d), cs(grid_w*grid_h*grid_d + 1), cp(n_) {}
};

static double rest_spacing(){
    double lo = 0.3*H, hi = 2.0*H;
    for(int it = 0; it < 60; it++){
        double mid = 0.5*(lo + hi), s = 0;
        for(int i = -4; i <= 4; i++)
            for(int j = -4; j <= 4; j++)
                for(int k = -4; k <= 4; k++)
                    s += PARTICLE_MASS*poly_6_kernel(sqrt((double)(i*i + j*j + k*k))*mid/H);
        if(s > TARGET_DENSITY) lo = mid; else hi = mid;
    }
    return 0.5*(lo + hi);
}

static void make_scene(Sim& s){
    double dx = rest_spacing();
    int side = (int)ceil(cbrt((double)s.n));
    double cx = (WORLD_MIN_X+WORLD_MAX_X)*0.5, cy = (WORLD_MIN_Y+WORLD_MAX_Y)*0.5, cz = (WORLD_MIN_Z+WORLD_MAX_Z)*0.5;
    double half = side*dx*0.5;
    for(int idx = 0; idx < s.n; idx++){
        int i = idx % side, j = (idx / side) % side, k = idx / (side*side);
        s.pos[idx] = vec3{cx - half + (i + 0.5)*dx, cy - half + (j + 0.5)*dx, cz - half + (k + 0.5)*dx};
        s.vel[idx] = vec3{0, 0, 0};
        s.acc[idx] = vec3{0, 0, 0};
    }
}

static void density_allpairs(Sim& s, bool par){
    #pragma omp parallel for schedule(static) if(par)
    for(int i = 0; i < s.n; i++){
        double sum = 0;
        for(int j = 0; j < s.n; j++) sum += density_at(s.pos[i], s.pos.data(), s.n, j);
        s.density[i] = sum;
    }
}

static void pressure_loop(Sim& s, bool par){
    #pragma omp parallel for schedule(static) if(par)
    for(int i = 0; i < s.n; i++) s.pressure[i] = (s.density[i] - TARGET_DENSITY)*PRESSURE_MULTIPLIER;
}

static void force_pair(Sim& s, int i, int j, double c0, vec3& a){
    if(i != j && s.density[j] != 0){
        vec3 diff = s.pos[j] - s.pos[i];
        double norm = vnorm(diff);
        if(norm != 0 && norm < 2*H){
            vec3 grad = spiky_kernel(norm/H, diff/norm);
            a -= (s.pressure[i]/(s.density[i]*s.density[i]) + s.pressure[j]/(s.density[j]*s.density[j]))*grad;
            double dot_prod = dot(diff, s.vel[j] - s.vel[i]);
            if(dot_prod < 0){
                double mu = H*dot_prod/(norm*norm + 0.0001*H*H);
                double avg_density = (s.density[i] + s.density[j])*0.5;
                double pi = (-VISCOSITY_ALPHA*c0*mu)/(avg_density + 1e-9);
                a -= pi*grad;
            }
        }
    }
}

static void forces_allpairs(Sim& s, bool par){
    double c0 = sqrt(PRESSURE_MULTIPLIER);
    #pragma omp parallel for schedule(static) if(par)
    for(int i = 0; i < s.n; i++){
        vec3 a = vec3{0, 0, 0};
        for(int j = 0; j < s.n; j++) force_pair(s, i, j, c0, a);
        a.y -= GRAVITY;
        s.acc[i] = a;
    }
}

static void drift(Sim& s, double dt, bool par){
    #pragma omp parallel for schedule(static) if(par)
    for(int i = 0; i < s.n; i++) s.pos[i] += s.vel[i]*dt;
}

static void kick(Sim& s, double dt, bool par){
    #pragma omp parallel for schedule(static) if(par)
    for(int i = 0; i < s.n; i++) s.vel[i] += s.acc[i]*dt;
}

static void step_cpu(Sim& s, Variant v, double dt){
    bool par = (v != SEQ);
    drift(s, dt, par);
    enforce_boundary(s.pos.data(), s.vel.data(), s.n);
    if(v == HASH){
        update_forces(s.pos.data(), s.vel.data(), s.acc.data(), s.density, s.pressure, s.n, s.n, s.cid.data(), s.cc.data(), s.cs.data(), s.cp.data());
    }
    else{
        density_allpairs(s, par);
        pressure_loop(s, par);
        forces_allpairs(s, par);
    }
    kick(s, dt, par);
}

static void cpu_hash_stages(Sim& s, double dt, double* t){
    int n = s.n;
    int num_cells = grid_w*grid_h*grid_d;
    clk::time_point t0;

    t0 = clk::now();
    drift(s, dt, true);
    enforce_boundary(s.pos.data(), s.vel.data(), n);
    t[0] = ms_since(t0);

    t0 = clk::now();
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < n; i++){
        int cx = (int)((s.pos[i].x - grid_min_x)/(2*H));
        int cy = (int)((s.pos[i].y - grid_min_y)/(2*H));
        int cz = (int)((s.pos[i].z - grid_min_z)/(2*H));
        if(cx < 0) cx = 0; else if(cx >= grid_w) cx = grid_w-1;
        if(cy < 0) cy = 0; else if(cy >= grid_h) cy = grid_h-1;
        if(cz < 0) cz = 0; else if(cz >= grid_d) cz = grid_d-1;
        s.cid[i] = cx + cy*grid_w + cz*grid_w*grid_h;
    }
    for(int c = 0; c < num_cells; c++) s.cc[c] = 0;
    for(int i = 0; i < n; i++) s.cc[s.cid[i]]++;
    s.cs[0] = 0;
    for(int c = 0; c < num_cells; c++) s.cs[c+1] = s.cs[c] + s.cc[c];
    vector<int> cursor(s.cs.begin(), s.cs.begin() + num_cells);
    for(int i = 0; i < n; i++) s.cp[cursor[s.cid[i]]++] = i;
    t[1] = ms_since(t0);

    t0 = clk::now();
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < n; i++){
        int cx = s.cid[i] % grid_w;
        int cy = (s.cid[i] / grid_w) % grid_h;
        int cz = s.cid[i] / (grid_w*grid_h);
        double sum = 0;
        for(int dz = -1; dz <= 1; dz++){
            int nz = cz + dz;
            if(nz < 0 || nz >= grid_d) continue;
            for(int dy = -1; dy <= 1; dy++){
                int ny = cy + dy;
                if(ny < 0 || ny >= grid_h) continue;
                for(int dx = -1; dx <= 1; dx++){
                    int nx = cx + dx;
                    if(nx < 0 || nx >= grid_w) continue;
                    int c = nx + ny*grid_w + nz*grid_w*grid_h;
                    for(int k = s.cs[c]; k < s.cs[c+1]; k++) sum += density_at(s.pos[i], s.pos.data(), n, s.cp[k]);
                }
            }
        }
        s.density[i] = sum;
    }
    t[2] = ms_since(t0);

    t0 = clk::now();
    s.pressure = (s.density - TARGET_DENSITY)*PRESSURE_MULTIPLIER;
    t[3] = ms_since(t0);

    t0 = clk::now();
    double c0 = sqrt(PRESSURE_MULTIPLIER);
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < n; i++){
        int cx = s.cid[i] % grid_w;
        int cy = (s.cid[i] / grid_w) % grid_h;
        int cz = s.cid[i] / (grid_w*grid_h);
        vec3 a = vec3{0, 0, 0};
        for(int dz = -1; dz <= 1; dz++){
            int nz = cz + dz;
            if(nz < 0 || nz >= grid_d) continue;
            for(int dy = -1; dy <= 1; dy++){
                int ny = cy + dy;
                if(ny < 0 || ny >= grid_h) continue;
                for(int dx = -1; dx <= 1; dx++){
                    int nx = cx + dx;
                    if(nx < 0 || nx >= grid_w) continue;
                    int c = nx + ny*grid_w + nz*grid_w*grid_h;
                    for(int k = s.cs[c]; k < s.cs[c+1]; k++) force_pair(s, i, s.cp[k], c0, a);
                }
            }
        }
        a.y -= GRAVITY;
        s.acc[i] = a;
    }
    t[4] = ms_since(t0);

    t0 = clk::now();
    kick(s, dt, true);
    t[5] = ms_since(t0);
}

static void gpu_stages(double dt, double* t){
    clk::time_point t0;
    t0 = clk::now(); gpuDrift(dt); gpuEnforceBoundary(); gpuSync(); t[0] = ms_since(t0);
    t0 = clk::now(); gpuBuildGrid(); gpuSync(); t[1] = ms_since(t0);
    t0 = clk::now(); gpuDensity(); gpuSync(); t[2] = ms_since(t0);
    t0 = clk::now(); gpuPressure(); gpuSync(); t[3] = ms_since(t0);
    t0 = clk::now(); gpuForces(); gpuSync(); t[4] = ms_since(t0);
    t0 = clk::now(); gpuKick(dt); gpuSync(); t[5] = ms_since(t0);
}

static long long g_sink = 0;

static void cpu_grid_build(Sim& s){
    int n = s.n;
    int num_cells = grid_w*grid_h*grid_d;
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < n; i++){
        int cx = (int)((s.pos[i].x - grid_min_x)/(2*H));
        int cy = (int)((s.pos[i].y - grid_min_y)/(2*H));
        int cz = (int)((s.pos[i].z - grid_min_z)/(2*H));
        if(cx < 0) cx = 0; else if(cx >= grid_w) cx = grid_w-1;
        if(cy < 0) cy = 0; else if(cy >= grid_h) cy = grid_h-1;
        if(cz < 0) cz = 0; else if(cz >= grid_d) cz = grid_d-1;
        s.cid[i] = cx + cy*grid_w + cz*grid_w*grid_h;
    }
    for(int c = 0; c < num_cells; c++) s.cc[c] = 0;
    for(int i = 0; i < n; i++) s.cc[s.cid[i]]++;
    s.cs[0] = 0;
    for(int c = 0; c < num_cells; c++) s.cs[c+1] = s.cs[c] + s.cc[c];
    vector<int> cursor(s.cs.begin(), s.cs.begin() + num_cells);
    for(int i = 0; i < n; i++) s.cp[cursor[s.cid[i]]++] = i;
}

static long long cpu_density_search(Sim& s){
    int n = s.n;
    long long total = 0;
    #pragma omp parallel for schedule(static) reduction(+:total)
    for(int i = 0; i < n; i++){
        int cx = s.cid[i] % grid_w;
        int cy = (s.cid[i] / grid_w) % grid_h;
        int cz = s.cid[i] / (grid_w*grid_h);
        for(int dz = -1; dz <= 1; dz++){
            int nz = cz + dz;
            if(nz < 0 || nz >= grid_d) continue;
            for(int dy = -1; dy <= 1; dy++){
                int ny = cy + dy;
                if(ny < 0 || ny >= grid_h) continue;
                for(int dx = -1; dx <= 1; dx++){
                    int nx = cx + dx;
                    if(nx < 0 || nx >= grid_w) continue;
                    int c = nx + ny*grid_w + nz*grid_w*grid_h;
                    for(int k = s.cs[c]; k < s.cs[c+1]; k++){
                        vec3 d = s.pos[s.cp[k]] - s.pos[i];
                        if(d.x*d.x + d.y*d.y + d.z*d.z < 4*H*H) total++;
                    }
                }
            }
        }
    }
    return total;
}

static void cpu_density_full(Sim& s){
    int n = s.n;
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < n; i++){
        int cx = s.cid[i] % grid_w;
        int cy = (s.cid[i] / grid_w) % grid_h;
        int cz = s.cid[i] / (grid_w*grid_h);
        double sum = 0;
        for(int dz = -1; dz <= 1; dz++){
            int nz = cz + dz;
            if(nz < 0 || nz >= grid_d) continue;
            for(int dy = -1; dy <= 1; dy++){
                int ny = cy + dy;
                if(ny < 0 || ny >= grid_h) continue;
                for(int dx = -1; dx <= 1; dx++){
                    int nx = cx + dx;
                    if(nx < 0 || nx >= grid_w) continue;
                    int c = nx + ny*grid_w + nz*grid_w*grid_h;
                    for(int k = s.cs[c]; k < s.cs[c+1]; k++) sum += density_at(s.pos[i], s.pos.data(), n, s.cp[k]);
                }
            }
        }
        s.density[i] = sum;
    }
}

static long long cpu_force_search(Sim& s){
    int n = s.n;
    long long total = 0;
    #pragma omp parallel for schedule(static) reduction(+:total)
    for(int i = 0; i < n; i++){
        int cx = s.cid[i] % grid_w;
        int cy = (s.cid[i] / grid_w) % grid_h;
        int cz = s.cid[i] / (grid_w*grid_h);
        for(int dz = -1; dz <= 1; dz++){
            int nz = cz + dz;
            if(nz < 0 || nz >= grid_d) continue;
            for(int dy = -1; dy <= 1; dy++){
                int ny = cy + dy;
                if(ny < 0 || ny >= grid_h) continue;
                for(int dx = -1; dx <= 1; dx++){
                    int nx = cx + dx;
                    if(nx < 0 || nx >= grid_w) continue;
                    int c = nx + ny*grid_w + nz*grid_w*grid_h;
                    for(int k = s.cs[c]; k < s.cs[c+1]; k++){
                        int j = s.cp[k];
                        if(i == j || s.density[j] == 0) continue;
                        vec3 d = s.pos[j] - s.pos[i];
                        double r2 = d.x*d.x + d.y*d.y + d.z*d.z;
                        if(r2 != 0 && r2 < 4*H*H) total++;
                    }
                }
            }
        }
    }
    return total;
}

static void cpu_force_full(Sim& s){
    int n = s.n;
    double c0 = sqrt(PRESSURE_MULTIPLIER);
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < n; i++){
        int cx = s.cid[i] % grid_w;
        int cy = (s.cid[i] / grid_w) % grid_h;
        int cz = s.cid[i] / (grid_w*grid_h);
        vec3 a = vec3{0, 0, 0};
        for(int dz = -1; dz <= 1; dz++){
            int nz = cz + dz;
            if(nz < 0 || nz >= grid_d) continue;
            for(int dy = -1; dy <= 1; dy++){
                int ny = cy + dy;
                if(ny < 0 || ny >= grid_h) continue;
                for(int dx = -1; dx <= 1; dx++){
                    int nx = cx + dx;
                    if(nx < 0 || nx >= grid_w) continue;
                    int c = nx + ny*grid_w + nz*grid_w*grid_h;
                    for(int k = s.cs[c]; k < s.cs[c+1]; k++) force_pair(s, i, s.cp[k], c0, a);
                }
            }
        }
        a.y -= GRAVITY;
        s.acc[i] = a;
    }
}

static void cpu_hash_stages2(Sim& s, double dt, double* t){
    clk::time_point t0;
    t0 = clk::now(); drift(s, dt, true); enforce_boundary(s.pos.data(), s.vel.data(), s.n); t[0] = ms_since(t0);
    t0 = clk::now(); cpu_grid_build(s); t[1] = ms_since(t0);
    t0 = clk::now(); g_sink += cpu_density_search(s); t[2] = ms_since(t0);
    t0 = clk::now(); cpu_density_full(s); t[3] = ms_since(t0);
    t0 = clk::now(); s.pressure = (s.density - TARGET_DENSITY)*PRESSURE_MULTIPLIER; t[4] = ms_since(t0);
    t0 = clk::now(); g_sink += cpu_force_search(s); t[5] = ms_since(t0);
    t0 = clk::now(); cpu_force_full(s); t[6] = ms_since(t0);
    t0 = clk::now(); kick(s, dt, true); t[7] = ms_since(t0);
}

static void gpu_stages2(double dt, double* t){
    clk::time_point t0;
    t0 = clk::now(); gpuDrift(dt); gpuEnforceBoundary(); gpuSync(); t[0] = ms_since(t0);
    t0 = clk::now(); gpuBuildGrid(); gpuSync(); t[1] = ms_since(t0);
    t0 = clk::now(); gpuDensitySearchOnly(); gpuSync(); t[2] = ms_since(t0);
    t0 = clk::now(); gpuDensity(); gpuSync(); t[3] = ms_since(t0);
    t0 = clk::now(); gpuPressure(); gpuSync(); t[4] = ms_since(t0);
    t0 = clk::now(); gpuForcesSearchOnly(); gpuSync(); t[5] = ms_since(t0);
    t0 = clk::now(); gpuForces(); gpuSync(); t[6] = ms_since(t0);
    t0 = clk::now(); gpuKick(dt); gpuSync(); t[7] = ms_since(t0);
}

static long long allpairs_density_search(Sim& s, bool par){
    long long total = 0;
    #pragma omp parallel for schedule(static) reduction(+:total) if(par)
    for(int i = 0; i < s.n; i++){
        for(int j = 0; j < s.n; j++){
            vec3 d = s.pos[j] - s.pos[i];
            if(d.x*d.x + d.y*d.y + d.z*d.z < 4*H*H) total++;
        }
    }
    return total;
}

static long long allpairs_force_search(Sim& s, bool par){
    long long total = 0;
    #pragma omp parallel for schedule(static) reduction(+:total) if(par)
    for(int i = 0; i < s.n; i++){
        for(int j = 0; j < s.n; j++){
            if(i == j || s.density[j] == 0) continue;
            vec3 d = s.pos[j] - s.pos[i];
            double r2 = d.x*d.x + d.y*d.y + d.z*d.z;
            if(r2 != 0 && r2 < 4*H*H) total++;
        }
    }
    return total;
}

static void cpu_allpairs_stages2(Sim& s, double dt, bool par, double* t){
    clk::time_point t0;
    t[1] = 0;
    t0 = clk::now(); drift(s, dt, par); enforce_boundary(s.pos.data(), s.vel.data(), s.n); t[0] = ms_since(t0);
    t0 = clk::now(); g_sink += allpairs_density_search(s, par); t[2] = ms_since(t0);
    t0 = clk::now(); density_allpairs(s, par); t[3] = ms_since(t0);
    t0 = clk::now(); pressure_loop(s, par); t[4] = ms_since(t0);
    t0 = clk::now(); g_sink += allpairs_force_search(s, par); t[5] = ms_since(t0);
    t0 = clk::now(); forces_allpairs(s, par); t[6] = ms_since(t0);
    t0 = clk::now(); kick(s, dt, par); t[7] = ms_since(t0);
}

static void mean_stdev(const vector<double>& v, double& mean, double& sd){
    mean = 0;
    for(double x : v) mean += x;
    mean /= v.size();
    sd = 0;
    for(double x : v) sd += (x - mean)*(x - mean);
    sd = v.size() > 1 ? sqrt(sd/(v.size() - 1)) : 0;
}

int main(int argc, char** argv){
    string variant_name = "gpu";
    int n = 10000, threads = 1, steps = 100;
    bool breakdown = false, checksum = false, breakdown2 = false, mem = false, transfer = false;
    for(int i = 1; i < argc; i++){
        if(!strcmp(argv[i], "--variant") && i + 1 < argc) variant_name = argv[++i];
        else if(!strcmp(argv[i], "--n") && i + 1 < argc) n = atoi(argv[++i]);
        else if(!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if(!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if(!strcmp(argv[i], "--breakdown")) breakdown = true;
        else if(!strcmp(argv[i], "--checksum")) checksum = true;
        else if(!strcmp(argv[i], "--breakdown2")) breakdown2 = true;
        else if(!strcmp(argv[i], "--mem")) mem = true;
        else if(!strcmp(argv[i], "--transfer")) transfer = true;
    }
    Variant v = variant_name == "sequential" ? SEQ : variant_name == "prehash" ? PREHASH : variant_name == "hash" ? HASH : GPU;
    if(v == SEQ) threads = 1;
    omp_set_num_threads(threads);

    const double dt = 0.001, WINDOW = 10;
    Sim* s = new Sim(n);
    make_scene(*s);

    auto step = [&](){
        if(v == GPU) integrate(s->pos.data(), s->vel.data(), s->acc.data(), s->density, s->pressure, n, n, dt);
        else step_cpu(*s, v, dt);
    };

    clk::time_point first_t0 = clk::now();
    step();
    double first_ms = ms_since(first_t0);
    step();

    if(mem){
        if(v != GPU){ fprintf(stderr, "mem only for gpu\n"); return 1; }
        printf("MEM variant=%s n=%d device_bytes=%zu\n", variant_name.c_str(), n, gpuAllocatedBytes());
        return 0;
    }

    if(transfer){
        if(v != GPU){ fprintf(stderr, "transfer only for gpu\n"); return 1; }
        const int reps = 5;
        double step_ms = 0, fetch_ms = 0;
        for(int r = 0; r < reps; r++){ clk::time_point t0 = clk::now(); step(); step_ms += ms_since(t0); }
        fetch_positions(s->pos.data(), n);
        for(int r = 0; r < reps; r++){ clk::time_point t0 = clk::now(); fetch_positions(s->pos.data(), n); fetch_ms += ms_since(t0); }
        printf("TRANSFER variant=%s n=%d first_step_ms=%.4f step_ms=%.4f fetch_ms=%.4f\n", variant_name.c_str(), n, first_ms, step_ms/reps, fetch_ms/reps);
        return 0;
    }

    if(breakdown2){
        int windows = max(1, steps/(int)WINDOW);
        vector<vector<double>> win(6);
        for(int w = 0; w < windows; w++){
            double acc_d[6] = {0, 0, 0, 0, 0, 0};
            for(int k = 0; k < (int)WINDOW; k++){
                double t[8];
                if(v == GPU) gpu_stages2(dt, t);
                else if(v == HASH) cpu_hash_stages2(*s, dt, t);
                else cpu_allpairs_stages2(*s, dt, v != SEQ, t);
                acc_d[0] += t[1] + t[2];
                acc_d[1] += max(0.0, t[3] - t[2]);
                acc_d[2] += t[5];
                acc_d[3] += max(0.0, t[6] - t[5]);
                acc_d[4] += t[4];
                acc_d[5] += t[0] + t[7];
            }
            for(int q = 0; q < 6; q++) win[q].push_back(acc_d[q]/WINDOW);
        }
        printf("STAGES2 variant=%s n=%d threads=%d", variant_name.c_str(), n, threads);
        const char* names2[6] = {"A", "B", "C", "D", "E", "F"};
        for(int q = 0; q < 6; q++){
            double m, sd;
            mean_stdev(win[q], m, sd);
            printf(" %s_mean=%.6f %s_stdev=%.6f", names2[q], m, names2[q], sd);
        }
        printf("\n");
        return 0;
    }

    if(breakdown){
        if(v != HASH && v != GPU){ fprintf(stderr, "breakdown only for hash and gpu\n"); return 1; }
        int windows = max(1, steps/(int)WINDOW);
        vector<vector<double>> win(6);
        for(int w = 0; w < windows; w++){
            double acc_t[6] = {0, 0, 0, 0, 0, 0};
            for(int k = 0; k < (int)WINDOW; k++){
                double t[6];
                if(v == GPU) gpu_stages(dt, t); else cpu_hash_stages(*s, dt, t);
                for(int q = 0; q < 6; q++) acc_t[q] += t[q];
            }
            for(int q = 0; q < 6; q++) win[q].push_back(acc_t[q]/WINDOW);
        }
        printf("STAGES variant=%s n=%d threads=%d", variant_name.c_str(), n, threads);
        const char* names[6] = {"drift_boundary", "grid_build", "density", "pressure", "forces", "kick"};
        for(int q = 0; q < 6; q++){
            double m, sd;
            mean_stdev(win[q], m, sd);
            printf(" %s_mean=%.6f %s_stdev=%.6f", names[q], m, names[q], sd);
        }
        printf("\n");
        return 0;
    }

    int windows = max(1, steps/(int)WINDOW);
    vector<double> window_ms;
    for(int w = 0; w < windows; w++){
        clk::time_point t0 = clk::now();
        for(int k = 0; k < (int)WINDOW; k++) step();
        window_ms.push_back(ms_since(t0)/WINDOW);
    }
    double m, sd;
    mean_stdev(window_ms, m, sd);
    printf("RESULT variant=%s n=%d threads=%d steps=%d mean_ms=%.6f stdev_ms=%.6f", variant_name.c_str(), n, threads, windows*(int)WINDOW, m, sd);

    if(checksum){
        double dsum = 0, ysum = 0;
        vector<double> rho(n);
        if(v == GPU){
            gpuDownloadDensity(rho.data());
            fetch_positions(s->pos.data(), n);
        }
        else for(int i = 0; i < n; i++) rho[i] = s->density[i];
        for(int i = 0; i < n; i++){ dsum += rho[i]; ysum += s->pos[i].y; }
        printf(" density_sum=%.12f y_sum=%.9f", dsum, ysum);
    }
    printf("\n");
    return 0;
}
