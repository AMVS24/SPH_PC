#include <physics.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <vector>
#include <chrono>

using namespace std;
typedef chrono::steady_clock clk;

static double ms(clk::time_point a, clk::time_point b){
    return chrono::duration<double, milli>(b - a).count();
}

struct Nbr { int j; double norm; vec2 diff; };

int main(int argc, char** argv){
    int n = argc > 1 ? atoi(argv[1]) : 500;
    omp_set_num_threads(argc > 2 ? atoi(argv[2]) : omp_get_max_threads());
    const double dt = 0.001;

    vec2* pos = (vec2*)malloc(sizeof(vec2)*n);
    vec2* vel = (vec2*)calloc(n, sizeof(vec2));
    vec2* acc = (vec2*)calloc(n, sizeof(vec2));
    vecN density(n);
    vecN pressure(n);

    vec2 c{(WORLD_MIN_X+WORLD_MAX_X)*0.5, (WORLD_MIN_Y+WORLD_MAX_Y)*0.5};
    uniform_real_distribution<double> xd(c.x - SPAWN_HALF_EXTENT, c.x + SPAWN_HALF_EXTENT);
    uniform_real_distribution<double> yd(c.y - SPAWN_HALF_EXTENT, c.y + SPAWN_HALF_EXTENT);
    for(int i = 0; i < n; i++){
        mt19937 gen(42 + i);
        pos[i].x = xd(gen);
        pos[i].y = yd(gen);
    }

    vector<vector<Nbr>> nb(n);
    double c0 = sqrt(PRESSURE_MULTIPLIER);
    double tA = 0, tB = 0, tC = 0, tD = 0, tE = 0, tF = 0, tG = 0;

    for(int s = 0; ; s++){
        auto g0 = clk::now();

        #pragma omp parallel for schedule(static)
        for(int i = 0; i < n; i++) pos[i] += vel[i]*dt;

        auto t0 = clk::now();
        enforce_boundary(pos, vel, n);
        auto t1 = clk::now();
        tF = ms(t0, t1);

        t0 = clk::now();
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < n; i++){
            nb[i].clear();
            for(int j = 0; j < n; j++){
                vec2 diff = pos[j] - pos[i];
                double norm = vnorm(diff);
                if(norm < 2*H) nb[i].push_back({j, norm, diff});
            }
        }
        t1 = clk::now();
        tA = ms(t0, t1);

        t0 = clk::now();
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < n; i++){
            density[i] = 0;
            for(const Nbr& k : nb[i]) density[i] += PARTICLE_MASS*poly_6_kernel(k.norm/H);
        }
        t1 = clk::now();
        tB = ms(t0, t1);

        t0 = clk::now();
        pressure = (density - TARGET_DENSITY)*PRESSURE_MULTIPLIER;
        t1 = clk::now();
        tE = ms(t0, t1);

        t0 = clk::now();
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < n; i++){
            nb[i].clear();
            for(int j = 0; j < n; j++){
                if(i != j && density[j] != 0){
                    vec2 diff = pos[j] - pos[i];
                    double norm = vnorm(diff);
                    if(norm != 0 && norm < 2*H) nb[i].push_back({j, norm, diff});
                }
            }
        }
        t1 = clk::now();
        tC = ms(t0, t1);

        t0 = clk::now();
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < n; i++){
            acc[i] = vec2{0, 0};
            for(const Nbr& k : nb[i]){
                int j = k.j;
                vec2 grad = spiky_kernel(k.norm/H, k.diff/k.norm);
                acc[i] -= (pressure[i]/(density[i]*density[i]) + pressure[j]/(density[j]*density[j]))*grad;
                double dp = dot(k.diff, vel[j] - vel[i]);
                if(dp < 0){
                    double mu = H*dp/(k.norm*k.norm + 0.0001*H*H);
                    double avg_d = (density[i] + density[j])*0.5;
                    double pi = (-VISCOSITY_ALPHA*c0*mu)/(avg_d + 1e-9);
                    acc[i] -= pi*grad;
                }
            }
            acc[i].y -= GRAVITY;
        }
        t1 = clk::now();
        tD = ms(t0, t1);

        #pragma omp parallel for schedule(static)
        for(int i = 0; i < n; i++) vel[i] += acc[i]*dt;

        tG = ms(g0, clk::now());

        if((s + 1) % 3 == 0){
            printf("Step %d, n = %d, threads = %d, time for this timestep (ms):\n", s + 1, n, omp_get_max_threads());
            printf("A) Density NBHD search : %10.4f\n", tA);
            printf("B) Density computation : %10.4f\n", tB);
            printf("C) Force NBHD search   : %10.4f\n", tC);
            printf("D) Force computation   : %10.4f\n", tD);
            printf("E) Pressure computation: %10.4f\n", tE);
            printf("F) Boundary condition  : %10.4f\n", tF);
            printf("G) Total timestep      : %10.4f\n\n", tG);
        }
    }

    free(pos); free(vel); free(acc);
}
