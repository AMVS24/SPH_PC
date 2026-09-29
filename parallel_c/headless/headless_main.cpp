#include <physics.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>

using namespace std;

int main(int argc, char** argv){
    int n = argc > 1 ? atoi(argv[1]) : 500;
    int steps = argc > 2 ? atoi(argv[2]) : 1000;
    const char* out_path = argc > 3 ? argv[3] : "positions.bin";
    omp_set_num_threads(argc > 4 ? atoi(argv[4]) : omp_get_max_threads());
    printf("threads = %d\n", omp_get_max_threads());
    const double dt = 0.001;

    vec2* pos = (vec2*)malloc(sizeof(vec2)*n);
    vec2* vel = (vec2*)calloc(n, sizeof(vec2));
    vec2* acc = (vec2*)calloc(n, sizeof(vec2));
    vecN density(n);
    vecN pressure(n);

    const int num_cells = grid_w * grid_h;
    int* cell_id_arr    = (int*)malloc(sizeof(int)*n);
    int* cell_count     = (int*)malloc(sizeof(int)*num_cells);
    int* cell_start     = (int*)malloc(sizeof(int)*(num_cells + 1));
    int* cell_particles = (int*)malloc(sizeof(int)*n);

    vec2 c{(WORLD_MIN_X+WORLD_MAX_X)*0.5, (WORLD_MIN_Y+WORLD_MAX_Y)*0.5};
    uniform_real_distribution<double> xd(c.x - SPAWN_HALF_EXTENT, c.x + SPAWN_HALF_EXTENT);
    uniform_real_distribution<double> yd(c.y - SPAWN_HALF_EXTENT, c.y + SPAWN_HALF_EXTENT);
    for(int i = 0; i < n; i++){
        mt19937 gen(42 + i);
        pos[i].x = xd(gen);
        pos[i].y = yd(gen);
    }

    FILE* f = fopen(out_path, "wb");
    if(!f){
        printf("could not open %s\n", out_path);
        return 1;
    }
    fwrite(&n, sizeof(int), 1, f);

    for(int s = 0; s < steps; s++){
        integrate(pos, vel, acc, density, pressure, n, n, dt, cell_id_arr, cell_count, cell_start, cell_particles);
        fwrite(pos, sizeof(vec2), n, f);
        fflush(f);
        printf("\rstep %d / %d", s + 1, steps);
        fflush(stdout);
    }
    printf("\nwrote %s\n", out_path);

    fclose(f);
    free(pos); free(vel); free(acc);
}
