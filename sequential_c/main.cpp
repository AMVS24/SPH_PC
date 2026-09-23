#include <physics.h>
#include <random>
#include <vector>
#include "../Rendering/Renderer.h"
#include "profiler/Profiler.h"
#define avg(a,b) (a+b)/2


// #define pos_i position_array[i]
// #define pos_j position_array[j]
// #define density_i density_array[i]
// #define density_j density_array[j]
// #define pressure_i pressure_array[i]
// #define pressure_j pressure_array[j]


using namespace std;

void spawn_particles(int n, int seed, vec2 bb_min, vec2 bb_max, vec2*& position_array, vec2*& velocity_array,vec2*& acceleration_array){
    position_array = (vec2*)malloc(sizeof(vec2)*n);
    velocity_array = (vec2*)calloc(sizeof(vec2),n);
    acceleration_array = (vec2*)calloc(sizeof(vec2),n);

    uniform_real_distribution<double> x_pos(bb_min.x, bb_max.x);
    uniform_real_distribution<double> y_pos(bb_min.y, bb_max.y);

    // #pragma omp parallel for
    for(int i = 0;i < n;i++){
        // this can be parallelised with threads
        mt19937 gen(seed+i); // If we use a single seeded rng stream then eventually with multiple threads we won't get rhe same particles

        position_array[i].x = x_pos(gen);
        position_array[i].y = y_pos(gen);

        velocity_array[i].x = 0;
        velocity_array[i].y = 0;
    }
}


void world_to_ndc(const vec2* world, vec2* ndc, int n, vec2 world_min, vec2 world_max){
    const double sx = 2.0/(world_max.x - world_min.x);
    const double sy = 2.0/(world_max.y - world_min.y);

    for(int i = 0; i < n; i++){
        ndc[i].x = (world[i].x - world_min.x)*sx - 1.0;
        ndc[i].y = (world[i].y - world_min.y)*sy - 1.0;
    }
}

int main(int argc, char** argv){
    int n = atoi(argv[1]);

    vec2 WORLD_MIN{WORLD_MIN_X, WORLD_MIN_Y};
    vec2 WORLD_MAX{WORLD_MAX_X, WORLD_MAX_Y};

    Renderer renderer(1000,1000, "SPH");
    renderer.setParticleColor(0.0f, 0.0f, 1.0f);

    const double dt = 0.001;
    const int substeps_per_frame = 20;

    const vec2 world_center{(WORLD_MIN_X+WORLD_MAX_X)*0.5, (WORLD_MIN_Y+WORLD_MAX_Y)*0.5};

    vec2 *position_array, *velocity_array, *acceleration_array;
    vecN density_array(n);
    vecN pressure_array(n);
    {
        ScopedTimer t(g_profiler, "Initialisation");
        spawn_particles(n, 42, world_center - SPAWN_HALF_EXTENT, world_center + SPAWN_HALF_EXTENT, position_array, velocity_array, acceleration_array);

        for(int i=0; i<n;i++){
            velocity_array[i] += acceleration_array[i]*0.5*dt; // base case for first frame for leapfrog integration, first kick in ->(kick)-drift-kick-drift-kick.....
        }
    }

    vector<vec2> ndc_positions(n);
    const double particle_scale_xy = PARTICLE_SIZE_TO_H*H*PARTICLE_NDC_PER_WORLD_UNIT;
    const vec2 particle_scale{particle_scale_xy, particle_scale_xy};

    const int PROFILER_REPORT_EVERY_FRAMES = 60;
    int frame_count = 0;

    while(!renderer.shouldClose()){
        for(int step = 0; step < substeps_per_frame; step++){
            integrate(position_array, velocity_array, acceleration_array, density_array, pressure_array, n, n, dt); // Where all the physics happens
        }

        world_to_ndc(position_array, ndc_positions.data(), n, WORLD_MIN, WORLD_MAX);
        renderer.renderParticles(ndc_positions.data(), n, particle_scale);

        frame_count++;
        if(frame_count % PROFILER_REPORT_EVERY_FRAMES == 0){
            g_profiler.report();
        }
    }

    g_profiler.report();
    g_profiler.write_csv("../sequential_c/profiler/results.csv"); // cwd is Rendering/ -- see README/Makefile run target

}