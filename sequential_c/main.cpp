#include <physics.h>
#include <random.h>

using namespace std;

void spawn_particles(int n, int seed, vec3 bb_min, vec3 bb_max){
    vec3 (*particle_array)[n] = malloc(sizeof(vec3)*n);

    uniform_real_distribution<double> x_pos(bb_min.x, bb_max.x);
    uniform_real_distribution<double> y_pos(bb_min.y, bb_max.y);
    uniform_real_distribution<double> z_pos(bb_min.z, bb_max.z);

    // #pragma omp parallel for
    for(int i = 0;i < n;i++){
        // this can be parallelised with threads
        mt19937 gen(seed+i); // If we use a single seeded rng stream then eventually with multiple threads we won't get rhe same particles

        particle_array[i].p.x = x_pos(gen);
        particle_array[i].p.y = y_pos(gen);
        particle_array[i].p.z = z_pos(gen);
        
        particle_array[i].v.x = 0;
        particle_array[i].v.y = 0;
        particle_array[i].v.z = 0;
    }
}

int main(int* argc, char** argv){
    int n = atoi(argv[1]);

    vec3 WORLD_MIN{0,0,0};
    vec3 WORLD_MAX{100,100,100};

    spawn_particles(n, 42, vec3{10,10,10}, vec3{30,30,30});
}