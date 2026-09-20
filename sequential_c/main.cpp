#include <physics.h>
#include <random>
#define avg(a,b) (a+b)/2


// #define pos_i position_array[i]
// #define pos_j position_array[j]
// #define density_i density_array[i]
// #define density_j density_array[j]
// #define pressure_i pressure_array[i]
// #define pressure_j pressure_array[j]


using namespace std;

vec2*  spawn_particles(int n, int seed, vec2 bb_min, vec2 bb_max, vec2* position_array, vec2* velocity_array,vec2* acceleration_array){
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
        
        position_array[i].x = 0;
        position_array[i].y = 0;
    }
}


int main(int argc, char** argv){
    int n = atoi(argv[1]);

    vec2 WORLD_MIN{0,0};
    vec2 WORLD_MAX{1000,1000};

    Renderer renderer(1000,1000, "SPH");
    renderer.setParticleColor(0.0f, 0.0f, 1.0f);

    const double dt = 0.001;

    vec2 *position_array, *velocity_array, *acceleration_array;
    spawn_particles(n, 42, vec2{10,10}, vec2{30,30}, position_array, velocity_array, acceleration_array);

    double* density_pointer = (double*)calloc(sizeof(double), n);
    vecN density_array{density_pointer, n};
    double* pressure_pointer = (double*)calloc(sizeof(double), n);
    vecN pressure_array{density_pointer, n};

    velocity_array[i] += acceleration_array[i]*0.5*dt; // base case for first frame for leapfrog integration, first kick in ->(kick)-drift-kick-drift-kick.....

    while(true){
        integrate(position_array, velocity_array, acceleration_array, n, dt);
    }

}