#include <physics.h>
#include <random>

#define PRESSURE_MULTIPLIER 100
#define TARGET_DENISTY 1
#define PARTICLE_MASS 1
#define avg(a,b) (a+b)/2

#define GRAVITY 1

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

double density_at(vec2 pos_i, vec2* position_array, int n, int i){
    double result;
    result = PARTICLE_MASS*poly_6_kernel(position_array[j] - pos_i);

}

int main(int* argc, char** argv){
    int n = atoi(argv[1]);

    vec2 WORLD_MIN{0,0};
    vec2 WORLD_MAX{100,100};

    vec2 *position_array, *velocity_array, *acceleration_array;
    spawn_particles(n, 42, vec2{10,10}, vec2{30,30}, position_array, velocity_array, acceleration_array);

    double* density_pointer = (double*)calloc(sizeof(double), n);
    vecN density_array{density_pointer, n};
    double* pressure_pointer = (double*)calloc(sizeof(double), n);
    vecN pressure_array{density_pointer, n};

    // Can do multithreading here with reduce() over shared result variable
    for(int i = 0; i<n; i++){
        for(int j = 0; j<n; j++){
            if(i != j){
                density_array[i] += density_at(position_array[i], position_array, n, i);
            }
        }

    }

    // This can also technically be done with multithreading and for loops but subtraction and multiplication for 
    // vecN is ready defined and will be done via multithreading
    pressure_array = PRESSURE_MULTIPLIER*(density_array-TARGET_DENISTY);    


    for(int i = 0; i<n; i++){
        for(int j = 0; j<n; j++){
            if(i != j && density_array[j] != 0){
                double norm = vnorm(position_array[j]-position_array[]);
                acceleration_array[i] += ((pressure_array[i]*pressure_array[j])/(2*density_array[j]))*spiky_kernel(); //m_j = m_i so they cancel out during a =F/m_i
                
            }
        }
        acceleration_array[i][1] += GRAVITY;

    }



}