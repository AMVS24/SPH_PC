#include <physics.h>
#include <stdio.h>


double poly_6_kernel(double q){
    static const double normalisation_const = 10/(7*PI*H*H);

    if(q < 1){
        double q2 = q*q;
        return normalisation_const * (1 - 1.5 * q2 + 0.75 * q2*q);
    }
    else if(q < 2){
        double q2 = q*q;
        return normalisation_const * (0.25 * (2 - q) * (2 - q) * (2 - q));
    }
    else{
        return 0;
    }
}

vec2 spiky_kernel(double q, vec2 dir){
    static const double normalisation_const = 10/(7*PI*H*H*H);

    if(q<1){
        return -1*dir * normalisation_const * (-3 * q + 2.25 * q * q);
    }
    else if(q<2){
        return -1*dir * normalisation_const * (-3 * (2 - q) * (2-q) / 4);
    }
    else{
        return dir*0;
    }
}

double density_at(vec2 pos_i, vec2* position_array, int n, int j, bool& in_range){
    double result = 0;
    double q = vnorm(position_array[j] - pos_i)/H;
    in_range = q < 2;
    if(in_range){
        result = PARTICLE_MASS*poly_6_kernel(q);
    }
    return result;
}

void enforce_boundary(vec2* position_array, vec2* velocity_array, int n){
    double center_x = (WORLD_MIN_X+WORLD_MAX_X)*0.5;
    double center_y = (WORLD_MIN_Y+WORLD_MAX_Y)*0.5;
    double min_x = center_x - BOUNDARY_WIDTH*0.5;
    double max_x = center_x + BOUNDARY_WIDTH*0.5;
    double min_y = center_y - BOUNDARY_HEIGHT*0.5;
    double max_y = center_y + BOUNDARY_HEIGHT*0.5;

    for(int i = 0; i<n; i++){
        if(position_array[i].x < min_x){
            position_array[i].x = min_x;
            velocity_array[i].x = -velocity_array[i].x*WALL_RESTITUTION;
        }
        else if(position_array[i].x > max_x){
            position_array[i].x = max_x;
            velocity_array[i].x = -velocity_array[i].x*WALL_RESTITUTION;
        }

        if(position_array[i].y < min_y){
            position_array[i].y = min_y;
            velocity_array[i].y = -velocity_array[i].y*WALL_RESTITUTION;
        }
        else if(position_array[i].y > max_y){
            position_array[i].y = max_y;
            velocity_array[i].y = -velocity_array[i].y*WALL_RESTITUTION;
        }
    }
}

void integrate(vec2* position_array, vec2* velocity_array, vec2* acceleration_array, vecN& density_array, vecN& pressure_array, int n, int n_fluid, double dt){
//threads
    for(int i=0; i<n_fluid; i++){
        position_array[i] += velocity_array[i]*dt;
    }

    enforce_boundary(position_array, velocity_array, n_fluid);

    update_forces(position_array, velocity_array, acceleration_array, density_array, pressure_array, n, n_fluid);

    for(int i=0; i<n_fluid; i++){
        velocity_array[i] += acceleration_array[i]*dt;
    }
}

void update_forces(vec2* position_array, vec2* velocity_array, vec2* acceleration_array,vecN& density_array, vecN& pressure_array,  int n, int n_fluid ){
    for(int i = 0; i<n; i++){
        density_array[i] = 0;
    }

    // Can do multithreading here with reduce() over shared result variable
    for(int i = 0; i<n; i++){
        for(int j = 0; j<n; j++){
            bool in_range;
            density_array[i] += density_at(position_array[i], position_array, n, j, in_range);
        }

    }

    // This can also technically be done with multithreading and for loops but subtraction and multiplication for
    // vecN is ready defined and will be done via multithreading
    pressure_array = (density_array-TARGET_DENSITY)*PRESSURE_MULTIPLIER;

    double c0 = sqrt(PRESSURE_MULTIPLIER);

    for(int i = 0; i<n_fluid; i++){
        acceleration_array[i] = vec2{0,0};
    }

    for(int i = 0; i<n_fluid; i++){
        for(int j = 0; j<n; j++){
            if(i != j && density_array[j] != 0){
                vec2 diff = position_array[j]-position_array[i];
                double norm = vnorm(diff);
                if(norm != 0 && norm < 2*H){
                    vec2 grad = spiky_kernel(norm/H, diff/norm); // shared by pressure + viscosity, computed once

                    acceleration_array[i] -= (pressure_array[i]/(density_array[i]*density_array[i]) + pressure_array[j]/(density_array[j]*density_array[j]))*grad; //m_j = m_i so they cancel out during a =F/m_i

                    // visc force =  -m*sum(X*spikey kernel) where X is viscocity term = constant*{h(vi-vj).(ri-rj)/|ri-rj|^2}/avg_density   { |ri-rj|^2 = q^2*h^2 }
                    double dot_prod = dot(diff, velocity_array[j] - velocity_array[i]);
                    if(dot_prod < 0){
                        double mu = H*dot_prod/(norm*norm + 0.0001*H*H);
                        double avg_density = (density_array[i] + density_array[j])*0.5;
                        double pi = (-VISCOSITY_ALPHA*c0*mu)/(avg_density + 1e-9);
                        acceleration_array[i] -= pi*grad;
                    }
                }
            }
        }
        acceleration_array[i].y -= GRAVITY;

    }
}
