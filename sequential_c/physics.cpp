#include <physics.h>


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

double density_at(vec2 pos_i, vec2* position_array, int n, int j){
    double result;
    result = PARTICLE_MASS*poly_6_kernel(vnorm(position_array[j] - pos_i)/H);
    return result;
}

void integrate(vec2* position_array, vec2* velocity_array, vec2* acceleration_array, vecN& density_array, vecN& pressure_array, int n, double dt){
//threads
    for(int i=0; i<n; i++){
        position_array[i] += v_half_step;
    }

    update_forces(position_array, velocity_array, acceleration_array, vecN* density_array, vecN* pressure_array, n);

    for(int i=0; i<n; i++){
        velocity_array[i] += acceleration_array[i]*0.5*dt;
        position_array[i] += v_half_step;
    }


}

void update_forces(vec2* position_array, vec2* velocity_array, vec2* acceleration_array,vecN& density_array, vecN& pressure_array,  int n ){
        // Can do multithreading here with reduce() over shared result variable
    for(int i = 0; i<n; i++){
        for(int j = 0; j<n; j++){
            if(i != j){
                density_array->[i] += density_at(position_array[i], position_array, n, j);
            }
        }

    }

    // This can also technically be done with multithreading and for loops but subtraction and multiplication for 
    // vecN is ready defined and will be done via multithreading
    pressure_array = (density_array-TARGET_DENISTY)*PRESSURE_MULTIPLIER;    


    for(int i = 0; i<n; i++){
        for(int j = 0; j<n; j++){
            if(i != j && density_array[j] != 0){
                double norm = vnorm(position_array[j]-position_array[i]);
                acceleration_array[i] += ((pressure_array[i]*pressure_array[j])/(2*density_array[j]))*spiky_kernel(norm/H, (position_array[j]-position_array[i])/norm); //m_j = m_i so they cancel out during a =F/m_i
                
            }
        }
        acceleration_array[i].y += GRAVITY;

    }

}