#include <physics.h>


double poly_6_kernel(double q){
    static const double normalisation_const = 10/(7*pi*H*H);

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

double spiky_kernel(double q, vec2 dir){
    static const double normalisation_const = 10/(7*pi*H*H*H);

    if(q<1){
        return -dir * normalisation_const * ();
    }
    else if(q<2){
        return -1*dir * normalisation_const * ();
    }
    else{
        return dir*
    }
}

// threads
void integrate(vec2* position_array, vec2* velocity_array, vec2* acceleration_array, int n, double dt){
    for(int i=0; i<n; i++){
        vec2 v_half_step = velocity_array[i] + acceleration_array[i]*0.5*dt;
        position_array[i] += v_half_step
    }
}