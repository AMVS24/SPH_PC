#pragma once

#include <constants.h>
#include <backend/linalg.h>

double poly_6_kernel(double q);

vec3 spiky_kernel(double q, vec3 dir);

double density_at(vec3 pos_i, vec3* position_array, int n, int j);

void enforce_boundary(vec3* position_array, vec3* velocity_array, int n);

void fetch_positions(vec3* position_array, int n);

void integrate(vec3* position_array, vec3* velocity_array, vec3* acceleration_array, vecN& density_array, vecN& pressure_array, int n, int n_fluid, double dt, int* cell_id = nullptr, int* cell_count = nullptr, int* cell_start = nullptr, int* cell_particles = nullptr);

void update_forces(vec3* position_array, vec3* velocity_array, vec3* acceleration_array,vecN& density_array, vecN& pressure_array,  int n, int n_fluid, int* cell_id = nullptr, int* cell_count = nullptr, int* cell_start = nullptr, int* cell_particles = nullptr);

inline int cell_id(vec3 p){
    int cell_id = (int)p.x/H + ((int)p.y/H)*grid_w + ((int)p.z/H)*grid_w*grid_h; return cell_id;
}
