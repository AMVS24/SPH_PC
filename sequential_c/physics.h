#pragma once

#include <constants.h>
#include <backend/linalg.h>

// When false, suppresses console prints (per-step debug output, profiler reports).
extern bool g_verbose;

double poly_6_kernel(double q);

vec2 spiky_kernel(double q, vec2 dir);

double density_at(vec2 pos_i, vec2* position_array, int n, int j, bool& in_range);

void enforce_boundary(vec2* position_array, vec2* velocity_array, int n);

void integrate(vec2* position_array, vec2* velocity_array, vec2* acceleration_array, vecN& density_array, vecN& pressure_array, int n, int n_fluid, double dt);

void update_forces(vec2* position_array, vec2* velocity_array, vec2* acceleration_array,vecN& density_array, vecN& pressure_array,  int n, int n_fluid);


