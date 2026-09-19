#include <math.h>

#include <backend/linalg.h>

struct {
    vec2 p;
    vec2 v;
    vec2 a;
} Particle;

double poly_6_kernel(double q);

double spiky_kernel(double q, vec2 dir);