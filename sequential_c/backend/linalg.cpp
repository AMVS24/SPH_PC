#include <linalg.h>

vec2 vadd(const vec2* a, const vec2* b, vec2* c){
    c->x = a->x + b->x;
    c->y = a->y + b->y;

    // for matrix add and mul this could use threads
}
vec2 vsub(const vec2* a, const vec2* b, vec2* c){
    c->x = a->x - b->x;
    c->y = a->y - b->y;

    // for matrix add and mul this could use threads
}
vec2 vmul(const vec2* a, const vec2* b, vec2* c){
    c->x = a->x * b->x;
    c->y = a->y * b->y;

    // for matrix add and mul this could use threads
}
vec2 vadd_constant(const vec2* a, int b, vec2* c){
    c->x += b;
    c->y += b;

    // for matrix add and mul this could use threads
}
vec2 vsub_constant(const vec2* a, int b, vec2* c){
    c->x -= b;
    c->y -= b;

    // for matrix add and mul this could use threads
}
vec2 vmul_constant(const vec2* a, int b, vec2* c){
    c->x *= b;
    c->y *= b;

    // for matrix add and mul this could use threads
}
double vnorm(const vec2* a){
    double norm;
    norm = sqrt(a->x*a->x + a->y*a->y);
}



vec3 vadd(const vec3* a, const vec3* b, vec3* c){
    c->x = a->x + b->x;
    c->y = a->y + b->y;
    c->z = a->z + b->z;

    // for matrix add and mul this could use threads
}
vec3 vsub(const vec3* a, const vec3* b, vec3* c){
    c->x = a->x - b->x;
    c->y = a->y - b->y;
    c->z = a->z - b->z;

    // for matrix add and mul this could use threads
}
vec3 vmul(const vec3* a, const vec3* b, vec3* c){
    c->x = a->x * b->x;
    c->y = a->y * b->y;
    c->z = a->z * b->z;

    // for matrix add and mul this could use threads
}
vec3 vadd_constant(const vec3* a, int b, vec3* c){
    c->x += b;
    c->y += b;
    c->z += b;

    // for matrix add and mul this could use threads
}
vec3 vsub_constant(const vec3* a, int b, vec3* c){
    c->x -= b;
    c->y -= b;
    c->z -= b;

    // for matrix add and mul this could use threads
}
vec3 vmul_constant(const vec3* a, int b, vec3* c){
    c->x *= b;
    c->y *= b;
    c->z *= b;

    // for matrix add and mul this could use threads
}


/*Two ways of using this to make matrices/tensors, one is to say 
1)vecN (*matrix{pointer, n})[m] = malloc and just handle what happens when you do matrix1*matrix2 seperately
2)smthing else???
*/
vecN vadd(const vecN* a, vecN* b, vec2* c){
    for(int i = 0; i<a->size; i++){
        a->data[i] += b->data[i];
    }
    //this could use threads
}
vecN vsub(const vecN* a, const vecN* b, vec2* c){
    for(int i = 0; i<a->size; i++){
        a->data[i] -= b->data[i];
    }
    //this could use threads
}
vecN vmul(const vecN* a, const vecN* b, vec2* c){
    for(int i = 0; i<a->size; i++){
        a->data[i] *= b->data[i];
    }
    //this could use threads
}
vecN vadd_constant(const vecN* a, int b, vec2* c){
    for(int i = 0; i<a->size; i++){
        a->data[i] += b;
    }

    //this could use threads
}
vecN vsub_constant(const vecN* a, int b, vec2* c){
    for(int i = 0; i<a->size; i++){
        a->data[i] -= b;
    }

    //this could use threads
}
vecN vmul_constant(const vecN* a, int b, vec2* c){
    for(int i = 0; i<a->size; i++){
        a->data[i] *= b;
    }

    //this could use threads
}
