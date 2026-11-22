#include <linalg.h>



void vadd(const vec2& a, const vec2& b, vec2& c){
    c.x = a.x + b.x;
    c.y = a.y + b.y;

    // for matrix add and mul this could use threads
}
void vsub(const vec2& a, const vec2& b, vec2& c){
    c.x = a.x - b.x;
    c.y = a.y - b.y;

    // for matrix add and mul this could use threads
}
void vmul(const vec2& a, const vec2& b, vec2& c){
    c.x = a.x * b.x;
    c.y = a.y * b.y;

    // for matrix add and mul this could use threads
}
void vdiv(const vec2& a, const vec2& b, vec2& c){
    c.x = a.x / b.x;
    c.y = a.y / b.y;

    // for matrix add and mul this could use threads
}
void vadd_constant(const vec2& a, double b, vec2& c){
    c.x = a.x + b;
    c.y = a.y + b;

    // for matrix add and mul this could use threads
}
void vsub_constant(const vec2& a, double b, vec2& c){
    c.x = a.x - b;
    c.y = a.y - b;

    // for matrix add and mul this could use threads
}
void vmul_constant(const vec2& a, double b, vec2& c){
    c.x = a.x * b;
    c.y = a.y * b;

    // for matrix add and mul this could use threads
}
void vdiv_constant(const vec2& a, double b, vec2& c){
    c.x = a.x / b;
    c.y = a.y / b;

    // for matrix add and mul this could use threads
}
double vnorm(const vec2& a){
    double norm;
    norm = sqrt(a.x*a.x + a.y*a.y);
    return norm;
}

double dot(const vec2& a, const vec2& b){
    double dot;
    dot = a.x*b.x + a.y*b.y;
    return dot;
}


vec3 vadd(const vec3& a, const vec3& b, vec3& c){
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;

    // for matrix add and mul this could use threads
}
vec3 vsub(const vec3& a, const vec3& b, vec3& c){
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    c.z = a.z - b.z;

    // for matrix add and mul this could use threads
}
vec3 vmul(const vec3& a, const vec3& b, vec3& c){
    c.x = a.x * b.x;
    c.y = a.y * b.y;
    c.z = a.z * b.z;

    // for matrix add and mul this could use threads
}
vec3 vdiv(const vec3& a, const vec3& b, vec3& c){
    c.x = a.x / b.x;
    c.y = a.y / b.y;

    // for matrix add and mul this could use threads
}
vec3 vadd_constant(const vec3& a, double b, vec3& c){
    c.x = a.x + b;
    c.y = a.y + b;
    c.z = a.z + b;

    // for matrix add and mul this could use threads
}
vec3 vsub_constant(const vec3& a, double b, vec3& c){
    c.x = a.x - b;
    c.y = a.y - b;
    c.z = a.z - b;

    // for matrix add and mul this could use threads
}
vec3 vmul_constant(const vec3& a, double b, vec3& c){
    c.x = a.x * b;
    c.y = a.y * b;
    c.z = a.z * b;

    // for matrix add and mul this could use threads
}


/*Two ways of using this to make matrices/tensors, one is to say 
1)vecN (*matrix{pointer, n})[m] = malloc and just handle what happens when you do matrix1*matrix2 seperately
2)smthing else???
*/
void vadd(const vecN& a, const vecN& b, vecN& c){
    for(int i = 0; i<a.size; i++){
        c.data[i] = a.data[i] + b.data[i];
    }
    //this could use threads
}
void vsub(const vecN& a, const vecN& b, vecN& c){
    for(int i = 0; i<a.size; i++){
        c.data[i] = a.data[i] - b.data[i];    }
    //this could use threads
}
void vmul(const vecN& a, const vecN& b, vecN& c){
    for(int i = 0; i<a.size; i++){
        c.data[i] = a.data[i] * b.data[i];
    }
    //this could use threads
}
void vdiv(const vecN& a, const vecN& b, vecN& c){
    for(int i = 0; i<a.size; i++){
        c.data[i] = a.data[i] / b.data[i];
    }
    // for matrix add and mul this could use threads
}
void vadd_constant(const vecN& a, double b, vecN& c){
    for(int i = 0; i<a.size; i++){
        c.data[i] = a.data[i] + b;
    }

    //this could use threads
}
void vsub_constant(const vecN& a, double b, vecN& c){
    for(int i = 0; i<a.size; i++){
        c.data[i] = a.data[i] - b;
    }

    //this could use threads
}
void vmul_constant(const vecN& a, double b, vecN& c){
    for(int i = 0; i<a.size; i++){
        c.data[i] = a.data[i] * b;
    }

    //this could use threads
}
void vdiv_constant(const vecN& a, double b, vecN& c){
    for(int i = 0; i<a.size; i++){
        c.data[i] = a.data[i] / b;
    }

    //this could use threads
}
