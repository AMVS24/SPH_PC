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
vec2 vdiv(const vec2* a, const vec2* b, vec2* c){
    if(b->x == 0 || b->y==0){
        throw std::runtime_error("Vec2 Division threw divide-by-0 error");
    }
    else if(abs(b->x) <1e-9 || abs(b->y) < 1e-9){
        std::cerr << "WARNING: Vec2 Division close to zero"<< '\n';;
    }
    c->x = a->x / b->x;
    c->y = a->y / b->y;

    // for matrix add and mul this could use threads
}
vec2 vadd_constant(const vec2* a, double b, vec2* c){
    c->x = a->x + b;
    c->y = a->y + b;

    // for matrix add and mul this could use threads
}
vec2 vsub_constant(const vec2* a, double b, vec2* c){
    c->x = a->x - b;
    c->y = a->y - b;

    // for matrix add and mul this could use threads
}
vec2 vmul_constant(const vec2* a, double b, vec2* c){
    c->x = a->x * b;
    c->y = a->y * b;

    // for matrix add and mul this could use threads
}
vec2 vdiv_constant(const vec2* a, double b, vec2* c){
    if(b==0){
        throw std::runtime_error("Vec2 const Division threw divide-by-0 error");
    }
    else if(abs(b) <1e-9){
        std::cerr << "WARNING: Vec2 const Division close to zero"<< '\n';;
    }
    c->x = a->x / b;
    c->y = a->y / b;

    // for matrix add and mul this could use threads
}
double vnorm(const vec2* a){
    double norm;
    norm = sqrt(a->x*a->x + a->y*a->y);
    return norm;
}
double vnorm(const vec2 a){
    double norm;
    norm = sqrt(a.x*a.x + a.y*a.y);
    return norm;
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
vec3 vdiv(const vec3* a, const vec3* b, vec3* c){
    if(b->x == 0 || b->y==0 || b->z ==0){
        throw std::runtime_error("Vec3 Division threw divide-by-0 error");
    }
    else if(abs(b->x) <1e-9 || abs(b->y) < 1e-9 || abs(b->z) < 1e-9){
        std::cerr << "WARNING: Vec3 Division close to zero"<< '\n';;
    }
    c->x = a->x / b->x;
    c->y = a->y / b->y;

    // for matrix add and mul this could use threads
}
vec3 vadd_constant(const vec3* a, double b, vec3* c){
    c->x += b;
    c->y += b;
    c->z += b;

    // for matrix add and mul this could use threads
}
vec3 vsub_constant(const vec3* a, double b, vec3* c){
    c->x -= b;
    c->y -= b;
    c->z -= b;

    // for matrix add and mul this could use threads
}
vec3 vmul_constant(const vec3* a, double b, vec3* c){
    c->x *= b;
    c->y *= b;
    c->z *= b;

    // for matrix add and mul this could use threads
}


/*Two ways of using this to make matrices/tensors, one is to say 
1)vecN (*matrix{pointer, n})[m] = malloc and just handle what happens when you do matrix1*matrix2 seperately
2)smthing else???
*/
vecN vadd(const vecN& a, vecN& b, vecN& c){
    for(int i = 0; i<a->size; i++){
        c->data[i] = a->data[i] + b->data[i];
    }
    //this could use threads
}
vecN vsub(const vecN& a, const vecN& b, vecN& c){
    for(int i = 0; i<a->size; i++){
        c->data[i] = a->data[i] - b->data[i];    }
    //this could use threads
}
vecN vmul(const vecN& a, const vecN& b, vecN& c){
    for(int i = 0; i<a->size; i++){
        c->data[i] = a->data[i] * b->data[i];
    }
    //this could use threads
}
vecN vdiv(const vecN& a, const vecN& b, vecN& c){
    for(int i = 0; i<a->size; i++){
        if(b->data[i] == 0 || b->data[i]==0){
            throw std::runtime_error("VecN Division threw divide-by-0 error");
        }
        else if(abs(b->data[i]) <1e-9 || abs(b->data[i]) < 1e-9){
            std::cerr << "WARNING: VecN Division close to zero"<< '\n';
        }
        c->data[i] = a->data[i] / b->data[i];
    }
    // for matrix add and mul this could use threads
}
vecN vadd_constant(const vecN& a, double b, vecN& c){
    for(int i = 0; i<a->size; i++){
        a->data[i] += b;
    }

    //this could use threads
}
vecN vsub_constant(const vecN& a, double b, vecN& c){
    for(int i = 0; i<a->size; i++){
        a->data[i] -= b;
    }

    //this could use threads
}
vecN vmul_constant(const vecN& a, double b, vecN& c){
    for(int i = 0; i<a->size; i++){
        a->data[i] *= b;
    }

    //this could use threads
}
vecN vdiv_constant(const vecN& a, double b, vecN& c){
    for(int i = 0; i<a->size; i++){
        if(b == 0 || b==0){
            throw std::runtime_error("VecN const Division threw divide-by-0 error");
        }
        else if(abs(b) <1e-9 || abs(b) < 1e-9){
            std::cerr << "WARNING: VecN const Division close to zero"<< '\n';
        }
        a->data[i] /= b;
    }

    //this could use threads
}
