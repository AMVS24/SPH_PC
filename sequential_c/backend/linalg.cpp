#include <linalg.h>
#include <constants.h>

// "Close to zero" divisor threshold, scaled to the problem's own length scale
// (the SPH smoothing radius H) rather than a fixed absolute epsilon.
static const double DIV_ZERO_EPS = H * 1e-12;


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
    if(b.x == 0 || b.y==0){
        throw std::runtime_error("Vec2 Division threw divide-by-0 error");
    }
    else if(abs(b.x) <DIV_ZERO_EPS || abs(b.y) < DIV_ZERO_EPS){
        static int warn_count = 0;
        if(warn_count < 5){
            std::cerr << "WARNING: Vec2 Division close to zero"<< '\n';
            if(++warn_count == 5) std::cerr << "  (further Vec2 Division warnings suppressed)\n";
        }
    }
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
    if(b==0){
        throw std::runtime_error("Vec2 const Division threw divide-by-0 error");
    }
    else if(abs(b) <DIV_ZERO_EPS){
        static int warn_count = 0;
        if(warn_count < 5){
            std::cerr << "WARNING: Vec2 const Division close to zero"<< '\n';
            if(++warn_count == 5) std::cerr << "  (further Vec2 const Division warnings suppressed)\n";
        }
    }
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
    if(b.x == 0 || b.y==0 || b.z ==0){
        throw std::runtime_error("Vec3 Division threw divide-by-0 error");
    }
    else if(abs(b.x) <DIV_ZERO_EPS || abs(b.y) < DIV_ZERO_EPS || abs(b.z) < DIV_ZERO_EPS){
        static int warn_count = 0;
        if(warn_count < 5){
            std::cerr << "WARNING: Vec3 Division close to zero"<< '\n';
            if(++warn_count == 5) std::cerr << "  (further Vec3 Division warnings suppressed)\n";
        }
    }
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
        if(b.data[i] == 0 || b.data[i]==0){
            throw std::runtime_error("VecN Division threw divide-by-0 error");
        }
        else if(abs(b.data[i]) <DIV_ZERO_EPS || abs(b.data[i]) < DIV_ZERO_EPS){
            static int warn_count = 0;
            if(warn_count < 5){
                std::cerr << "WARNING: VecN Division close to zero"<< '\n';
                if(++warn_count == 5) std::cerr << "  (further VecN Division warnings suppressed)\n";
            }
        }
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
        if(b == 0 || b==0){
            throw std::runtime_error("VecN const Division threw divide-by-0 error");
        }
        else if(abs(b) <DIV_ZERO_EPS || abs(b) < DIV_ZERO_EPS){
            static int warn_count = 0;
            if(warn_count < 5){
                std::cerr << "WARNING: VecN const Division close to zero"<< '\n';
                if(++warn_count == 5) std::cerr << "  (further VecN const Division warnings suppressed)\n";
            }
        }
        c.data[i] = a.data[i] / b;
    }

    //this could use threads
}
