#include <cmath>
#include <stdexcept>
#include <iostream>

typedef struct {double x,y,z;} vec3;
typedef struct {double x,y;} vec2;
typedef struct {
    double* data; 
    int size;

    double& operator[](int i) {
            return data[i];
        }

    const double& operator[](int i) const {
        return data[i];
    }
} vecN;

vec2 vadd(const vec2* a, const vec2* b, vec2* c);
vec2 vsub(const vec2* a, const vec2* b, vec2* c);
vec2 vmul(const vec2* a, const vec2* b, vec2* c);
vec2 vdiv(const vec2* a, const vec2* b, vec2* c);
vec2 vadd_constant(const vec2* a, double b, vec2* c);
vec2 vsub_constant(const vec2* a, double b, vec2* c);
vec2 vmul_constant(const vec2* a, double b, vec2* c);
vec2 vdiv_constant(const vec2* a, double b, vec2* c);
double vnorm(const vec2* a);
double vnorm(const vec2 a);

vec3 vadd(const vec3* a, const vec3* b, vec3* c);
vec3 vsub(const vec3* a, const vec3* b, vec3* c);
vec3 vmul(const vec3* a, const vec3* b, vec3* c);
vec3 vadd_constant(const vec3* a, double b, vec3* c);
vec3 vsub_constant(const vec3* a, double b, vec3* c);
vec3 vmul_constant(const vec3* a, double b, vec3* c);

vecN vadd(const vecN& a, vecN& b, vec2& c);
vecN vsub(const vecN& a, const vecN& b, vec2& c);
vecN vmul(const vecN& a, const vecN& b, vec2& c);
vecN vdiv(const vecN& a, const vecN& b, vec2& c);
vecN vadd_constant(const vecN& a, double b, vec2* c);
vecN vsub_constant(const vecN& a, double b, vec2* c);
vecN vmul_constant(const vecN& a, double b, vec2* c);
vecN vdiv_constant(const vecN& a, double b, vec2* c);

inline vec2 operator+(vec2 a, vec2 b) { vec2 c; vadd(&a, &b, &c); return c; }
inline vec2 operator-(vec2 a, vec2 b) { vec2 c; vsub(&a, &b, &c); return c; }
inline vec2 operator*(vec2 a, vec2 b) { vec2 c; vmul(&a, &b, &c); return c; }
inline vec2 operator/(vec2 a, vec2 b) { vec2 c; vdiv(&a, &b, &c); return c; }
inline vec2 operator+(vec2 a, double b) { vec2 c; vadd_constant(&a, b, &c); return c; }
inline vec2 operator-(vec2 a, double b) { vec2 c; vsub_constant(&a, b, &c); return c; }
inline vec2 operator*(vec2 a, double b) { vec2 c; vmul_constant(&a, b, &c); return c; }
inline vec2 operator/(vec2 a, double b) { vec2 c; vdiv_constant(&a, b, &c); return c; }
inline vec2 operator*(double k, vec2 a) { return a * k; }

inline vec2& operator+=(vec2& a, vec2 b) { vadd(&a, &b, &a); return a; }
inline vec2& operator-=(vec2& a, vec2 b) { vsub(&a, &b, &a); return a; }
inline vec2& operator*=(vec2& a, double k) { vmul_constant(&a, k, &a); return a; }

inline vecN operator+(vecN a, vecN b) { vecN c; vadd(&a, &b, &c); return c; }
inline vecN operator-(vecN a, vecN b) { vecN c; vsub(&a, &b, &c); return c; }
inline vecN operator*(vecN a, vecN b) { vecN c; vmul(&a, &b, &c); return c; }
inline vecN operator/(vecN a, vecN b) { vecN c; vdiv(&a, &b, &c); return c; }
inline vecN operator+(vecN a, double b) { vecN c; vadd_constant(&a, b, &c); return c; }
inline vecN operator-(vecN a, double b) { vecN c; vsub_constant(&a, b, &c); return c; }
inline vecN operator*(vecN a, double b) { vecN c; vmul_constant(&a, b, &c); return c; }
inline vecN operator/(vecN a, double b) { vecN c; vdiv_constant(&a, b, &c); return c; }
//inline vecN operator[](vecN a, int i) {return a->data[i]}