#pragma once

#include <cmath>
#include <stdexcept>
#include <iostream>

typedef struct {double x,y,z;} vec3;
typedef struct {double x,y;} vec2;
typedef struct vecN {
    double* data;
    int size;

    double& operator[](int i) {
            return data[i];
        }

    const double& operator[](int i) const {
        return data[i];
    }

    // For assigning values from temp rvalue objects like a = b+c -> a = R1
    vecN operator=(vecN&& other) noexcept { // lhs is hardcoded to be called "this" and other is a convention for the rhs
        delete[] this->data; //Free outdated memory
        this->data = other.data;
        other.data = nullptr;
        return *this;
    }
    // For copyings from perm lvalue objects like a = b
    vecN operator=(vecN& other){ 
        if(this == &other) return *this; // if a=a is passed then delete[] a.data will happen
        // threading
        for(int i = 0;i<this->size; i++){
            this->data[i] = other.data[i]; // copy data
        }
        return *this;
    }

    vecN(const int n){
        this->data = (double*)malloc(sizeof(double)*n);
        this->size = n;
    }

    vecN(const vecN& other)                    // copy constructor
        :data(new double[other.size]), size(other.size) {
        for (int i = 0; i < size; i++) data[i] = other.data[i];
    }

    vecN(vecN&& other) noexcept                // move constructor
        : data(other.data), size(other.size) {
            other.data = nullptr;
            other.size = 0;
    }

    ~vecN() { delete[] data; }  // destructor runs whenever the scope the vecN was created in ends
} vecN;


void vadd(const vec2& a, const vec2& b, vec2& c);
void vsub(const vec2& a, const vec2& b, vec2& c);
void vmul(const vec2& a, const vec2& b, vec2& c);
void vdiv(const vec2& a, const vec2& b, vec2& c);
void vadd_constant(const vec2& a, double b, vec2& c);
void vsub_constant(const vec2& a, double b, vec2& c);
void vmul_constant(const vec2& a, double b, vec2& c);
void vdiv_constant(const vec2& a, double b, vec2& c);
double vnorm(const vec2& a);
double dot(const vec2& a, const vec2& b);

vec3 vadd(const vec3* a, const vec3* b, vec3* c);
vec3 vsub(const vec3* a, const vec3* b, vec3* c);
vec3 vmul(const vec3* a, const vec3* b, vec3* c);
vec3 vadd_constant(const vec3* a, double b, vec3* c);
vec3 vsub_constant(const vec3* a, double b, vec3* c);
vec3 vmul_constant(const vec3* a, double b, vec3* c);

void vadd(const vecN& a, const vecN& b, vecN& c);
void vsub(const vecN& a, const vecN& b, vecN& c);
void vmul(const vecN& a, const vecN& b, vecN& c);
void vdiv(const vecN& a, const vecN& b, vecN& c);
void vadd_constant(const vecN& a, double b, vecN& c);
void vsub_constant(const vecN& a, double b, vecN& c);
void vmul_constant(const vecN& a, double b, vecN& c);
void vdiv_constant(const vecN& a, double b, vecN& c);

inline vec2 operator+(const vec2& a, const vec2& b) { vec2 c; vadd(a, b, c); return c; }
inline vec2 operator-(const vec2& a, const vec2& b) { vec2 c; vsub(a, b, c); return c; }
inline vec2 operator*(const vec2& a, const vec2& b) { vec2 c; vmul(a, b, c); return c; }
inline vec2 operator/(const vec2& a, const vec2& b) { vec2 c; vdiv(a, b, c); return c; }
inline vec2 operator+(const vec2& a, double b) { vec2 c; vadd_constant(a, b, c); return c; }
inline vec2 operator-(const vec2& a, double b) { vec2 c; vsub_constant(a, b, c); return c; }
inline vec2 operator*(const vec2& a, double b) { vec2 c; vmul_constant(a, b, c); return c; }
inline vec2 operator/(const vec2& a, double b) { vec2 c; vdiv_constant(a, b, c); return c; }
inline vec2 operator*(double k, vec2 a) { return a * k; }

inline vec2& operator+=(vec2& a, const vec2& b) { vadd(a, b, a); return a; }
inline vec2& operator-=(vec2& a, const vec2& b) { vsub(a, b, a); return a; }
inline vec2& operator*=(vec2& a, double k) { vmul_constant(a, k, a); return a; }

inline vecN operator+(const vecN& a, const vecN& b) { vecN c(a.size); vadd(a, b, c); return c; }
inline vecN operator-(const vecN& a, const vecN& b) { vecN c(a.size); vsub(a, b, c); return c; }
inline vecN operator*(const vecN& a, const vecN& b) { vecN c(a.size); vmul(a, b, c); return c; }
inline vecN operator/(const vecN& a, const vecN& b) { vecN c(a.size); vdiv(a, b, c); return c; }
inline vecN operator+(const vecN& a, double b) { vecN c(a.size); vadd_constant(a, b, c); return c; }
inline vecN operator-(const vecN& a, double b) { vecN c(a.size); vsub_constant(a, b, c); return c; }
inline vecN operator*(const vecN& a, double b) { vecN c(a.size); vmul_constant(a, b, c); return c; }
inline vecN operator/(const vecN& a, double b) { vecN c(a.size); vdiv_constant(a, b, c); return c; }
//inline vecN operator[](vecN a, int i) {return a->data[i]}

