#include <cmath>

typedef struct {double x,y,z;} vec3;
typedef struct {double x,y;} vec2;
typedef struct {double* data; int size;} vecN;

vec2 vadd(const vec2* a, const vec2* b, vec2* c);
vec2 vsub(const vec2* a, const vec2* b, vec2* c);
vec2 vmul(const vec2* a, const vec2* b, vec2* c);
vec2 vadd_constant(const vec2* a, int b, vec2* c);
vec2 vsub_constant(const vec2* a, int b, vec2* c);
vec2 vmul_constant(const vec2* a, int b, vec2* c);
double vnorm(const vec2* a);

vec3 vadd(const vec3* a, const vec3* b, vec3* c);
vec3 vsub(const vec3* a, const vec3* b, vec3* c);
vec3 vmul(const vec3* a, const vec3* b, vec3* c);
vec3 vadd_constant(const vec3* a, int b, vec3* c);
vec3 vsub_constant(const vec3* a, int b, vec3* c);
vec3 vmul_constant(const vec3* a, int b, vec3* c);

vecN vadd(const vecN* a, vecN* b, vec2* c);
vecN vsub(const vecN* a, const vecN* b, vec2* c);
vecN vmul(const vecN* a, const vecN* b, vec2* c);
vecN vadd_constant(const vecN* a, int b, vec2* c);
vecN vsub_constant(const vecN* a, int b, vec2* c);
vecN vmul_constant(const vecN* a, int b, vec2* c);

inline vec2 operator+(vec2 a, vec2 b) { vec2 c; vadd(&a, &b, &c); return c; }
inline vec2 operator-(vec2 a, vec2 b) { vec2 c; vsub(&a, &b, &c); return c; }
inline vec2 operator*(vec2 a, vec2 b) { vec2 c; vmul(&a, &b, &c); return c; }

inline vec2 operator+(vec2 a, double k) { vec2 c; vadd_constant(&a, k, &c); return c; }
inline vec2 operator-(vec2 a, double k) { vec2 c; vsub_constant(&a, k, &c); return c; }
inline vec2 operator*(vec2 a, double k) { vec2 c; vmul_constant(&a, k, &c); return c; }
inline vec2 operator*(double k, vec2 a) { return a * k; }

inline vec2& operator+=(vec2& a, vec2 b) { vadd(&a, &b, &a); return a; }
inline vec2& operator-=(vec2& a, vec2 b) { vsub(&a, &b, &a); return a; }
inline vec2& operator*=(vec2& a, double k) { vmul_constant(&a, k, &a); return a; }
