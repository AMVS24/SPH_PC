struct {double x, double y, double z} vec3; 

vec3 vadd(const vec3* a, const vec3* b, vec3* c){
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;

    // for matrix add and mul this could use threads
}
vec3 vsub(const vec3* a, const vec3* b, vec3* c){
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    c.z = a.z - b.z;

    // for matrix add and mul this could use threads
}
vec3 vmul(const vec3* a, const vec3* b, vec3* c){
    c.x = a.x * b.x;
    c.y = a.y * b.y;
    c.z = a.z * b.z;

    // for matrix add and mul this could use threads
}