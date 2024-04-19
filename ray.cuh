#pragma once
#include "vec3.cuh"

class ray {
public:
    __device__ ray() {}

    __device__ ray(const point3& origin, const vec3& direction) : orig(origin), dir(direction) {}

    __device__ const point3& origin() const  { return orig; }
    __device__ const vec3& direction() const { return dir; }

    __device__ point3 at(double t) const {
        return orig + t*dir;
    }

    
    __device__ vec3 point_at_parameter(float t) const { return A + t*B; }

    vec3 A;
    vec3 B;

private:
    point3 orig;
    vec3 dir;
};