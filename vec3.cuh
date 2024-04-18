#pragma once

#include <cmath>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using std::sqrt; // I guess is the fatest we can go

// I´m gonna at least document all of this since I didn't wrote anything, might as well understand it
class vec3 {
public:
    double e[3]; // Vector (the same as in the geometry) (x, y, z)

    __host__ __device__
    vec3() : e{0,0,0} {} // Constructor, initializes the vector with (0,0,0)

    /**
     * @brief Construct a new vec3 object
     * 
     * @param e0 Value of X
     * @param e1 Value of Y
     * @param e2 Value of Z
     */
    __host__ __device__
    vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

    /**
     * @brief Get the X value of the vector
     * 
     * @return double 
     */
    __host__ __device__
    double x() const { return e[0]; }

    /**
     * @brief Get the Y value of the vector
     * 
     * @return double 
     */
    __host__ __device__
    double y() const { return e[1]; }

    /**
     * @brief Get the Z value of the vector
     * 
     * @return double 
     */
    __host__ __device__
    double z() const { return e[2]; }

    __host__ __device__ inline float r() const { return e[0]; }
    __host__ __device__ inline float g() const { return e[1]; }
    __host__ __device__ inline float b() const { return e[2]; }

    __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ double operator[](int i) const { return e[i]; }
    __host__ __device__ double& operator[](int i) { return e[i]; }

    /**
     * @brief CPU Addition
     * 
     * @param v Vector to add
     * @return vec3 += v
     */
    __host__ __device__
    vec3& operator+=(const vec3& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    /**
     * @brief CPU Multiplication
     * 
     * @param t Vector to multiply
     * @return vec3 *= t
     */
    __host__ __device__
    vec3& operator*=(double t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    /**
     * @brief CPU Division
     * 
     * @param t Vector to divide
     * @return vec3 /= t 
     */
    __host__ __device__
    vec3& operator/=(double t) {
        return *this *= 1/t;
    }

    /**
     * @brief Vector lenght (a.k.a Vector Module)
     * 
     * @return double vec3 lenght
     */
    __host__ __device__
    double length() const {
        return sqrt(length_squared());
    }

    /**
     * @brief Sum of squares from the vec3 coordinates
     * 
     * @return double 
     */
    __host__ __device__
    double length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }
};

// point3 is just an alias for vec3, but useful for geometric clarity in the code.
using point3 = vec3;


// Vector Utility Functions

// Overload for use with the C OUT
inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

// Very slow CPU functions that the book originally provided
__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(double t, const vec3& v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v, double t) {
    return t * v;
}

__host__ __device__ inline vec3 operator/(const vec3& v, double t) {
    return (1/t) * v;
}

__host__ __device__ inline double dot(const vec3& u, const vec3& v) {
    return u.e[0] * v.e[0]
         + u.e[1] * v.e[1]
         + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(const vec3& v) {
    return v / v.length();
}