#ifndef GLOBAL_CUH
#define GLOBAL_CUH

#include <eigen3/Eigen/Eigen>

constexpr double MY_PI = 3.1415926;
constexpr float MY_FLOAT_INFINITY = std::numeric_limits<float>::infinity();
__device__ float MY_EPSILON = 0.00001;

__host__ __device__ __inline__ float clamp(float lo, float hi, float v) {
    return fmaxf(lo, fminf(hi, v));
}

#endif // GLOBAL_CUH