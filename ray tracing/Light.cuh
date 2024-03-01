#ifndef LIGHT_CUH
#define LIGHT_CUH

#include <eigen3/Eigen/Eigen>

class Light {
public:
    Light(Eigen::Vector3f position, Eigen::Vector3f intensity)
        : position(position)
        , intensity(intensity)
    { }

    Eigen::Vector3f position;
    Eigen::Vector3f intensity;
};

class Device_Light {
public:
    Device_Light(Light light)
        : position(light.position)
        , intensity(light.intensity)
    { }

    Eigen::Vector3f position;
    Eigen::Vector3f intensity;
};

#endif // LIGHT_CUH