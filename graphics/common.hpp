#ifndef COMMON_HPP
#define COMMON_HPP

#include <eigen3/Eigen/Eigen>

constexpr double MY_PI = 3.1415926;

Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

    float cos_theta = std::cos(rotation_angle / 180.0 * MY_PI);
    float sin_theta = std::sin(rotation_angle / 180.0 * MY_PI);

    Eigen::Matrix4f rotate_around_z_axis;    
    rotate_around_z_axis <<
        cos_theta, -sin_theta, 0.0, 0.0,
        sin_theta, cos_theta, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0;

    model = rotate_around_z_axis * model;

    return model;
}

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate <<
        1, 0, 0, -eye_pos[0],
        0, 1, 0, -eye_pos[1],
        0, 0, 1, -eye_pos[2],
        0, 0, 0, 1;

    view = translate * view;

    return view;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar)
{

    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    float l, r, b, t;

    // zNear, zFar are positive, and n, f are negative
    float n = -zNear;
    float f = -zFar;

    t = -n * std::tan(eye_fov / 2 / 180.0 * MY_PI);
    b = -t;
    r = aspect_ratio * t;
    l = -r;

    float p11 = 2 * n / (r - l);
    float p13 = -(r + l) / (r - l);
    float p22 = 2 * n / (t - b);
    float p23 = -(t + b) / (t - b);
    float p33 = (n + f) / (n - f);
    float p34 = -2 * n * f / (n - f);

    Eigen::Matrix4f perspective_proj;
    perspective_proj <<
        p11, 0.0, p13, 0.0,
        0.0, p22, p23, 0.0,
        0.0, 0.0, p33, p34,
        0.0, 0.0, 1.0, 0.0;

    projection = perspective_proj * projection;

    return projection;
}

Eigen::Matrix4f get_model_matrix_hw3(float angle)
{
    Eigen::Matrix4f rotation;
    angle = angle * MY_PI / 180.f;
    rotation << cos(angle), 0, sin(angle), 0,
                0, 1, 0, 0,
                -sin(angle), 0, cos(angle), 0,
                0, 0, 0, 1;

    Eigen::Matrix4f scale;
    scale << 2.5, 0, 0, 0,
              0, 2.5, 0, 0,
              0, 0, 2.5, 0,
              0, 0, 0, 1;

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;

    return translate * rotation * scale;
}

void write_ppm(const char *fname, uint32_t w, uint32_t h, uint8_t *fb) {
    std::string fpath;
    fpath += fname;
    fpath += ".ppm";
    FILE *wf = fopen(fpath.c_str(), "w");
    fprintf(wf, "P3\n");
    fprintf(wf, "%u %u\n", w, h);
    fprintf(wf, "255\n");
    for (uint32_t i = 0; i < h; i++) {
        for (uint32_t j = 0; j < w; j++) {
            uint32_t index = i * w + j;
            uint8_t r = fb[index * 4 + 0];
            uint8_t g = fb[index * 4 + 1];
            uint8_t b = fb[index * 4 + 2];
            fprintf(wf, "%d %d %d\n", r, g, b);
        }
    }

    fclose(wf);
}

#endif