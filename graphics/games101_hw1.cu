#include <eigen3/Eigen/Eigen>

#include "config.hpp"
#include "common.hpp"

__global__ void vertex_shader(
    Eigen::Vector3f *in_positions,
    Eigen::Matrix4f *in_model,
    Eigen::Matrix4f *in_view,
    Eigen::Matrix4f *in_projection,
    Eigen::Vector4f *out_positions
) {
    uint32_t i = threadIdx.x;

    // MVP
    Eigen::Vector4f pos;
    pos << in_positions[i], 1.0;
    pos = (*in_projection) * (*in_view) * (*in_model) * pos;

    // Homogeneous division
    pos[0] /= pos[3];
    pos[1] /= pos[3];
    pos[2] /= pos[3];

    // Viewport transform
    // (x, y) is in the screen space
    float x = pos[0] * (WIDTH / 2) + (WIDTH / 2);
    float y = pos[1] * (HEIGHT / 2) + (HEIGHT / 2);

    // z is between -1.0 (far) and 1.0 (near)
    float z = pos[2];

    // w is the original z before perspective projection
    float w = pos[3];

    Eigen::Vector4f xyzw;
    xyzw << x, y, z, w;

    out_positions[i] = xyzw;
}

__global__ void fragment_shader(
    float *in_endpoints,
    float *in_attributes,
    uint32_t *in_start,
    int *in_is_x_major,
    uint8_t *out_color_buffer
) {
    uint32_t axis = threadIdx.x + (*in_start);

    float p0 = in_endpoints[0];
    float p1 = in_endpoints[1];
    float a0 = in_attributes[0];
    float a1 = in_attributes[1];

    // Line interpolation of attributes
    float t = (axis - p0) / (p1 - p0);
    uint32_t result = (uint32_t)(a0 + t * (a1 - a0));

    if (*in_is_x_major) {
        // axis is x, result is y
        uint32_t pixel_id = result * WIDTH + axis;
        out_color_buffer[pixel_id * 4 + 0] = 255;
        out_color_buffer[pixel_id * 4 + 1] = 255;
        out_color_buffer[pixel_id * 4 + 2] = 255;
        out_color_buffer[pixel_id * 4 + 3] = 255;
    } else {
        // axis is y, result is x
        uint32_t pixel_id = axis * WIDTH + result;
        out_color_buffer[pixel_id * 4 + 0] = 255;
        out_color_buffer[pixel_id * 4 + 1] = 255;
        out_color_buffer[pixel_id * 4 + 2] = 255;
        out_color_buffer[pixel_id * 4 + 3] = 255;
    }
}

int main() {
    // 1. Data preparation
    Eigen::Vector3f positions[3] = {
        {2, 0, -2},
        {0, 2, -2},
        {-2, 0, -2}
    };
    float angle = 0;
    Eigen::Vector3f eye_pos = {0, 0, 5};
    Eigen::Matrix4f model = get_model_matrix(angle);
    Eigen::Matrix4f view = get_view_matrix(eye_pos);
    Eigen::Matrix4f projection = get_projection_matrix(45, 1, 0.1, 50);

    // 2. Vertex shader
    Eigen::Vector3f *vs_in_positions;
    Eigen::Matrix4f *vs_in_model;
    Eigen::Matrix4f *vs_in_view;
    Eigen::Matrix4f *vs_in_projection;
    Eigen::Vector4f *vs_out_positions;

    cudaMalloc(&vs_in_positions, 3 * sizeof(Eigen::Vector3f));
    cudaMalloc(&vs_in_model, sizeof(Eigen::Matrix4f));
    cudaMalloc(&vs_in_view, sizeof(Eigen::Matrix4f));
    cudaMalloc(&vs_in_projection, sizeof(Eigen::Matrix4f));
    cudaMalloc(&vs_out_positions, 3 * sizeof(Eigen::Vector4f));

    cudaMemcpy(vs_in_positions, positions, 3 * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice);
    cudaMemcpy(vs_in_model, &model, sizeof(Eigen::Matrix4f), cudaMemcpyHostToDevice);
    cudaMemcpy(vs_in_view, &view, sizeof(Eigen::Matrix4f), cudaMemcpyHostToDevice);
    cudaMemcpy(vs_in_projection, &projection, sizeof(Eigen::Matrix4f), cudaMemcpyHostToDevice);

    vertex_shader<<<1, 3>>>(vs_in_positions, vs_in_model, vs_in_view, vs_in_projection, vs_out_positions);

    // 3. Fragment shader
    Eigen::Vector4f *fs_positions = (Eigen::Vector4f *) malloc(3 * sizeof(Eigen::Vector4f));

    cudaMemcpy(fs_positions, vs_out_positions, 3 * sizeof(Eigen::Vector4f), cudaMemcpyDeviceToHost);

    float *fs_in_endpoints;
    float *fs_in_attributes;
    uint32_t *fs_in_start;
    int *fs_in_is_x_major;
    uint8_t *fs_out_color_buffer;

    cudaMalloc(&fs_in_endpoints, 2 * sizeof(float));
    cudaMalloc(&fs_in_attributes, 2 * sizeof(float));
    cudaMalloc(&fs_in_start, sizeof(uint32_t));
    cudaMalloc(&fs_in_is_x_major, sizeof(int));
    cudaMalloc(&fs_out_color_buffer, WIDTH * HEIGHT * 4);

    float line_dx[3];
    float line_dy[3];
    int line_is_x_major[3];
    uint32_t line_start_x[3];
    uint32_t line_end_x[3];
    uint32_t line_start_y[3];
    uint32_t line_end_y[3];

    // Iterate over lines
    for (uint32_t i = 0; i < 3; i++) {
        line_dx[i] = fs_positions[(i + 1) % 3].x() - fs_positions[i % 3].x();
        line_dy[i] = fs_positions[(i + 1) % 3].y() - fs_positions[i % 3].y();

        line_is_x_major[i] = (fabs(line_dx[i]) >= fabs(line_dy[i]));

        if (line_dx[i] >= 0) {
            line_start_x[i] = (uint32_t)fs_positions[i % 3].x();
            line_end_x[i] = (uint32_t)fs_positions[(i + 1) % 3].x();
        } else {
            line_start_x[i] = (uint32_t)fs_positions[(i + 1) % 3].x();
            line_end_x[i] = (uint32_t)fs_positions[i % 3].x();
        }

        if (line_dy[i] >= 0) {
            line_start_y[i] = (uint32_t)fs_positions[i % 3].y();
            line_end_y[i] = (uint32_t)fs_positions[(i + 1) % 3].y();
        } else {
            line_start_y[i] = (uint32_t)fs_positions[(i + 1) % 3].y();
            line_end_y[i] = (uint32_t)fs_positions[i % 3].y();
        }

        // Finding the intersection point of a line and an axis is equivalent to doing line interpolation of attributes
        if (line_is_x_major[i]) {
            // x-major
            float p0 = fs_positions[i % 3].x();
            float p1 = fs_positions[(i + 1) % 3].x();
            float a0 = fs_positions[i % 3].y();
            float a1 = fs_positions[(i + 1) % 3].y();

            float line_endpoints[2] = {p0, p1};
            float line_attributes[2] = {a0, a1};

            cudaMemcpy(fs_in_endpoints, line_endpoints, 2 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(fs_in_attributes, line_attributes, 2 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(fs_in_start, &line_start_x[i], sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(fs_in_is_x_major, &line_is_x_major[i], sizeof(int), cudaMemcpyHostToDevice);

            fragment_shader<<<1, line_end_x[i] - line_start_x[i] + 1>>>(fs_in_endpoints, fs_in_attributes, fs_in_start, fs_in_is_x_major, fs_out_color_buffer);

            cudaDeviceSynchronize();
        } else {
            // y-major
            float p0 = fs_positions[i % 3].y();
            float p1 = fs_positions[(i + 1) % 3].y();
            float a0 = fs_positions[i % 3].x();
            float a1 = fs_positions[(i + 1) % 3].x();

            float line_endpoints[2] = {p0, p1};
            float line_attributes[2] = {a0, a1};

            cudaMemcpy(fs_in_endpoints, line_endpoints, 2 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(fs_in_attributes, line_attributes, 2 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(fs_in_start, &line_start_y[i], sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(fs_in_is_x_major, &line_is_x_major[i], sizeof(int), cudaMemcpyHostToDevice);

            fragment_shader<<<1, line_end_y[i] - line_start_y[i] + 1>>>(fs_in_endpoints, fs_in_attributes, fs_in_start, fs_in_is_x_major, fs_out_color_buffer);

            cudaDeviceSynchronize();
        }
    }

    // 4. Write to image
    uint8_t *color_buffer = (uint8_t *) malloc(WIDTH * HEIGHT * 4);

    cudaMemcpy(color_buffer, fs_out_color_buffer, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);

    uint8_t *image = (uint8_t *) calloc(WIDTH * HEIGHT * 4, sizeof(uint8_t));

    uint32_t img_pid, cb_pid;

    for (uint32_t x = 0; x < WIDTH; x++) {
        for (uint32_t y = 0; y < HEIGHT; y++) {
            img_pid = y * WIDTH + x;
            cb_pid = (HEIGHT - 1 - y) * WIDTH + x;

            image[img_pid * 4 + 0] = color_buffer[cb_pid * 4 + 0];
            image[img_pid * 4 + 1] = color_buffer[cb_pid * 4 + 1];
            image[img_pid * 4 + 2] = color_buffer[cb_pid * 4 + 2];
            image[img_pid * 4 + 3] = color_buffer[cb_pid * 4 + 3];
        }
    }

    write_ppm("games101_hw1", WIDTH, HEIGHT, image);
}