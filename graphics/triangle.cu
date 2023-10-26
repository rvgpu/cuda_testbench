#include <eigen3/Eigen/Eigen>

#include "config.hpp"
#include "common.hpp"

struct triangle {
    Eigen::Vector4f v[3];
    Eigen::Vector3f color[3];
};

__global__ void fragment_shader(
    struct triangle *in_triangle,
    uint8_t *out_color_buffer
) {
    uint32_t pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pixel_y = blockIdx.y * blockDim.y + threadIdx.y;

    struct triangle t = *in_triangle;

    float v0_x = t.v[0][0];
    float v0_y = t.v[0][1];
    float v1_x = t.v[1][0];
    float v1_y = t.v[1][1];
    float v2_x = t.v[2][0];
    float v2_y = t.v[2][1];

    float center_x = pixel_x + 0.5f;
    float center_y = pixel_y + 0.5f;

    float denominator = (v0_x * v1_y - v0_y * v1_x) + (v1_x * v2_y - v1_y * v2_x) + (v2_x * v0_y - v2_y * v0_x);

    float bary0 = ((v1_y - v2_y) * center_x + (v2_x - v1_x) * center_y + (v1_x * v2_y - v1_y * v2_x)) / denominator;
    float bary1 = ((v2_y - v0_y) * center_x + (v0_x - v2_x) * center_y + (v2_x * v0_y - v2_y * v0_x)) / denominator;
    float bary2 = ((v0_y - v1_y) * center_x + (v1_x - v0_x) * center_y + (v0_x * v1_y - v0_y * v1_x)) / denominator;

    int pixel_in_triangle = (bary0 >= 0) && (bary1 >= 0) && (bary2 >= 0);

    if (pixel_in_triangle) {
        Eigen::Vector3f color = bary0 * t.color[0] + bary1 * t.color[1] + bary2 * t.color[2];

        uint32_t pixel_id = pixel_y * WIDTH + pixel_x;

        out_color_buffer[pixel_id * 4 + 0] = (uint8_t)(color[0] * 255);
        out_color_buffer[pixel_id * 4 + 1] = (uint8_t)(color[1] * 255);
        out_color_buffer[pixel_id * 4 + 2] = (uint8_t)(color[2] * 255);
        out_color_buffer[pixel_id * 4 + 3] = 255;
    } else {
        // Do nothing
    }
}

int main() {
    // 1. Data preparation
    Eigen::Vector4f positions[3] = {
        {-0.5, 0.5, 0.0, 1.0},
        {0.5, 0.5, 0.0, 1.0},
        {0, -0.5, 0.0, 1.0}
    };
    Eigen::Vector3f colors[3] = {
        {0.0, 0.0, 1.0},
        {0.0, 1.0, 0.0},
        {1.0, 0.0, 0.0}
    };

    for (uint32_t i = 0; i < 3; i++) {
        positions[i][0] = positions[i][0] * (WIDTH / 2) + (WIDTH / 2);
        positions[i][1] = positions[i][1] * (HEIGHT / 2) + (HEIGHT / 2);
    }

    struct triangle t;
    for (uint32_t i = 0; i < 3; i++) {
        t.v[i] = positions[i];
        t.color[i] = colors[i];
    }

    // 2. Fragment shader
    struct triangle *fs_in_triangle;
    uint8_t *fs_out_color_buffer;

    cudaMalloc(&fs_in_triangle, sizeof(struct triangle));
    cudaMalloc(&fs_out_color_buffer, WIDTH * HEIGHT * 4);

    cudaMemcpy(fs_in_triangle, &t, sizeof(struct triangle), cudaMemcpyHostToDevice);

    dim3 threads_per_block(32, 32, 1);
    dim3 num_blocks(WIDTH / 32, HEIGHT / 32, 1);

    fragment_shader<<<num_blocks, threads_per_block>>>(fs_in_triangle, fs_out_color_buffer);

    cudaDeviceSynchronize();

    // 3. Write to image
    uint8_t *color_buffer = (uint8_t *) malloc(WIDTH * HEIGHT * 4);

    cudaMemcpy(color_buffer, fs_out_color_buffer, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);

    write_ppm("triangle", WIDTH, HEIGHT, color_buffer);
}