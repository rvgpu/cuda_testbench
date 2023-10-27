#include <eigen3/Eigen/Eigen>

#include "config.hpp"
#include "common.hpp"

__global__ void fragment_shader(
    Eigen::Vector2f *in_positions,
    uint8_t *out_color_buffer
) {
    Eigen::Vector2f p0 = in_positions[0];
    Eigen::Vector2f p1 = in_positions[1];
    Eigen::Vector2f p2 = in_positions[2];
    Eigen::Vector2f p3 = in_positions[3];

    float t = threadIdx.x * 0.001;
    float s = 1 - t;

    Eigen::Vector2f point = pow(s, 3) * p0 + 3 * t * pow(s, 2) * p1 + 3 * pow(t, 2) * s * p2 + pow(t, 3) * p3;

    uint32_t pixel_x = (uint32_t)point[0];
    uint32_t pixel_y = (uint32_t)point[1];

    uint32_t pixel_id = pixel_y * WIDTH + pixel_x;

    out_color_buffer[pixel_id * 4 + 0] = 255;
    out_color_buffer[pixel_id * 4 + 1] = 255;
    out_color_buffer[pixel_id * 4 + 3] = 255;
}

int main() {
    // 1. Data preparation
    Eigen::Vector2f positions[4] = {
        {154.0f, 323.0f},
        {320.0f, 200.0f},
        {541.0f, 331.0f},
        {524.0f, 526.0f}
    };

    // 2. Fragment shader
    Eigen::Vector2f *fs_in_positions;
    uint8_t *fs_out_color_buffer;

    cudaMalloc(&fs_in_positions, 4 * sizeof(Eigen::Vector2f));
    cudaMalloc(&fs_out_color_buffer, WIDTH * HEIGHT * 4);

    cudaMemcpy(fs_in_positions, positions, 4 * sizeof(Eigen::Vector2f), cudaMemcpyHostToDevice);

    fragment_shader<<<1, 1000>>>(fs_in_positions, fs_out_color_buffer);

    cudaDeviceSynchronize();

    // 3. Write to image
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

    write_ppm("games101_hw4", WIDTH, HEIGHT, image);
}