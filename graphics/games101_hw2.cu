#include <eigen3/Eigen/Eigen>

#include "config.hpp"
#include "common.hpp"

#define VERTEX_COUNT 6
#define TRIANGLE_COUNT (VERTEX_COUNT / 3)

struct triangle {
    Eigen::Vector4f v[3];
    Eigen::Vector3f color[3];
};

struct box_info {
    uint32_t box_l;
    uint32_t box_b;
    uint32_t box_width;
    uint32_t box_height;
};

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
    struct triangle *in_triangle,
    struct box_info *in_box,
    float *in_depth_buffer,
    uint8_t *out_color_buffer
) {
    uint32_t box_l = in_box->box_l;
    uint32_t box_b = in_box->box_b;
    uint32_t box_width = in_box->box_width;
    uint32_t box_height = in_box->box_height;

    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= box_width || y >= box_height) {
        return;
    }

    uint32_t pixel_x = x + box_l;
    uint32_t pixel_y = y + box_b;

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

    // In-triangle test
    if (pixel_in_triangle) {
        float v0_z = t.v[0][2];
        float v1_z = t.v[1][2];
        float v2_z = t.v[2][2];

        // Since z has gone through perspective projection, we use linear interpolation here
        float z = bary0 * v0_z + bary1 * v1_z + bary2 * v2_z;

        uint32_t pixel_id = pixel_y * WIDTH + pixel_x;

        // Depth test
        // We use the convention 0 > n > f in projection, so nearer objects have larger z
        if (z > in_depth_buffer[pixel_id]) {
            in_depth_buffer[pixel_id] = z;

            Eigen::Vector3f color = bary0 * t.color[0] + bary1 * t.color[1] + bary2 * t.color[2];

            out_color_buffer[pixel_id * 4 + 0] = (uint8_t)color[0];
            out_color_buffer[pixel_id * 4 + 1] = (uint8_t)color[1];
            out_color_buffer[pixel_id * 4 + 2] = (uint8_t)color[2];
            out_color_buffer[pixel_id * 4 + 3] = 255;
        }
    }
}

int main() {
    // 1. Data preparation
    Eigen::Vector3f positions[VERTEX_COUNT] = {
        {2, 0, -2},
        {0, 2, -2},
        {-2, 0, -2},
        {3.5, -1, -5},
        {2.5, 1.5, -5},
        {-1, 0.5, -5}
    };
    Eigen::Vector3f colors[VERTEX_COUNT] = {
        {217.0, 238.0, 185.0},
        {217.0, 238.0, 185.0},
        {217.0, 238.0, 185.0},
        {185.0, 217.0, 238.0},
        {185.0, 217.0, 238.0},
        {185.0, 217.0, 238.0}
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

    cudaMalloc(&vs_in_positions, VERTEX_COUNT * sizeof(Eigen::Vector3f));
    cudaMalloc(&vs_in_model, sizeof(Eigen::Matrix4f));
    cudaMalloc(&vs_in_view, sizeof(Eigen::Matrix4f));
    cudaMalloc(&vs_in_projection, sizeof(Eigen::Matrix4f));
    cudaMalloc(&vs_out_positions, VERTEX_COUNT * sizeof(Eigen::Vector4f));

    cudaMemcpy(vs_in_positions, positions, VERTEX_COUNT * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice);
    cudaMemcpy(vs_in_model, &model, sizeof(Eigen::Matrix4f), cudaMemcpyHostToDevice);
    cudaMemcpy(vs_in_view, &view, sizeof(Eigen::Matrix4f), cudaMemcpyHostToDevice);
    cudaMemcpy(vs_in_projection, &projection, sizeof(Eigen::Matrix4f), cudaMemcpyHostToDevice);

    vertex_shader<<<1, VERTEX_COUNT>>>(vs_in_positions, vs_in_model, vs_in_view, vs_in_projection, vs_out_positions);

    // 3. Fragment shader
    Eigen::Vector4f *fs_positions = (Eigen::Vector4f *) malloc(VERTEX_COUNT * sizeof(Eigen::Vector4f));

    cudaMemcpy(fs_positions, vs_out_positions, VERTEX_COUNT * sizeof(Eigen::Vector4f), cudaMemcpyDeviceToHost);

    struct triangle triangles[TRIANGLE_COUNT];

    for (int i = 0; i < TRIANGLE_COUNT; i++) {
        for (int j = 0; j < 3; j++) {
            triangles[i].v[j] = fs_positions[(i * 3) + j];
            triangles[i].color[j] = colors[(i * 3) + j];
        }
    }

    struct triangle *fs_in_triangle;
    struct box_info *fs_in_box;
    float *fs_in_depth_buffer;
    uint8_t *fs_out_color_buffer;

    cudaMalloc(&fs_in_triangle, sizeof(struct triangle));
    cudaMalloc(&fs_in_box, sizeof(struct box_info));
    cudaMalloc(&fs_in_depth_buffer, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc(&fs_out_color_buffer, WIDTH * HEIGHT * 4);

    float *depth_buffer = (float *) malloc(WIDTH * HEIGHT * sizeof(float));
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        depth_buffer[i] = -1.0f;
    }

    cudaMemcpy(fs_in_depth_buffer, depth_buffer, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);

    // Iterate over triangles
    for (int i = 0; i < TRIANGLE_COUNT; i++) {
        struct triangle t = triangles[i];

        Eigen::Vector3f triangle_x(t.v[0].x(), t.v[1].x(), t.v[2].x());
        Eigen::Vector3f triangle_y(t.v[0].y(), t.v[1].y(), t.v[2].y());

        // Bounding box
        uint32_t box_l = (uint32_t)triangle_x.minCoeff();
        uint32_t box_r = (uint32_t)triangle_x.maxCoeff() + 1;
        uint32_t box_b = (uint32_t)triangle_y.minCoeff();
        uint32_t box_t = (uint32_t)triangle_y.maxCoeff() + 1;

        uint32_t box_width = box_r - box_l;
        uint32_t box_height = box_t - box_b;

        struct box_info box = {box_l, box_b, box_width, box_height};

        cudaMemcpy(fs_in_triangle, &t, sizeof(struct triangle), cudaMemcpyHostToDevice);
        cudaMemcpy(fs_in_box, &box, sizeof(struct box_info), cudaMemcpyHostToDevice);

        dim3 threads_per_block(32, 32, 1);
        dim3 num_blocks((box_width - 1) / 32 + 1, (box_height - 1) / 32 + 1, 1);

        fragment_shader<<<num_blocks, threads_per_block>>>(fs_in_triangle, fs_in_box, fs_in_depth_buffer, fs_out_color_buffer);

        cudaDeviceSynchronize();
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

    write_ppm("games101_hw2", WIDTH, HEIGHT, image);
}