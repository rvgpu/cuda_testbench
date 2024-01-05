#include "OBJ_Loader.h"
#include <eigen3/Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "common.hpp"

struct triangle {
    Eigen::Vector4f v[3];
    Eigen::Vector3f color[3];
    Eigen::Vector2f tex_coords[3];
    Eigen::Vector3f normal[3];
};

struct box_info {
    uint32_t box_l;
    uint32_t box_b;
    uint32_t box_width;
    uint32_t box_height;
};

struct light {
    Eigen::Vector3f position;
    Eigen::Vector3f intensity;
};

__global__ void vertex_shader(
    Eigen::Vector4f *in_positions,
    Eigen::Matrix4f *in_model,
    Eigen::Matrix4f *in_view,
    Eigen::Matrix4f *in_projection,
    uint32_t *in_vertex_num,
    Eigen::Vector4f *out_positions
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= *in_vertex_num) {
        return;
    }

    // MVP
    Eigen::Vector4f pos = in_positions[i];
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

__device__ Eigen::Vector3f phong_shader_color(
    Eigen::Vector3f color,
    Eigen::Vector3f viewspace_normal,
    Eigen::Vector3f viewspace_pos
) {
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    Eigen::Vector3f eye_pos{0, 0, 10};
    float p = 150;

    Eigen::Vector3f result_color = {0, 0, 0};

    // Lights
    Eigen::Vector3f point = viewspace_pos;
    Eigen::Vector3f normal = viewspace_normal;

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};
    struct light lights[2] = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};

    for (auto& light : lights) {
        Eigen::Vector3f La;
        Eigen::Vector3f Ld;
        Eigen::Vector3f Ls;

        Eigen::Vector3f l = (light.position - point).normalized();
        Eigen::Vector3f v = (eye_pos - point).normalized();
        Eigen::Vector3f h = (v + l).normalized();

        // Ambient lights
        for (int i = 0; i < 3; i++) {
            La[i] = ka[i] * amb_light_intensity[i];
        }

        // Diffuse lights
        if (normal.dot(l) > 0.0) {
            for (int i = 0; i < 3; i++) {
                Ld[i] = kd[i] * light.intensity[i] / (light.position - point).dot(light.position - point) * normal.dot(l);
            }
        } else {
            Ld << 0.0, 0.0, 0.0;
        }

        // Specular lights
        if (normal.dot(h) > 0.0) {
            for (int i = 0; i < 3; i++) {
                Ls[i] = ks[i] * light.intensity[i] / (light.position - point).dot(light.position - point) * std::pow(normal.dot(h), p);
            }
        } else {
            Ls << 0.0, 0.0, 0.0;
        }

        result_color += La + Ld + Ls;
    }

    return result_color;
}

__global__ void fragment_shader(
    struct triangle *in_triangle,
    struct box_info *in_box,
    Eigen::Vector4f *in_viewspace_pos,
    float *out_depth_buffer,
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
        if (z > out_depth_buffer[pixel_id]) {
            out_depth_buffer[pixel_id] = z;

            Eigen::Vector3f interpolated_color = bary0 * t.color[0] + bary1 * t.color[1] + bary2 * t.color[2];

            Eigen::Vector3f interpolated_normal = bary0 * t.normal[0] + bary1 * t.normal[1] + bary2 * t.normal[2];
            interpolated_normal = interpolated_normal.normalized();

            Eigen::Vector3f interpolated_pos = bary0 * in_viewspace_pos[0].head<3>() + bary1 * in_viewspace_pos[1].head<3>() + bary2 * in_viewspace_pos[2].head<3>();
            
            Eigen::Vector3f color = phong_shader_color(interpolated_color, interpolated_normal, interpolated_pos);            

            out_color_buffer[pixel_id * 4 + 0] = (uint8_t)(color[0] * 255);
            out_color_buffer[pixel_id * 4 + 1] = (uint8_t)(color[1] * 255);
            out_color_buffer[pixel_id * 4 + 2] = (uint8_t)(color[2] * 255);
            out_color_buffer[pixel_id * 4 + 3] = 255;
        }
    }
}

int main() {
    // 1. Data preparation
    // 1.1 Matrices
    float angle = 140.0;
    Eigen::Vector3f eye_pos = {0,0,10};

    Eigen::Matrix4f model = get_model_matrix_hw3(angle);
    Eigen::Matrix4f view = get_view_matrix(eye_pos);
    Eigen::Matrix4f projection = get_projection_matrix(45, 1, 0.1, 50);

    // 1.2 Triangles
    std::vector<struct triangle *> TriangleList;
    objl::Loader Loader;

    std::string obj_path = "../models/spot/";

    // Load .obj File
    bool loadout = Loader.LoadFile(obj_path + "spot_triangulated_good.obj");

    for(auto mesh : Loader.LoadedMeshes)
    {
        for(unsigned int i = 0; i < mesh.Vertices.size(); i += 3)
        {
            struct triangle *t = (struct triangle *) malloc(sizeof(struct triangle));

            for(int j = 0; j < 3; j++)
            {
                t->v[j] = Eigen::Vector4f(
                    mesh.Vertices[i+j].Position.X,
                    mesh.Vertices[i+j].Position.Y,
                    mesh.Vertices[i+j].Position.Z,
                    1.0);

                t->normal[j] = Eigen::Vector3f(
                    mesh.Vertices[i+j].Normal.X,
                    mesh.Vertices[i+j].Normal.Y,
                    mesh.Vertices[i+j].Normal.Z);

                t->tex_coords[j] = Eigen::Vector2f(
                    mesh.Vertices[i+j].TextureCoordinate.X,
                    mesh.Vertices[i+j].TextureCoordinate.Y);
            }

            TriangleList.push_back(t);
        }
    }

    uint32_t triangle_num = TriangleList.size();
    uint32_t vertex_num = triangle_num * 3;

    Eigen::Vector4f *positions = (Eigen::Vector4f *) malloc(vertex_num * sizeof(Eigen::Vector4f));

    for (uint32_t i = 0; i < triangle_num; i ++) {
        positions[i * 3 + 0] = TriangleList[i]->v[0];
        positions[i * 3 + 1] = TriangleList[i]->v[1];
        positions[i * 3 + 2] = TriangleList[i]->v[2];
    }

    // 2. Vertex shader
    Eigen::Vector4f *vs_in_positions;
    Eigen::Matrix4f *vs_in_model;
    Eigen::Matrix4f *vs_in_view;
    Eigen::Matrix4f *vs_in_projection;
    uint32_t *vs_in_vertex_num;
    Eigen::Vector4f *vs_out_positions;

    cudaMalloc(&vs_in_positions, vertex_num * sizeof(Eigen::Vector4f));
    cudaMalloc(&vs_in_model, sizeof(Eigen::Matrix4f));
    cudaMalloc(&vs_in_view, sizeof(Eigen::Matrix4f));
    cudaMalloc(&vs_in_projection, sizeof(Eigen::Matrix4f));
    cudaMalloc(&vs_in_vertex_num, sizeof(uint32_t));
    cudaMalloc(&vs_out_positions, vertex_num * sizeof(Eigen::Vector4f));

    cudaMemcpy(vs_in_positions, positions, vertex_num * sizeof(Eigen::Vector4f), cudaMemcpyHostToDevice);
    cudaMemcpy(vs_in_model, &model, sizeof(Eigen::Matrix4f), cudaMemcpyHostToDevice);
    cudaMemcpy(vs_in_view, &view, sizeof(Eigen::Matrix4f), cudaMemcpyHostToDevice);
    cudaMemcpy(vs_in_projection, &projection, sizeof(Eigen::Matrix4f), cudaMemcpyHostToDevice);
    cudaMemcpy(vs_in_vertex_num, &vertex_num, sizeof(uint32_t), cudaMemcpyHostToDevice);

    dim3 vs_threads_per_block(1024, 1, 1);
    dim3 vs_num_blocks((vertex_num - 1) / 1024 + 1, 1, 1);

    vertex_shader<<<vs_num_blocks, vs_threads_per_block>>>(vs_in_positions, vs_in_model, vs_in_view, vs_in_projection, vs_in_vertex_num, vs_out_positions);

    cudaDeviceSynchronize();

    // 3. Fragment shader
    Eigen::Vector4f *fs_positions = (Eigen::Vector4f *) malloc(vertex_num * sizeof(Eigen::Vector4f));

    cudaMemcpy(fs_positions, vs_out_positions, vertex_num * sizeof(Eigen::Vector4f), cudaMemcpyDeviceToHost);

    struct triangle *fs_in_triangle;
    struct box_info *fs_in_box;
    Eigen::Vector4f *fs_in_viewspace_pos;
    float *fs_out_depth_buffer;
    uint8_t *fs_out_color_buffer;

    cudaMalloc(&fs_in_triangle, sizeof(struct triangle));
    cudaMalloc(&fs_in_box, sizeof(struct box_info));
    cudaMalloc(&fs_in_viewspace_pos, 3 * sizeof(Eigen::Vector4f));
    cudaMalloc(&fs_out_depth_buffer, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc(&fs_out_color_buffer, WIDTH * HEIGHT * 4);

    float *depth_buffer = (float *) malloc(WIDTH * HEIGHT * sizeof(float));
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        depth_buffer[i] = -1.0f;
    }

    cudaMemcpy(fs_out_depth_buffer, depth_buffer, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);

    Eigen::Matrix4f view_model = view * model;
    Eigen::Matrix4f inv_trans = view_model.inverse().transpose();

    // Iterate over triangles
    for (int i = 0; i < triangle_num; i++) {
        struct triangle t;

        Eigen::Vector4f viewspace_normal[3];
        Eigen::Vector4f viewspace_pos[3];

        for (int j = 0; j < 3; j++) {
            t.v[j] = fs_positions[i * 3 + j];

            t.color[j] = {148.0 / 255.0, 121.0 / 255.0, 92.0 / 255.0};

            viewspace_normal[j] << TriangleList[i]->normal[j], 0.0f;
            viewspace_normal[j] = inv_trans * viewspace_normal[j];
            t.normal[j] = viewspace_normal[j].head<3>();

            viewspace_pos[j] = view_model * TriangleList[i]->v[j];
        }

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
        cudaMemcpy(fs_in_viewspace_pos, viewspace_pos, 3 * sizeof(Eigen::Vector4f), cudaMemcpyHostToDevice);

        dim3 threads_per_block(32, 32, 1);
        dim3 num_blocks((box_width - 1) / 32 + 1, (box_height - 1) / 32 + 1, 1);

        fragment_shader<<<num_blocks, threads_per_block>>>(fs_in_triangle, fs_in_box, fs_in_viewspace_pos, fs_out_depth_buffer, fs_out_color_buffer);

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

    write_ppm("games101_hw3_phong_shader", WIDTH, HEIGHT, image);
}