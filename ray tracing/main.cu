#include "Renderer.cuh"

int main() {
    // Create scene
    Scene scene(1280, 960);

    // Create sphere
    Sphere sphere1 = Sphere(Eigen::Vector3f(-1, 0, -12), 2);    
    sphere1.material.material_type = DIFFUSE_AND_SPECULAR;
    sphere1.material.diffuse_color = Eigen::Vector3f(0.6, 0.7, 0.8);

    Sphere sphere2 = Sphere(Eigen::Vector3f(0.5, -0.5, -8), 1.5);
    sphere2.material.ior = 1.5;
    sphere2.material.material_type = REFLECTION_AND_REFRACTION;

    // Add sphere to scene
    scene.add_sphere(sphere1);
    scene.add_sphere(sphere2);

    // Create mesh
    Eigen::Vector3f vertices[4] = {
        {-5,-3,-6},
        {5,-3,-6},
        {5,-3,-16},
        {-5,-3,-16}
    };
    Eigen::Vector2f st_coords[4] = {
        {0, 0}, {1, 0}, {1, 1}, {0, 1}
    };
    int vertex_indices[6] = {
        0, 1, 3,
        1, 2, 3
    };    
    Triangle_Mesh mesh = Triangle_Mesh(2, vertices, st_coords, vertex_indices);

    // Add mesh to scene
    scene.add_triangle_mesh(mesh);

    // Create light
    Light light1 = Light(Eigen::Vector3f(-20, 70, 20), Eigen::Vector3f(0.5, 0.5, 0.5));
    Light light2 = Light(Eigen::Vector3f(30, 50, -12), Eigen::Vector3f(0.5, 0.5, 0.5));

    // Add light to scene
    scene.add_light(light1);
    scene.add_light(light2); 

    // Render scene to image
    Renderer r;
    r.render(scene);

    return 0;
}