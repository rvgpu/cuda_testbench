#ifndef SCENE_CUH
#define SCENE_CUH

#include <eigen3/Eigen/Eigen>
#include "Global.cuh"
#include "Object.cuh"
#include "Light.cuh"

class Scene {
public:
    Scene(int width, int height)
        : width(width)
        , height(height)
    { }

    int width = 1280;
    int height = 960;
    float fov = 90;
    Eigen::Vector3f eye_position = Eigen::Vector3f(0, 0, 0);
    Eigen::Vector3f background_color = Eigen::Vector3f(0.235294, 0.67451, 0.843137);

    void add_sphere(Sphere sphere) {
        spheres.push_back(sphere);
    }

    std::vector<Sphere> get_spheres() {
        return spheres;
    }

    void add_triangle_mesh(Triangle_Mesh mesh) {
        meshes.push_back(mesh);
    }

    std::vector<Triangle_Mesh> get_triangle_meshes() {
        return meshes;
    }

    void add_light(Light light) {
        lights.push_back(light);
    }

    std::vector<Light> get_lights() {
        return lights;
    }

private:
    std::vector<Sphere> spheres;
    std::vector<Triangle_Mesh> meshes;
    std::vector<Light> lights;
};

class Device_Scene {
public:
    Device_Scene(Scene scene) {
        this->width = scene.width;
        this->height = scene.height;
        this->eye_position = scene.eye_position;
        this->background_color = scene.background_color;

        // Compute other scene information
        this->scale = std::tan(scene.fov * 0.5 * MY_PI / 180.0);
        this->aspect_ratio = scene.width / (float)scene.height;

        // Get spheres from scene
        std::vector<Sphere> spheres = scene.get_spheres();
        this->num_spheres = spheres.size();

        // Allocate host spheres
        Device_Sphere *host_spheres = (Device_Sphere *) malloc(num_spheres * sizeof(Device_Sphere));
        for (int i = 0; i < num_spheres; i++) {
            host_spheres[i] = Device_Sphere(spheres[i]);
        }

        // Allocate device spheres
        cudaMalloc(&device_spheres, num_spheres * sizeof(Device_Sphere));
        cudaMemcpy(device_spheres, host_spheres, num_spheres * sizeof(Device_Sphere), cudaMemcpyHostToDevice);

        // Get meshes from scene
        std::vector<Triangle_Mesh> meshes = scene.get_triangle_meshes();
        this->num_meshes = meshes.size();

        // Allocate host meshes
        Device_Triangle_Mesh *host_meshes = (Device_Triangle_Mesh *) malloc(num_meshes * sizeof(Device_Triangle_Mesh));
        for (int i = 0; i < num_meshes; i++) {
            host_meshes[i] = Device_Triangle_Mesh(meshes[i]);
        }

        // Allocate device meshes
        cudaMalloc(&device_meshes, num_meshes * sizeof(Device_Triangle_Mesh));
        cudaMemcpy(device_meshes, host_meshes, num_meshes * sizeof(Device_Triangle_Mesh), cudaMemcpyHostToDevice);

        // Get lights from scene
        std::vector<Light> lights = scene.get_lights();
        this->num_lights = lights.size();

        // Allocate host lights
        Device_Light *host_lights = (Device_Light *) malloc(num_lights * sizeof(Device_Light));
        for (int i = 0; i < num_lights; i++) {
            host_lights[i] = Device_Light(lights[i]);
        }

        // Allocate device lights
        cudaMalloc(&device_lights, num_lights * sizeof(Device_Light));
        cudaMemcpy(device_lights, host_lights, num_lights * sizeof(Device_Light), cudaMemcpyHostToDevice);
    }    

    int width;
    int height;
    Eigen::Vector3f eye_position;
    Eigen::Vector3f background_color;
    float scale;
    float aspect_ratio;

    int num_spheres = 0;
    Device_Sphere *device_spheres;
    int num_meshes = 0;
    Device_Triangle_Mesh *device_meshes;
    int num_lights = 0;
    Device_Light *device_lights;
};

#endif // SCENE_CUH