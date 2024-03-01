#include "Global.cuh"
#include "Renderer.cuh"

__device__ Eigen::Vector3f reflect(Eigen::Vector3f I, Eigen::Vector3f N) {
    // Incident light is incoming, and reflected light is outgoing
    return I - 2 * I.dot(N) * N;
}

__device__ Eigen::Vector3f refract(Eigen::Vector3f I, Eigen::Vector3f N, float ior) {
    float cos_i = clamp(-1, 1, I.dot(N));

    float eta_i = 1;
    float eta_t = ior;
    Eigen::Vector3f normal = N;

    // Go into media
    if (cos_i < 0) {
        cos_i = -cos_i;
    }
    // Go out of media
    else {
        // Swap eta_i, eta_t
        float tmp;
        tmp = eta_i; eta_i = eta_t; eta_t = tmp;

        // Reverse normal
        normal = -N;
    }    

    // Compute eta, (cos_t)^2
    float eta = eta_i / eta_t;
    float k = 1 - eta * eta * (1 - cos_i * cos_i);

    // Total reflection
    if (k < 0) {
        return Eigen::Vector3f(0, 0, 0);
    }
    // Partial reflection
    else {
        float cos_t = sqrtf(k);

        // Snell's law
        return eta * I + (eta * cos_i - cos_t) * normal;
    }
}

__device__ float fresnel(Eigen::Vector3f I, Eigen::Vector3f N, float ior) {
    float cos_i = clamp(-1, 1, I.dot(N));

    float eta_i = 1;
    float eta_t = ior;

    // Go into media
    if (cos_i < 0) {
        cos_i = -cos_i;
    }
    // Go out of media
    else {
        // Swap eta_i, eta_t
        float tmp;
        tmp = eta_i; eta_i = eta_t; eta_t = tmp;
    }

    // Compute eta, (cos_t)^2
    float eta = eta_i / eta_t;
    float k = 1 - eta * eta * (1 - cos_i * cos_i);

    // Total reflection
    if (k < 0) {
        return 1;
    }
    // Partial reflection
    else {
        float cos_t = sqrtf(k);

        // Fresnel reflectance formula
        float Rs = ((eta_t * cos_i) - (eta_i * cos_t)) / ((eta_t * cos_i) + (eta_i * cos_t));
        float Rp = ((eta_i * cos_i) - (eta_t * cos_t)) / ((eta_i * cos_i) + (eta_t * cos_t));

        return (Rs * Rs + Rp * Rp) / 2;
    }
}

__device__ bool hit_objects(Ray *ray, Device_Scene *scene, Intersect *hit, Material *material) {
    // Get objects from scene
    int num_spheres = scene->num_spheres;
    Device_Sphere *spheres = scene->device_spheres;
    int num_meshes = scene->num_meshes;
    Device_Triangle_Mesh *meshes = scene->device_meshes;

    // Compute the nearest hit to objects
    float t_sphere = MY_FLOAT_INFINITY;
    float t_mesh = MY_FLOAT_INFINITY;
    Device_Sphere *hit_sphere = nullptr;
    Device_Triangle_Mesh *hit_mesh = nullptr;

    // Iterate over spheres in scene
    for (int i = 0; i < num_spheres; i++) {
        Device_Sphere sphere = spheres[i];

        if (sphere.intersect(ray, hit) && hit->t < t_sphere) {
            // Update the nearest hit
            t_sphere = hit->t;            
            hit_sphere = &spheres[i];
        }
    }

    // Pass to next iteration
    t_mesh = t_sphere;

    // Iterate over meshes in scene
    for (int i = 0; i < num_meshes; i++) {
        Device_Triangle_Mesh mesh = meshes[i];

        if (mesh.intersect(ray, hit) && hit->t < t_mesh) {
            // Update the nearest hit
            t_mesh = hit->t;
            hit_mesh = &meshes[i];
        }
    }

    // If not intersect, return false
    if (t_mesh == MY_FLOAT_INFINITY) {
        return false;
    }

    // If intersect, get material information of object
    if (t_mesh < t_sphere) {
        // Object is triangle mesh
        material->Kd = hit_mesh->material.Kd;
        material->Ks = hit_mesh->material.Ks;
        material->ior = hit_mesh->material.ior;
        material->exponent = hit_mesh->material.exponent;
        material->diffuse_color = hit_mesh->get_diffuse_color(hit);
        material->material_type = hit_mesh->material.material_type;
    } else {
        // Object is sphere
        material->Kd = hit_sphere->material.Kd;
        material->Ks = hit_sphere->material.Ks;
        material->ior = hit_sphere->material.ior;
        material->exponent = hit_sphere->material.exponent;
        material->diffuse_color = hit_sphere->get_diffuse_color(hit);
        material->material_type = hit_sphere->material.material_type;
    }

    return true;
}

__device__ Eigen::Vector3f diffuse_and_specular(Ray *ray, Device_Scene *scene, Intersect *hit, Material *material) {
    // Get lights from scene
    int num_lights = scene->num_lights;
    Device_Light *lights = scene->device_lights;

    // Get hit point and hit normal
    Eigen::Vector3f point = hit->point;
    Eigen::Vector3f normal = hit->normal;

    // Get material information of object
    float Kd = material->Kd;
    float Ks = material->Ks;
    float exponent = material->exponent;
    Eigen::Vector3f color = material->diffuse_color;

    Eigen::Vector3f diffuse(0, 0, 0);
    Eigen::Vector3f specular(0, 0, 0);

    // Iterate over lights in scene
    for (int i = 0; i < num_lights; i++) {
        Device_Light light = lights[i];

        // Compute light direction and reflect direction
        Eigen::Vector3f light_dir = (light.position - point).normalized();
        Eigen::Vector3f reflect_dir = reflect(-light_dir, normal).normalized();

        // Compute cosine
        float cos_light_normal = light_dir.dot(normal);
        float cos_reflect_ray = reflect_dir.dot(-ray->direction);

        Eigen::Vector3f shadow_point;
        
        // Perturb shadow point a little bit
        if (cos_light_normal > 0) {
            shadow_point = point + MY_EPSILON * normal;
        } else {
            shadow_point = point - MY_EPSILON * normal;
        }
        
        Ray shadow_ray(shadow_point, light_dir);
        Intersect tmp_hit;
        Material tmp_material;

        // Occlusion test
        if (hit_objects(&shadow_ray, scene, &tmp_hit, &tmp_material)) {
            // If occluded, skip to next iteration
            if (tmp_hit.t * tmp_hit.t < (light.position - point).dot(light.position - point)) {
                continue;
            }
        }        

        // Compute diffuse color and specular color
        for (int j = 0; j < 3; j++) {
            if (cos_light_normal > 0) {
                diffuse[j] += Kd * color[j] * light.intensity[j] * cos_light_normal;
            }

            if (cos_reflect_ray > 0) {
                specular[j] += Ks * light.intensity[j] * powf(cos_reflect_ray, exponent);
            }
        }        
    }
    
    // Return diffuse color plus specular color
    return diffuse + specular;
}



// Use stack to traverse binary tree (reflect ray and refract ray)
template <typename T, int max_size>
class Stack {
public:
    __device__ Stack() {
        index = -1;
    }

    __device__ int size() {
        return index + 1;
    }

    __device__ T * top() {
        return &stack[index];
    }

    __device__ void push(T value) {
        if (index < max_size - 1) {
            index += 1;
            stack[index] = value;
        }
    }

    __device__ void pop() {
        if (index > -1) {
            index -= 1;
        }
    }

private:
    T stack[max_size];
    int index;
};

// Node of binary tree
class Node {
public:
    Ray ray;
    Intersect hit;
    Material material;
    Eigen::Vector3f color;
    
    // Reflect and refract
    bool is_reflect;
    Eigen::Vector3f reflect_color;
    Eigen::Vector3f refract_color;
    bool reflect_visited = false;
    bool refract_visited = false;
};

// The ray-tracing function
__device__ Eigen::Vector3f cast_ray(Ray *ray, Device_Scene *scene) {
    // Initialize hit color by background color
    Eigen::Vector3f hit_color = scene->background_color;

    // Initialize stack
    Stack<Node, 100> S;

    // Push root node
    Node root;
    root.ray = *ray;
    S.push(root);

    // Bounce times
    int bounce_times = 0;
    int max_bounce_times = 5;

    Node node;

    while (true) {
        // Early return
        if (bounce_times >= max_bounce_times) {
            // top() and pop()
            node = *(S.top());
            S.pop();
            bounce_times -= 1;

            bool is_reflect = node.is_reflect;

            // Update node
            Node *tmp = S.top();
            if (is_reflect) {
                tmp->reflect_visited = true;
                tmp->reflect_color = Eigen::Vector3f(0, 0, 0);
            } else {
                tmp->refract_visited = true;
                tmp->refract_color = Eigen::Vector3f(0, 0, 0);
            }

            continue;
        }

        // Only top(). No pop()
        node = *(S.top());

        // If reflect not visited, do hit test
        if (!node.reflect_visited) {
            // Hit test
            bool is_hit = hit_objects(&(node.ray), scene, &(node.hit), &(node.material));

            // Update node
            Node *tmp = S.top();
            tmp->hit = node.hit;
            tmp->material = node.material;            

            // Case 1: Not hit
            if (!is_hit) {
                node.color = scene->background_color;
            }
            // Case 2: Hit diffuse and specular object
            else if (node.material.material_type == DIFFUSE_AND_SPECULAR) {
                node.color = diffuse_and_specular(&(node.ray), scene, &(node.hit), &(node.material));
            }
            // Case 3: Hit reflect and refract object. Push reflect node
            else {
                // Get hit point and hit normal
                Eigen::Vector3f point = node.hit.point;
                Eigen::Vector3f normal = node.hit.normal;
                
                // Compute reflect direcction
                Eigen::Vector3f reflect_dir = reflect(node.ray.direction, normal).normalized();
                float cos_reflect_normal = reflect_dir.dot(normal);
                
                Eigen::Vector3f reflect_point;
                
                // Perturb reflect point a little bit
                if (cos_reflect_normal > 0) {
                    reflect_point = point + MY_EPSILON * normal;
                } else {
                    reflect_point = point - MY_EPSILON * normal;
                }

                Ray reflect_ray(reflect_point, reflect_dir);

                // Push reflect node
                Node reflect_node;
                reflect_node.ray = reflect_ray;
                reflect_node.is_reflect = true;

                S.push(reflect_node);
                bounce_times += 1;
                continue;
            }
        }
        // If reflect visited but refract not visited, push refract node
        else if (!node.refract_visited) {
            // Get hit point and hit normal
            Eigen::Vector3f point = node.hit.point;
            Eigen::Vector3f normal = node.hit.normal;
            
            // Compute refract direcction
            Eigen::Vector3f refract_dir = refract(node.ray.direction, normal, node.material.ior).normalized();
            float cos_refract_normal = refract_dir.dot(normal);
            
            Eigen::Vector3f refract_point;
            
            // Perturb refract point a little bit
            if (cos_refract_normal > 0) {
                refract_point = point + MY_EPSILON * normal;
            } else {
                refract_point = point - MY_EPSILON * normal;
            }

            Ray refract_ray(refract_point, refract_dir);

            // Push refract node
            Node refract_node;
            refract_node.ray = refract_ray;
            refract_node.is_reflect = false;

            S.push(refract_node);
            bounce_times += 1;
            continue;
        }
        // If both visited, compute color by fresnel equation
        else {
            float factor = fresnel(node.ray.direction, node.hit.normal, node.material.ior);            
            node.color = factor * node.reflect_color + (1 - factor) * node.refract_color;
        }

        // Node color computed
        Eigen::Vector3f color = node.color;

        if (S.size() <= 1) {
            hit_color = color;
            break;
        } else {
            // top() and pop()
            node = *(S.top());
            S.pop();
            bounce_times -= 1;

            bool is_reflect = node.is_reflect;

            // Update node
            Node *tmp = S.top();
            if (is_reflect) {
                tmp->reflect_visited = true;
                tmp->reflect_color = color;
            } else {
                tmp->refract_visited = true;
                tmp->refract_color = color;
            }
            
            continue;
        }
    }

    return hit_color;
}



// The kernel
__global__ void render_image_kernel(
    Device_Scene *scene,
    Eigen::Vector3f *framebuffer
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int width = scene->width;
    int height = scene->height;

    if (i >= width || j >= height) {
        return;
    }

    // Scene information
    float scale = scene->scale;
    float aspect_ratio = scene->aspect_ratio;
    Eigen::Vector3f eye_position = scene->eye_position;

    // Ray direction
    float x = ((2 * (i + 0.5) / width) - 1) * scale * aspect_ratio;
    float y = (1 - (2 * (j + 0.5) / height)) * scale;    
    
    Eigen::Vector3f dir = Eigen::Vector3f(x, y, -1).normalized();
    
    // Cast ray to scene, and write color to framebuffer
    Ray ray(eye_position, dir);
    framebuffer[j * width + i] = cast_ray(&ray, scene);
}

void Renderer::render(Scene scene) {
    // Allocate host framebuffer
    Eigen::Vector3f *host_framebuffer = (Eigen::Vector3f *) malloc(scene.width * scene.height * sizeof(Eigen::Vector3f));

    // Allocate device framebuffer
    Eigen::Vector3f *device_framebuffer;
    cudaMalloc(&device_framebuffer, scene.width * scene.height * sizeof(Eigen::Vector3f));

    // Allocate host scene
    Device_Scene *host_scene = new Device_Scene(scene);

    // Allocate device scene
    Device_Scene *device_scene;
    cudaMalloc(&device_scene, sizeof(Device_Scene));
    cudaMemcpy(device_scene, host_scene, sizeof(Device_Scene), cudaMemcpyHostToDevice);

    // thread_N * thread_N <= 1024. Resources of GPU will be exhausted if thread_N is large
    int thread_N = 15;
    dim3 threads_per_block(thread_N, thread_N, 1);
    dim3 num_blocks((scene.width - 1) / thread_N + 1, (scene.height - 1) / thread_N + 1, 1);
    
    // Call kernel function
    render_image_kernel<<<num_blocks, threads_per_block>>>(device_scene, device_framebuffer);

    // Copy device framebuffer to host framebuffer
    cudaMemcpy(host_framebuffer, device_framebuffer, scene.width * scene.height * sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost);

    // Write host framebuffer to image
    FILE* fp = fopen("games101_hw5.ppm", "wb");
    fprintf(fp, "P6\n%d %d\n255\n", scene.width, scene.height);

    for (int i = 0; i < scene.height * scene.width; ++i) {
        uint8_t color[3];
        color[0] = (uint8_t)(255 * clamp(0, 1, host_framebuffer[i][0]));
        color[1] = (uint8_t)(255 * clamp(0, 1, host_framebuffer[i][1]));
        color[2] = (uint8_t)(255 * clamp(0, 1, host_framebuffer[i][2]));
        fwrite(color, 1, 3, fp);
    }

    fclose(fp);    
}