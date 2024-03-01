#ifndef OBJECT_CUH
#define OBJECT_CUH

#include <eigen3/Eigen/Eigen>
#include "Global.cuh"

// Class of ray
class Ray {
public:
    __device__ Ray() { }
    __device__ Ray(Eigen::Vector3f origin, Eigen::Vector3f direction)
        : origin(origin)
        , direction(direction)
    { }

    Eigen::Vector3f origin;
    Eigen::Vector3f direction;
};

// Class of ray-object intersection
class Intersect {
public:
    __device__ Intersect()
        : happened(false)
        , t(MY_FLOAT_INFINITY)
    { }

    bool happened;
    float t;
    Eigen::Vector3f point;
    Eigen::Vector3f normal;
    int index;
    Eigen::Vector2f uv;
};

// Class of material
enum Material_Type {
    DIFFUSE_AND_SPECULAR = 0,
    REFLECTION_AND_REFRACTION = 1
};

class Material {
public:
    __host__ __device__ Material()
        : Kd(0.8)
        , Ks(0.2)
        , ior(1.3)
        , exponent(25)
        , diffuse_color(Eigen::Vector3f(0.2, 0.2, 0.2))
        , material_type(DIFFUSE_AND_SPECULAR)
    { }

    float Kd;
    float Ks;
    float ior;
    float exponent;
    Eigen::Vector3f diffuse_color;
    Material_Type material_type;
};



/* Class of object, which has the following subclasses
** 1. Sphere
** 2. Triangle_Mesh
*/
class Object {
public:
    Object() { }

    Material material;
};

class Sphere : public Object {
public:
    Sphere(Eigen::Vector3f center, float radius)
        : center(center)
        , radius(radius)
    { }

    Eigen::Vector3f center;
    float radius;
};

class Triangle_Mesh : public Object {
public:
    Triangle_Mesh(
        int num_triangles,
        Eigen::Vector3f *vertices,
        Eigen::Vector2f *st_coords,
        int *vertex_indices
    ) {
        this->num_triangles = num_triangles;

        // Get maximal vertex index
        this->max_index = 0;
        for (int i = 0; i < num_triangles * 3; ++i) {
            if (vertex_indices[i] > this->max_index) {
                this->max_index = vertex_indices[i];
            }
        }
        this->max_index += 1;

        // Allocate vertices, st_coords. From 0 to max_index - 1
        this->vertices = (Eigen::Vector3f *) malloc(max_index * sizeof(Eigen::Vector3f));
        this->st_coords = (Eigen::Vector2f *) malloc(max_index * sizeof(Eigen::Vector2f));

        // Allocate vertex_indices. From 0 to 3 * num_triangles - 1
        this->vertex_indices = (int *) malloc(3 * num_triangles * sizeof(int));

        // Initialize vertices, st_coords. From 0 to max_index - 1
        for (int i = 0; i < max_index; i++) {
            this->vertices[i] = vertices[i];
            this->st_coords[i] = st_coords[i];
        }

        // Initialize vertex_indices. From 0 to 3 * num_triangles - 1
        for (int i = 0; i < 3 * num_triangles; i++) {
            this->vertex_indices[i] = vertex_indices[i];
        }
    }

    int num_triangles;
    int max_index;
    Eigen::Vector3f *vertices;
    Eigen::Vector2f *st_coords;
    int *vertex_indices;
};



// Class of sphere on device
class Device_Sphere {
public:
    Device_Sphere(Sphere sphere) {
        // Initialize material information of object
        this->material = sphere.material;

        // Initialize information of sphere
        this->center = sphere.center;
        this->radius = sphere.radius;
    }

    __device__ bool intersect(Ray *ray, Intersect *hit) {
        Eigen::Vector3f orig = ray->origin;
        Eigen::Vector3f dir = ray->direction;

        hit->happened = false;

        // Compute a, b, c of quadratic equation
        Eigen::Vector3f L = orig - this->center;
        float a = dir.dot(dir);
        float b = 2 * dir.dot(L);
        float c = L.dot(L) - (radius * radius);

        // Test ray-sphere intersection
        float tNear, t0, t1;        
        if (!solve_quadratic(a, b, c, &t0, &t1)) {
            return false;
        }

        if (t1 < 0) {
            return false;
        } else if (t0 < 0) {
            tNear = t1;
        } else {
            tNear = t0;
        }

        // Update intersection
        hit->happened = true;
        hit->t = tNear;
        hit->point = orig + tNear * dir;

        // Get normal by hit->point
        hit->normal = get_normal(hit);
        
        return true;
    }

    __device__ Eigen::Vector3f get_diffuse_color(Intersect *hit) {
        return material.diffuse_color;
    }

    __device__ Eigen::Vector3f get_normal(Intersect *hit) {
        return (hit->point - center).normalized();
    }

    __device__ bool solve_quadratic(float a, float b, float c, float *t0, float *t1) {
        float discr = b * b - 4 * a * c;

        float x0, x1;

        // Solve equation
        if (discr < 0) {
            return false;
        } else if (discr == 0) {
            x0 = -0.5 * b / a;
            x1 = -0.5 * b / a;
        } else {
            float q = (b > 0) ?
                -0.5 * (b + sqrt(discr)) :
                -0.5 * (b - sqrt(discr));
            x0 = q / a;
            x1 = c / q;
        }

        // Swap roots
        if (x0 > x1) {
            float tmp;
            tmp = x0; x0 = x1; x1 = tmp;
        }

        // Return roots
        *t0 = x0;
        *t1 = x1;
        return true;
    }

    // Material information of object
    Material material;
    
    // Information of sphere
    Eigen::Vector3f center;
    float radius;
};

// Class of triangle mesh on device
class Device_Triangle_Mesh {
public:
    Device_Triangle_Mesh(Triangle_Mesh mesh) {
        // Initialize material information of object
        this->material = mesh.material;

        // Initialize information of triangle mesh
        this->num_triangles = mesh.num_triangles;
        this->max_index = mesh.max_index;

        // Allocate device vertices. From 0 to max_index - 1
        cudaMalloc(&vertices, max_index * sizeof(Eigen::Vector3f));
        cudaMemcpy(vertices, mesh.vertices, max_index * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice);

        // Allocate device st_coords. From 0 to max_index - 1
        cudaMalloc(&st_coords, max_index * sizeof(Eigen::Vector2f));
        cudaMemcpy(st_coords, mesh.st_coords, max_index * sizeof(Eigen::Vector2f), cudaMemcpyHostToDevice);

        // Allocate device vertex_indices. From 0 to 3 * num_triangles - 1
        cudaMalloc(&vertex_indices, 3 * num_triangles * sizeof(int));
        cudaMemcpy(vertex_indices, mesh.vertex_indices, 3 * num_triangles * sizeof(int), cudaMemcpyHostToDevice);
    }

    __device__ bool intersect(Ray *ray, Intersect *hit) {
        Eigen::Vector3f orig = ray->origin;
        Eigen::Vector3f dir = ray->direction;

        hit->happened = false;

        // Iterate over triangles of mesh
        for (int k = 0; k < num_triangles; ++k) {
            Eigen::Vector3f v0 = vertices[vertex_indices[k * 3]];
            Eigen::Vector3f v1 = vertices[vertex_indices[k * 3 + 1]];
            Eigen::Vector3f v2 = vertices[vertex_indices[k * 3 + 2]];

            float tNear, u, v;

            // Test ray-triangle intersection
            if (ray_triangle_intersect(ray, v0, v1, v2, &tNear, &u, &v) && tNear < hit->t) {
                // Update intersection
                hit->happened = true;
                hit->t = tNear;
                hit->point = orig + tNear * dir;
                hit->index = k;
                hit->uv = Eigen::Vector2f(u, v);

                // Get normal by hit->index
                hit->normal = get_normal(hit);
            }
        }

        return hit->happened;
    }

    __device__ Eigen::Vector3f get_diffuse_color(Intersect *hit) {
        float scale = 5;
        int index = hit->index;
        Eigen::Vector2f uv = hit->uv;

        // Get st coordinates
        Eigen::Vector2f st0 = st_coords[vertex_indices[index * 3]];
        Eigen::Vector2f st1 = st_coords[vertex_indices[index * 3 + 1]];
        Eigen::Vector2f st2 = st_coords[vertex_indices[index * 3 + 2]];

        // Get pattern of mesh
        Eigen::Vector2f st = st0 * (1 - uv[0] - uv[1]) + st1 * uv[0] + st2 * uv[1];
        float pattern = (fmodf(st[0] * scale, 1) > 0.5) ^ (fmodf(st[1] * scale, 1) > 0.5);

        // Return red-yellow pattern
        Eigen::Vector3f red = Eigen::Vector3f(0.815, 0.235, 0.031);
        Eigen::Vector3f yellow = Eigen::Vector3f(0.937, 0.937, 0.231);

        return (1 - pattern) * red + pattern * yellow;
    }

    __device__ Eigen::Vector3f get_normal(Intersect *hit) {
        int index = hit->index;
        
        Eigen::Vector3f v0 = vertices[vertex_indices[index * 3]];
        Eigen::Vector3f v1 = vertices[vertex_indices[index * 3 + 1]];
        Eigen::Vector3f v2 = vertices[vertex_indices[index * 3 + 2]];

        Eigen::Vector3f E0 = (v1 - v0).normalized();
        Eigen::Vector3f E1 = (v2 - v1).normalized();
        
        return (E0.cross(E1)).normalized();

        return Eigen::Vector3f(0, 0, 0);
    }

    __device__ bool ray_triangle_intersect(
        Ray *ray,
        Eigen::Vector3f v0,
        Eigen::Vector3f v1,
        Eigen::Vector3f v2,
        float *tNear,
        float *u,
        float *v
    ) {
        Eigen::Vector3f orig = ray->origin;
        Eigen::Vector3f dir = ray->direction;
        
        // Compute some vectors
        Eigen::Vector3f S = orig - v0;
        Eigen::Vector3f E1 = v1 - v0;
        Eigen::Vector3f E2 = v2 - v0;
        Eigen::Vector3f S1 = dir.cross(E2);
        Eigen::Vector3f S2 = S.cross(E1);

        // Compute time t, barycentric coordinates b1, b2
        float t = S2.dot(E2) / S1.dot(E1);
        float b1 = S1.dot(S) / S1.dot(E1);
        float b2 = S2.dot(dir) / S1.dot(E1);

        // t >= 0, on the ray
        // b1, b2, 1 - b1 - b2 >= 0, inside the triangle
        if (t >= 0 && b1 >= 0 && b2 >= 0 && 1 - b1 - b2 >= 0) {
            *tNear = t;
            *u = b1;
            *v = b2;
            return true;
        } else {
            return false;
        }
    }

    // Material information of object
    Material material;

    // Information of triangle mesh
    int num_triangles;
    int max_index;
    Eigen::Vector3f *vertices;
    Eigen::Vector2f *st_coords;
    int *vertex_indices;
};

#endif // OBJECT_CUH