cmake_minimum_required(VERSION 3.10)
project(Graphics)

find_package(CUDA REQUIRED)

cuda_add_executable(games101_hw5
    Global.cuh
    Light.cuh
    main.cu
    Object.cuh
    Renderer.cu
    Renderer.cuh
    Scene.cuh
)