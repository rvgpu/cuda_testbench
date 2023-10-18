cmake_minimum_required(VERSION 3.21)

project(MyCUDAProject)
enable_testing()

function(add_cudatb target fname)
    set(fpath ${CMAKE_CURRENT_SOURCE_DIR}/${fname})
    set(cmd clang++ -o ${target} ${fpath} --cuda-gpu-arch=rv64g -L /usr/local/cuda/lib64/ -lcudart -ldl -lrt -lgtest -lgtest_main -pthread)

    add_custom_target(${target} ALL COMMAND ${cmd} DEPENDS ${fname})
    add_test(NAME ${target} COMMAND ${target})
endfunction()

add_subdirectory(basic_compute)
