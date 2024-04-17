cmake_minimum_required(VERSION 3.28)
project(first_weekend CUDA)

set(CMAKE_CUDA_STANDARD 23)

add_executable(first_weekend main.cu)

set_target_properties(first_weekend PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
