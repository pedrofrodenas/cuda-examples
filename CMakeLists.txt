cmake_minimum_required (VERSION 3.21)

project(cuda-examples LANGUAGES CUDA CXX C)

enable_language(CUDA)

include(CheckLanguage)
check_language(CUDA)


if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_subdirectory ("src")


