cmake_minimum_required (VERSION 3.21)

project(cuda-examples LANGUAGES CUDA CXX)

enable_language(CUDA)

include(CheckLanguage)
check_language(CUDA)


if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_subdirectory ("src")

# add_library(helpers STATIC
#     helper_cuda.h
#     helper_string.h
#   )

# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
#target_compile_features(helpers PUBLIC cxx_std_11)

# set_target_properties( helpers
#                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON
#                        LINKER_LANGUAGE CXX
#                       )


