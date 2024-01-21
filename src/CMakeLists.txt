
add_executable(cmake_cuda vector-addition/vectorAddition.cu )
target_include_directories(cmake_cuda PUBLIC "../include")
set_target_properties(cmake_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(cmake_cuda PRIVATE helpers)