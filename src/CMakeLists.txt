add_library(Utils SHARED
    "utils/load_image.c"
    "utils/matrix.c"
)

target_compile_features(Utils PUBLIC cxx_std_11)

target_include_directories (Utils PUBLIC "../include")

set_target_properties( Utils
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                       LINKER_LANGUAGE CXX
                      )                    


add_executable(cmake_cuda vector-addition/vectorAddition.cu )
target_include_directories(cmake_cuda PUBLIC "../include")
set_target_properties(cmake_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


add_executable(ImgtoGray image-grayscale/imageGray.cu )
target_include_directories(ImgtoGray PRIVATE "../include")
target_link_libraries (ImgtoGray PRIVATE Utils)
set_target_properties(ImgtoGray PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(ImgBlur image-blur/imageBlur.cu )
target_include_directories(ImgBlur PRIVATE "../include")
target_link_libraries (ImgBlur PRIVATE Utils)
set_target_properties(ImgBlur PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


add_executable(mMul mmul/matrixMul.cu)
target_include_directories(mMul PRIVATE "../include")
target_link_libraries (mMul PRIVATE Utils)
set_target_properties(mMul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
