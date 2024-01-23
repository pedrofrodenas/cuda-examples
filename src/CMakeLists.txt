add_library(Image SHARED
    "image-utils/load_image.c"
  )

target_compile_features(Image PUBLIC cxx_std_11)

target_include_directories (Image PUBLIC "../include")

set_target_properties( Image
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                       LINKER_LANGUAGE CXX
                      )
                    


add_executable(cmake_cuda vector-addition/vectorAddition.cu )
target_include_directories(cmake_cuda PUBLIC "../include")
set_target_properties(cmake_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


add_executable(ImgtoGray image-grayscale/imageGray.cu )
target_include_directories(ImgtoGray PRIVATE "../include")
target_link_libraries (ImgtoGray PRIVATE Image)


set_target_properties(ImgtoGray PROPERTIES CUDA_SEPARABLE_COMPILATION ON)