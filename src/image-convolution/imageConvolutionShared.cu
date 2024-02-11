#include "image.h"
#include <iostream>
#include "cuda_runtime.h"
#include "helper_cuda.h"

using namespace std;

#define FILTER_RADIUS 1
#define IN_TILE_DIM 8
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))

__constant__ float F[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

// We assume that the convolutional filter is square
__global__
void sharedImageConvolution(const float *A, float *B, int r, int width , int height)
{

    __shared__ double N_ds[IN_TILE_DIM][IN_TILE_DIM];

    int col = OUT_TILE_DIM * blockIdx.x + threadIdx.x - r;
    int row = OUT_TILE_DIM * blockIdx.y + threadIdx.y - r;

    int input_col = blockDim.x * blockIdx.x + threadIdx.x - r;
    int input_row = blockDim.y * blockIdx.y + threadIdx.y - r;

    // Load memory into shared
    if ( (row >= 0) && (row < height) && (col >= 0) && (col < width))
    {
        N_ds[threadIdx.x][threadIdx.y] = A[row * width + col];
    }
    else
    {
        N_ds[threadIdx.x][threadIdx.y] = 0.0f;
    }
    __syncthreads();

    
}

int main() {

    image im = load_image((char*)"../../data/gray.jpg");
    image output = make_image(1, im.h, im.w);

    image sobelXFilter = make_image(1, 3, 3);
    // First row values
    set_pixel(sobelXFilter, 0, 0, 0, -1);
    set_pixel(sobelXFilter, 0, 0, 2, 1);
    // Second row values
    set_pixel(sobelXFilter, 0, 1, 0, -2);
    set_pixel(sobelXFilter, 0, 1, 2, 2);
    // Third row values
    set_pixel(sobelXFilter, 0, 2, 0, -1);
    set_pixel(sobelXFilter, 0, 2, 2, 1);


    size_t inputImgBytes = im.c*im.w*im.h*sizeof(decltype(*im.data));
    size_t inputFilterBytes = sobelXFilter.w*sobelXFilter.h*sizeof(decltype(*sobelXFilter.data));
    size_t outputImgBytes = im.w*im.h*sizeof(decltype(*im.data));

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, inputImgBytes);

    if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, outputImgBytes);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy input filter
    err = cudaMemcpyToSymbol(F, sobelXFilter.data, inputFilterBytes);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy sobel filter to Global Memory (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_A, im.data, inputImgBytes, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to copy vector A from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    const dim3 dimGrid((int)ceil((im.w)/(float)OUT_TILE_DIM), (int)ceil((im.h)/(float)OUT_TILE_DIM));
	const dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM, 1);

    int radious = floor(sobelXFilter.w/2);

    cout << "Radious: " << radious << endl;
    cout << "dimGrid(" << dimGrid.x <<", "<< dimGrid.y << ")" << endl;
    cout << "dimBlock(" << dimBlock.x <<", "<< dimBlock.y << ")" << endl;

    sharedImageConvolution <<< dimGrid, dimBlock  >>> (d_A, d_B, radious, im.w, im.h);

    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cout << "Copy output data from the CUDA device to the host memory" << endl;

    err = cudaMemcpy(output.data, d_B, outputImgBytes, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to copy vector C from device to host (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    feature_normalize(output);
    save_image(output, (char *)"sobelShared");

    free_image(im);
    free_image(sobelXFilter);
    
    cudaFree(d_A);
    cudaFree(d_B);


    return 0;
}