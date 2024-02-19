#include "image.h"
#include <iostream>
#include "cuda_runtime.h"
#include "helper_cuda.h"

using namespace std;

#define FILTER_RADIUS 1
#define TILE_DIM 8

__constant__ float F[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

// We assume that the convolutional filter is square
__global__
void sharedImageConvolutionHalo(const float *A, float *B, int r, int width , int height)
{

    __shared__ double N_ds[TILE_DIM][TILE_DIM];

    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    // Load memory into shared
    if ((row < height) && (col < width))
    {
        N_ds[threadIdx.y][threadIdx.x] = A[row * width + col];
    }
    else
    {
        N_ds[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    if ((row < height) && (col < width))
    {
        float Pval = 0.0f;
        for (int i = 0; i < 2*FILTER_RADIUS + 1; ++i)
        {
            for (int j = 0; j < 2*FILTER_RADIUS + 1; ++j)
            {
                if ( (threadIdx.y - FILTER_RADIUS + i>= 0) && 
                     (threadIdx.x - FILTER_RADIUS + j >= 0) &&
                     (threadIdx.y - FILTER_RADIUS + i < blockDim.y) &&
                     (threadIdx.x - FILTER_RADIUS + j < blockDim.x))
                {
                    // Case where the processed image value is inside the shared memory block
                    Pval += F[i][j] * N_ds[threadIdx.y - FILTER_RADIUS + i][threadIdx.x - FILTER_RADIUS + j];
                }
                else 
                {
                    if ( (row - FILTER_RADIUS + i >= 0) &&
                         (row - FILTER_RADIUS + i < height) &&
                         (col - FILTER_RADIUS + j >= 0) &&
                         (col - FILTER_RADIUS + j < width))
                    {
                        // Case where the processed image value is outside shared memory and
                        // we need to copy it from global
                        Pval += F[i][j] * A[(row-FILTER_RADIUS+i) * width + col-FILTER_RADIUS+j];
                    }
                }   
            }
        }
        B[row*width + col] = Pval;
    }
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

    const dim3 dimGrid((int)ceil((im.w)/(float)TILE_DIM), (int)ceil((im.h)/(float)TILE_DIM));
	const dim3 dimBlock(TILE_DIM, TILE_DIM, 1);

    int radious = floor(sobelXFilter.w/2);

    cout << "Radious: " << radious << endl;
    cout << "dimGrid(" << dimGrid.x <<", "<< dimGrid.y << ")" << endl;
    cout << "dimBlock(" << dimBlock.x <<", "<< dimBlock.y << ")" << endl;

    sharedImageConvolutionHalo <<< dimGrid, dimBlock  >>> (d_A, d_B, radious, im.w, im.h);

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
    save_image(output, (char *)"sobelSharedHalo");

    free_image(im);
    free_image(sobelXFilter);
    
    cudaFree(d_A);
    cudaFree(d_B);


    return 0;
}