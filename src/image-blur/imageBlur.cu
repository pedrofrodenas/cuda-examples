#include "image.h"
#include <iostream>
#include "cuda_runtime.h"
#include "helper_cuda.h"

using namespace std;

__global__
void blurImgGray(const float *A, float *B, int width, int height, int kSize, int halfKSzie)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if ((col < width) && (row < height))
    {
        float sumPixel = 0.0f;
        float grayPixel = 0.0f;
        int totalPixels = 0;
        float resultPixel = 0.0f;
        for (int y = -halfKSzie; y != halfKSzie+1; y++) 
        {
            for (int x = -halfKSzie; x != halfKSzie+1; x++)
            {
                if ( (y >= 0) && (y < height) && (x >= 0) && (x < width))
                {
                    grayPixel = A[(row + y)*width + (col + x)];
                    sumPixel += grayPixel;
                    totalPixels++;
                }
            }
        }
        resultPixel = sumPixel / totalPixels;
        B[row * width + col] = resultPixel;
    }
}

int main() {

    image im = load_image((char*)"../../data/gray.jpg");
    image gray = make_image(1, im.h, im.w);

    size_t inputImgBytes = im.c*im.w*im.h*sizeof(decltype(*im.data));
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

    err = cudaMemcpy(d_A, im.data, inputImgBytes, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to copy vector A from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    const dim3 dimGrid((int)ceil((im.w)/16.0), (int)ceil((im.h)/16.0));
	const dim3 dimBlock(16, 16, 1);

    // Calculate kernel size
    constexpr int KernelSize = 15;
    int kernelCenter = floor(KernelSize/2);

    blurImgGray <<< dimGrid, dimBlock  >>> (d_A, d_B, im.w, im.h, KernelSize, kernelCenter);

    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cout << "Copy output data from the CUDA device to the host memory" << endl;

    err = cudaMemcpy(gray.data, d_B, outputImgBytes, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to copy vector C from device to host (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    save_image(gray, (char *)"blur");

    return 0;
}