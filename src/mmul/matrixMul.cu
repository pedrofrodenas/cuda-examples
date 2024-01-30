#include "matrix.h"
#include "image.h"
#include <iostream>
#include "cuda_runtime.h"
#include "helper_cuda.h"

__global__
void matrixMultipli(const double *M, const double *N, double *P, int width)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if ((col < width) && (row < width))
    {
        int index = row * width + col;
        float result  = 0.0;

        for (int i=0; i < width; ++i)
        {
            result += M[(width*row) + i] * N[ (i*width) + col];
        }
        P[index] = result;   
        
    }
}


int main() {

    
    matrix M = make_matrix(4, 4);
    matrix N = make_matrix(4, 4);
    matrix P = make_matrix(4, 4);
    
    for (int i=0; i<M.rows ; ++i)
    {
        for (int j=0; j < M.cols; ++j)
        {
            M.data[i*M.cols + j] = 2.0;
        }
    }
    for (int i=0; i<N.rows ; ++i)
    {
        for (int j=0; j < N.cols; ++j)
        {
            N.data[i*N.cols + j] = 4.0;
        }
    }

    size_t inputBytesM = M.rows*M.cols*sizeof(decltype(*M.data));
    size_t inputBytesN = N.rows*N.cols*sizeof(decltype(*N.data));
    size_t outputBytes = M.rows*N.cols*sizeof(decltype(*N.data));

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Allocate the device input matrix M
    double *d_M = NULL;
    err = cudaMalloc((void **)&d_M, inputBytesM);

    if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device matrix M (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }

    // Allocate the device input matrix N
    double *d_N = NULL;
    err = cudaMalloc((void **)&d_N, inputBytesN);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device matrix N (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Allocate the device output matrix P
    double *d_P = NULL;
    err = cudaMalloc((void **)&d_P, outputBytes);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device output matrix P (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_M, M.data, inputBytesM, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to copy matrix M from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_N, N.data, inputBytesN, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to copy matrix N from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    const dim3 dimGrid((int)ceil((M.cols)/16.0), (int)ceil((M.rows)/16.0));
	const dim3 dimBlock(16, 16, 1);

    matrixMultipli <<< dimGrid, dimBlock  >>> (d_M, d_N, d_P, M.rows);

    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch matrixMultipli kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    std::cout << "Copy output data from the CUDA device to the host memory" << std::endl;

    err = cudaMemcpy(P.data, d_P, outputBytes, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to copy matrix P from device to host (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    print_matrix(P);

    return 0;
}