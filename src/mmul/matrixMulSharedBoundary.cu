#include "matrix.h"
#include "image.h"
#include <iostream>
#include "cuda_runtime.h"
#include "helper_cuda.h"

#define TILE_WIDTH 2

__global__
void matrixMultipliShared(const double *M, const double *N, double *P, int width)
{
    __shared__ double Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ double Nds[TILE_WIDTH][TILE_WIDTH];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    double cum = 0.0;
    
    for (int ph = 0; ph < width/TILE_WIDTH; ++ph)
    {
        // Load data to shared memory

        if ( ((threadIdx.x + (TILE_WIDTH*ph)) < width) && (row < width))
        {
            Mds[threadIdx.y][threadIdx.x] = M[row  * width + (threadIdx.x+(TILE_WIDTH*ph))];
        }
        else
        {
            Mds[threadIdx.y][threadIdx.x] = 0.0;
        }
        if ( ((threadIdx.y + (TILE_WIDTH * ph)) < width) && (col < width))
        {
            Nds[threadIdx.y][threadIdx.x] = N[(threadIdx.y + (TILE_WIDTH * ph)) * width + col];
        }
        else
        {
            Nds[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++)
        {
            cum += Mds[threadIdx.y][i] * Nds[i][threadIdx.x];
        }
        __syncthreads();
    }

    if ( (row < width) && (col < width))
    {
        P[row * width + col] = cum;
    }  
}


int main() {

    // This script allows to multiply matrices that don't match
    // size with  shared_memory block_size
    
    matrix M = make_matrix(3, 3);
    matrix N = make_matrix(3, 3);
    matrix P = make_matrix(3, 3);
    
    for (int i=0; i<M.rows ; ++i)
    {
        for (int j=0; j < M.cols; ++j)
        {
            M.data[i*M.cols + j] = j+1;
        }
    }
    for (int i=0; i<N.rows ; ++i)
    {
        for (int j=0; j < N.cols; ++j)
        {
            N.data[i*N.cols + j] = 1;
        }
    }

    print_matrix(M);

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

    const dim3 dimGrid((int)ceil((M.cols)/2.0), (int)ceil((M.rows)/2.0));
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);

    matrixMultipliShared <<< dimGrid, blockDim  >>> (d_M, d_N, d_P, M.rows);

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

    free_matrix(M);
    free_matrix(N);
    free_matrix(P);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}