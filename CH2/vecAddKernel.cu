#include <cuda.h>
#include <iostream>
#include <random>

__global__ void vecAddKernel(float *A, float *B, float *C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float *A, float *B, float *C, int n) {
    float *d_A;
    float *d_B;
    float *d_C;
    
    int size = n * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // <<<Number of blocks, Number of threads per block>>>
    vecAddKernel<<<ceil(n / 256.0), 256>>>(d_A, d_B, d_C, n);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    float *A;
    float *B;
    float *C;

    int n;
    std::cout << "Enter number of elements: ";
    std::cin >> n;

    A = new float[n];
    B = new float[n];
    C = new float[n];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);

    for (int i = 0; i < n; ++i) {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }

    vecAdd(A, B, C, n);

    std::cout << "Addition of two vectors:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << A[i] << " + " << B[i] << " = " << C[i] << std::endl;
    }

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}