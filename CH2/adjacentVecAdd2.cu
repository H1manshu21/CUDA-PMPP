//  to calculate two adjacent elements of a vector addition.
#include <cuda.h>
#include <iostream>
#include <random>


__global__ void vecAddKernel(int *d_vec, int *d_resVec, int n) {
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    if (i < (n - 1)) {
        d_resVec[i] = d_vec[i] + d_vec[i + 1];
    }

    int j = i + blockDim.x;

    if (j < (n - 1)) {
        d_resVec[j] = d_vec[j] + d_vec[j + 1];
    }
}

void vecAdd(int *vec, int *resVec, int n) {
    int size = n * sizeof(int);
    int resSize = (n - 1) * sizeof(int);

    int *d_vec, *d_resVec;

    cudaMalloc(&d_vec, size);
    cudaMalloc(&d_resVec, resSize);

    cudaMemcpy(d_vec, vec, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + (threadsPerBlock * 2) - 1) / (threadsPerBlock * 2) ;

    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_vec, d_resVec, n);

    cudaMemcpy(resVec, d_resVec, resSize, cudaMemcpyDeviceToHost);

    cudaFree(d_vec);
    cudaFree(d_resVec);
}

int main() {
    int n;
    std::cin >> n;

    int *vec = new int[n];
    int *resVec = new int[n - 1];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 4);

    for (int i = 0; i < n; ++i) {
        vec[i] = dist(gen);
    }

    vecAdd(vec, resVec, n);

    std::cout << "Elements in vec:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << vec[i] << " ";
    }

    std::cout << "\nTwo adjacent elements of a vector addition:" << std::endl;

    for (int i = 0; i < 10; ++i) {
        std::cout << resVec[i] << " ";
    }

    std::cout << "\n";

    delete[] vec;
    delete[] resVec;

    return 0;
}