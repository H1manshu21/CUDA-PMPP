// Each thread process one input.
#include <cuda.h>
#include <iostream>
#include <random>

__global__ void adjacentAdd(int *d_vec, int *d_resVec, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < (n - 1)) {
        d_resVec[i] = d_vec[i] + d_vec[i + 1];
    }
}


void adjacentAddInit(int *vec, int *resVec, int n) {
    int size = n * sizeof(int);

    int *d_vec;
    int *d_resVec;

    cudaMalloc(&d_vec, size);
    cudaMalloc(&d_resVec, (n - 1) * sizeof(int));

    cudaMemcpy(d_vec, vec, size, cudaMemcpyHostToDevice);

    adjacentAdd<<<ceil(n / 256.0), 256>>>(d_vec, d_resVec, n);

    cudaMemcpy(resVec, d_resVec, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);

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
    std::uniform_int_distribution<int> dist(1, 9);
    
    for (int i = 0; i < n; ++i) {
        vec[i] = dist(gen);
    }

    for (int i = 0; i < n; ++i) {
        std::cout << vec[i] << " ";
    }

    std::cout << "\n";

    adjacentAddInit(vec, resVec, n);

    std::cout << "Ajacent add of a vector:" << std::endl;

    for (int i = 0; i < (n - 1); ++i) {
        std::cout << resVec[i] << " ";
    }

    std::cout << "\n";

    delete[] vec;
    delete[] resVec;

    return 0;
}