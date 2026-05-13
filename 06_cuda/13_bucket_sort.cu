#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>


__global__ void count_kernel(int* d_key, int* d_bucket, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int val = d_key[idx];
        atomicAdd(&d_bucket[val], 1);
    }
}

__global__ void fill_kernel(int* d_key, int* d_bucket, int range) {
    int j = 0;
    for (int i = 0; i < range; i++) {
        int count = d_bucket[i];
        for (int k = 0; k < count; k++) {
            d_key[j++] = i;
        }
    }
}

int main() {
    int n = 50;
    int range = 5;
    size_t key_size = n * sizeof(int);
    size_t bucket_size = range * sizeof(int);

    // Host
    std::vector<int> h_key(n);
    for (int i = 0; i < n; i++) {
        h_key[i] = rand() % range;
        printf("%d ", h_key[i]);
    }
    printf("\n");

    // Device
    int *d_key, *d_bucket;
    cudaMalloc(&d_key, key_size);
    cudaMalloc(&d_bucket, bucket_size);

    // Initialize device memory
    cudaMemcpy(d_key, h_key.data(), key_size, cudaMemcpyHostToDevice);
    cudaMemset(d_bucket, 0, bucket_size); 

    // start counting
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    count_kernel<<<gridSize, blockSize>>>(d_key, d_bucket, n);

    std::vector<int> h_bucket(range);
    cudaMemcpy(h_bucket.data(), d_bucket, bucket_size, cudaMemcpyDeviceToHost);

    int j = 0;
    for (int i = 0; i < range; i++) {
        while (h_bucket[i] > 0) {
            h_key[j++] = i;
            h_bucket[i]--;
        }
    }

    
    for (int i = 0; i < n; i++) {
        printf("%d ", h_key[i]);
    }
    printf("\n");

    
    cudaFree(d_key);
    cudaFree(d_bucket);

    return 0;
}