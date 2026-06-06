#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline_primitives.h>
#include <cublas_v2.h> // 僅用於產生對答案的基準矩陣

using namespace std;
using namespace nvcuda;

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

// ==========================================
// H100 終極非同步管線 Kernel (128x128 Tile + 16-Byte DMA)
// ==========================================
__global__ void wmma_h100_pipeline_kernel(int dim_m, int dim_n, int dim_k,
                                          const half *d_a, const half *d_b, float *d_c) {
    const int BLOCK_M = 128;
    const int BLOCK_N = 128;
    const int BLOCK_K = 32;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    
    // 【L2 Cache Grid Swizzling】
    const int SWIZZLE = 8;
    int mapped_x = (bx / SWIZZLE) * SWIZZLE + (by % SWIZZLE);
    int mapped_y = (by / SWIZZLE) * SWIZZLE + (bx % SWIZZLE);
    if (mapped_x < gridDim.x && mapped_y < gridDim.y) {
        bx = mapped_x; 
        by = mapped_y;
    }

    int offset_m = bx * BLOCK_M;
    int offset_n = by * BLOCK_N;

    // 【雙緩衝 Shared Memory + Padding】
    __shared__ half smem_A[2][BLOCK_K][BLOCK_M + 8]; 
    __shared__ half smem_B[2][BLOCK_N][BLOCK_K + 8]; 

    // Warp 分工 (每個 Warp 負責 32x64)
    int warp_id = tx / 32;
    int warp_row = (warp_id % 4) * 32; 
    int warp_col = (warp_id / 4) * 64; 

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            wmma::fill_fragment(acc[r][c], 0.0f);
        }
    }

    // 【硬體修正：16-Byte 完美對齊記憶體映射】
    int a_idx1 = tx;
    int a_idx2 = tx + 256;
    int a_col1 = a_idx1 / 16; 
    int a_row1 = (a_idx1 % 16) * 8;
    int a_col2 = a_idx2 / 16;
    int a_row2 = (a_idx2 % 16) * 8;

    int b_idx1 = tx;
    int b_idx2 = tx + 256;
    int b_col1 = b_idx1 / 4;
    int b_row1 = (b_idx1 % 4) * 8;
    int b_col2 = b_idx2 / 4;
    int b_row2 = (b_idx2 % 4) * 8;

    // 【Prologue：預先載入第 0 步】
    if (0 < dim_k) {
        __pipeline_memcpy_async(&smem_A[0][a_col1][a_row1], &d_a[a_col1 * dim_m + offset_m + a_row1], 16);
        __pipeline_memcpy_async(&smem_A[0][a_col2][a_row2], &d_a[a_col2 * dim_m + offset_m + a_row2], 16);
        
        __pipeline_memcpy_async(&smem_B[0][b_col1][b_row1], &d_b[(offset_n + b_col1) * dim_k + b_row1], 16);
        __pipeline_memcpy_async(&smem_B[0][b_col2][b_row2], &d_b[(offset_n + b_col2) * dim_k + b_row2], 16);
        __pipeline_commit(); 
    }

    // 主迴圈
    for (int k = 0; k < dim_k; k += BLOCK_K) {
        int next_k = k + BLOCK_K;
        int load_idx = (k / BLOCK_K + 1) % 2; 
        int calc_idx = (k / BLOCK_K) % 2;       

        // 【發起下一輪的非同步載入】
        if (next_k < dim_k) {
            __pipeline_memcpy_async(&smem_A[load_idx][a_col1][a_row1], &d_a[(next_k + a_col1) * dim_m + offset_m + a_row1], 16);
            __pipeline_memcpy_async(&smem_A[load_idx][a_col2][a_row2], &d_a[(next_k + a_col2) * dim_m + offset_m + a_row2], 16);
            
            __pipeline_memcpy_async(&smem_B[load_idx][b_col1][b_row1], &d_b[(offset_n + b_col1) * dim_k + next_k + b_row1], 16);
            __pipeline_memcpy_async(&smem_B[load_idx][b_col2][b_row2], &d_b[(offset_n + b_col2) * dim_k + next_k + b_row2], 16);
        }
        __pipeline_commit(); 
        
        __pipeline_wait_prior(1); 
        __syncthreads();

        // 內部 Tensor Core 運算迴圈
        for (int step = 0; step < BLOCK_K; step += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[4];

            #pragma unroll
            for (int r = 0; r < 2; r++) 
                wmma::load_matrix_sync(a_frag[r], &smem_A[calc_idx][step][warp_row + r * 16], BLOCK_M + 8);
            
            #pragma unroll
            for (int c = 0; c < 4; c++) 
                wmma::load_matrix_sync(b_frag[c], &smem_B[calc_idx][warp_col + c * 16][step], BLOCK_K + 8);

            #pragma unroll
            for (int r = 0; r < 2; r++) {
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    wmma::mma_sync(acc[r][c], a_frag[r], b_frag[c], acc[r][c]);
                }
            }
        }
        __syncthreads();
    }

    // 寫回 Global Memory
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            int c_m = offset_m + warp_row + r * 16;
            int c_n = offset_n + warp_col + c * 16;
            if (c_m < dim_m && c_n < dim_n) {
                wmma::store_matrix_sync(&d_c[c_n * dim_m + c_m], acc[r][c], dim_m, wmma::mem_col_major);
            }
        }
    }
}

int main(int argc, const char **argv) {
    int m = 10240;
    int k = 4096;
    int n = 8192;
    int Nt = 10;
    
    half *A, *B;
    float *C_custom, *C_ref;

    printf("Allocating Unified Memory...\n");
    CHECK_CUDA(cudaMallocManaged(&A, m * k * sizeof(half)));
    CHECK_CUDA(cudaMallocManaged(&B, k * n * sizeof(half)));
    CHECK_CUDA(cudaMallocManaged(&C_custom, m * n * sizeof(float)));
    CHECK_CUDA(cudaMallocManaged(&C_ref, m * n * sizeof(float)));

    printf("Initializing data on CPU (Converting float to half)...\n");
    for (int i = 0; i < m * k; i++) A[i] = __float2half((float)drand48());
    for (int i = 0; i < k * n; i++) B[i] = __float2half((float)drand48());
    for (int i = 0; i < n * m; i++) {
        C_custom[i] = 0.0f;
        C_ref[i] = 0.0f;
    }

    // ==========================================
    // 使用 CUBLAS 產生對答案的基準 (Ground Truth)
    // ==========================================
    printf("Generating ground truth via CUBLAS (this is not timed)...\n");
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    
    // 注意：因為 A, B 現在是 half，所以必須指定 CUDA_R_16F
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 m, n, k, &alpha,
                 A, CUDA_R_16F, m, 
                 B, CUDA_R_16F, k, 
                 &beta,
                 C_ref, CUDA_R_32F, m,
                 CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    CHECK_CUDA(cudaDeviceSynchronize());
    cublasDestroy(handle);

    // ==========================================
    // 測量 Custom Kernel 的極致效能
    // ==========================================
    int tile_m = 128;
    int tile_n = 128;
    dim3 block(256);
    dim3 grid((m + tile_m - 1) / tile_m, (n + tile_n - 1) / tile_n);

    printf("Running H100 Optimized Custom Kernel...\n");
    auto tic = chrono::steady_clock::now();
    for (int i = 0; i < Nt + 2; i++) {
        if (i == 2) tic = chrono::steady_clock::now(); 
        wmma_h100_pipeline_kernel<<<grid, block>>>(m, n, k, A, B, C_custom);
        CHECK_CUDA(cudaGetLastError()); 
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    auto toc = chrono::steady_clock::now();
    
    int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
    double tcustom = chrono::duration<double>(toc - tic).count() / Nt;
    double custom_flops = double(num_flops) / tcustom / 1.0e9;

    printf("\n====================================\n");
    printf(" Custom Kernel: %8.2f GFLOPS\n", custom_flops);
    printf("====================================\n\n");

    // ==========================================
    // 計算 Error Rate
    // ==========================================
    printf("Calculating error matrix (D2H migration)...\n");
    double err = 0.0;
    for (int i = 0; i < n * m; i++) {
        err += fabs(C_ref[i] - C_custom[i]);
    }
    double avg_error = err / (double)(n * m);
    printf("Average Error: %lf\n", avg_error);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C_custom);
    cudaFree(C_ref);
    
    return 0;
}