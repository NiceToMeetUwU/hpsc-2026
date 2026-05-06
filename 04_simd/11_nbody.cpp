#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
    const int N = 16;
    
    // AVX2 requires 32-byte alignment (8 floats * 4 bytes)
    float *x, *y, *m, *fx, *fy;
    x = (float*)aligned_alloc(32, N * sizeof(float));
    y = (float*)aligned_alloc(32, N * sizeof(float));
    m = (float*)aligned_alloc(32, N * sizeof(float));
    fx = (float*)aligned_alloc(32, N * sizeof(float));
    fy = (float*)aligned_alloc(32, N * sizeof(float));

    for(int i=0; i<N; i++) {
        x[i] = drand48();
        y[i] = drand48();
        m[i] = drand48();
        fx[i] = fy[i] = 0.0f;
    }

    // Load all 16 particles into two registers (8 each)
    __m256 v_xj_low  = _mm256_load_ps(x);
    __m256 v_xj_high = _mm256_load_ps(x + 8);
    __m256 v_yj_low  = _mm256_load_ps(y);
    __m256 v_yj_high = _mm256_load_ps(y + 8);
    __m256 v_mj_low  = _mm256_load_ps(m);
    __m256 v_mj_high = _mm256_load_ps(m + 8);

    for(int i = 0; i < N; i++) {
        __m256 v_xi = _mm256_set1_ps(x[i]);
        __m256 v_yi = _mm256_set1_ps(y[i]);
        __m256 v_sum_fx = _mm256_setzero_ps();
        __m256 v_sum_fy = _mm256_setzero_ps();

        // We process in 2 chunks of 8
        for(int block = 0; block < 2; block++) {
            __m256 v_xj = (block == 0) ? v_xj_low : v_xj_high;
            __m256 v_yj = (block == 0) ? v_yj_low : v_yj_high;
            __m256 v_mj = (block == 0) ? v_mj_low : v_mj_high;

            // rx = xi - xj
            __m256 v_rx = _mm256_sub_ps(v_xi, v_xj);
            __m256 v_ry = _mm256_sub_ps(v_yi, v_yj);

            // r2 = rx*rx + ry*ry
            __m256 v_r2 = _mm256_fmadd_ps(v_rx, v_rx, _mm256_mul_ps(v_ry, v_ry));

            // Masking i != j
            // In AVX2, we compare r2 > 0 to create a mask
            __m256 v_mask = _mm256_cmp_ps(v_r2, _mm256_setzero_ps(), _CMP_GT_OQ);

            // r = sqrt(r2)
            __m256 v_r = _mm256_sqrt_ps(v_r2);
            
            // fmag = m / (r * r2) -> this is m / r^3
            __m256 v_r3 = _mm256_mul_ps(v_r, v_r2);
            __m256 v_fmag = _mm256_div_ps(v_mj, v_r3);

            // Apply mask: if i == j, result is 0
            v_fmag = _mm256_and_ps(v_mask, v_fmag);

            v_sum_fx = _mm256_fmadd_ps(v_rx, v_fmag, v_sum_fx);
            v_sum_fy = _mm256_fmadd_ps(v_ry, v_fmag, v_sum_fy);
        }

        // Horizontal add for AVX2
        float res_x[8], res_y[8];
        _mm256_store_ps(res_x, v_sum_fx);
        _mm256_store_ps(res_y, v_sum_fy);

        for(int k=0; k<8; k++) {
            fx[i] -= res_x[k];
            fy[i] -= res_y[k];
        }

        printf("%d %g %g\n", i, fx[i], fy[i]);
    }

    free(x); free(y); free(m); free(fx); free(fy);
    return 0;
}