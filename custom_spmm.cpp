#include<stdio.h>
#include<stdlib.h>
#include<immintrin.h>
#include<omp.h>

/*
    调用customized_spmm()
    
*/

void Mat2CSR(float **arr, int m, int n, int *row_ptr, int *col, float *val);
void spmm_opt(int rows_A, int cols_A, 
                int *row_ptr, int *col, float *values,
                float **B, int cols_B,
                float **C, int tile_size_m, int tile_size_k, int tile_size_n,
                float **B_tiles, int num_thread);
void customized_spmm(float **spikes, float **weight, float **res, int m, int k, int n);

void Mat2CSR(float **arr, int m, int n, int *row_ptr, int *col, float *val) {
    int cnt = 0;
    row_ptr[0] = 0;
    for (int i = 0; i < m; i++) {
        float *row = arr[i];
        for (int j = 0; j < n; j+=8) {
            __m256 vec = _mm256_loadu_ps(&row[j]);
            __m256 mask = _mm256_cmp_ps(vec, _mm256_setzero_ps(), _CMP_NEQ_OQ);
            int bitmask = _mm256_movemask_ps(mask);
            for (int k = 0; k < 8 && j + k < n; k++) {
                if (bitmask & (1 << k)) {
                    col[cnt] = j + k;
                    val[cnt++] = row[j + k];
                }
            }
        }
        row_ptr[i + 1] = cnt;  // 移到外层循环，减少写入次数
    }
}

void spmm_opt(int rows_A, int cols_A, 
                int *row_ptr, int *col, float *values,
                float **B, int cols_B,
                float **C, int tile_size_m, int tile_size_k, int tile_size_n,
                float **B_tiles, int num_thread) {

    int Tm = (rows_A + tile_size_m - 1) / tile_size_m;
    int Tk = (cols_A + tile_size_k - 1) / tile_size_k;
    int Tn = (cols_B + tile_size_n - 1) / tile_size_n;

    // stationary B
    float *B_tile = B_tiles[omp_get_thread_num()];
    int *block_starts = (int*)malloc(rows_A * Tk * sizeof(int));
    
    // #pragma omp parallel for num_threads(32) schedule(dynamic)
    for (int i = 0; i < rows_A; i++) {
        int row_start = row_ptr[i];
        int row_end = row_ptr[i+1];
        for (int k_idx = 0; k_idx < Tk; k_idx++) {
            int reduct_start = k_idx * tile_size_k;
            int low = row_start;
            int high = row_end;
            while (high - low > 32) { // 小范围转线性搜索
                int mid = low + ((high - low) >> 1);
                low = (col[mid] < reduct_start) ? mid + 1 : low;
                high = (col[mid] >= reduct_start) ? mid : high;
            }
            while (low < high && col[low] < reduct_start) low++;
            block_starts[i * Tk + k_idx] = low;
        }
    }

    #pragma omp parallel for num_threads(num_thread) schedule(dynamic)
    for (int j = 0; j < Tn; j++) {
        const int colB_start = j * tile_size_n;
        const int colB_end = (colB_start + tile_size_n > cols_B) ? cols_B : colB_start + tile_size_n;
        const int colB_len = colB_end - colB_start;
        
        float *B_tile = (float*)malloc(tile_size_k * colB_len * sizeof(float));

        for (int k = 0; k < Tk; k++) {
            const int reduct_start = k * tile_size_k;
            const int reduct_end = (reduct_start + tile_size_k > cols_A) ? cols_A : reduct_start + tile_size_k;
            
            for (int bk = reduct_start; bk < reduct_end; bk++) {
                for (int bj = 0; bj < colB_len; bj++) {
                    B_tile[(bk - reduct_start) * colB_len + bj] = B[bk][colB_start + bj];
                }
            }
            
            for (int i = 0; i < Tm; i++) {
                const int rowA_start = i * tile_size_m;
                const int rowA_end = (rowA_start + tile_size_m > rows_A)  ? rows_A : rowA_start + tile_size_m;
                for (int ii = rowA_start; ii < rowA_end; ii++) {
                    // 通过block_start快速找到起始位置
                    const int kk_start = block_starts[ii * Tk + k];
                    const int kk_end = row_ptr[ii+1];
                    for (int kk = kk_start; kk < kk_end && col[kk] < reduct_end; kk++) {
                        const float val = values[kk];
                        const int b_row = col[kk] - reduct_start;
                        float *C_row = &C[ii][colB_start];
                        float *B_row = &B_tile[b_row * colB_len];
                    
                        // SIMD-256
                        #if defined(__AVX__)
                            __m256 val_vec = _mm256_set1_ps(val);
                            for (int jj = 0; jj <= colB_len - 8; jj += 8) {
                                __m256 b_vec = _mm256_loadu_ps(B_row + jj);
                                __m256 c_vec = _mm256_loadu_ps(C_row + jj);
                                c_vec = _mm256_fmadd_ps(val_vec, b_vec, c_vec);
                                _mm256_storeu_ps(C_row + jj, c_vec);
                            }
                            // 处理尾部元素
                            for (int jj = colB_len - (colB_len % 8); jj < colB_len; jj++) {
                                C_row[jj] += val * B_row[jj];
                            }
                        #else
                            // 非SIMD回退
                            for (int jj = 0; jj < colB_len; jj++) {
                                C_row[jj] += val * B_row[jj];
                            }
                        #endif
                    }
                }
            }
        }
    }
    free(block_starts);
}

void customized_spmm(float **spikes, float **weight, float **res, int m, int k, int n) {
    
    int tile_size_m, tile_size_k, tile_size_n;
    int num_threads;
    // 判断tile_size和num_threads
    if (m == 197 && (k == 2304 || k == 3072) && n == 768) {
        tile_size_m = 64, tile_size_k = 1024, tile_size_n = 32, num_threads = 64;
    }
    else if (m == 197 && k == 768 && n == 768) {
        tile_size_m = 32, tile_size_k = 1024, tile_size_n = 32, num_threads = 32;
    }
    else if (m == 197 && k == 768 && n == 3072) {
        tile_size_m = 64, tile_size_k = 1024, tile_size_n = 64, num_threads = 32;
    }
    else {
        tile_size_m = 32, tile_size_k = 1024, tile_size_n = 32, num_threads = 32;
    }
    printf("Size: %d %d %d, tile: %d %d %d, threads: %d\n", m, k, n, tile_size_m, tile_size_k, tile_size_n, num_threads);
    
    float **B_tiles = (float**)malloc(num_threads * sizeof(float*));
    for (int t = 0; t < num_threads; t++) {
        B_tiles[t] = (float*)malloc(tile_size_k * tile_size_n * sizeof(float));
    }
    int *row_ptr = (int*)malloc((m + 1) * sizeof(int));
    int *col = (int*)malloc(m * k * sizeof(int));
    float *val = (float*)malloc(m * k * sizeof(float));
    Mat2CSR(spikes, m, k, row_ptr, col, val);
    spmm_opt(m, k, row_ptr, col, val, weight, n, res, tile_size_m, tile_size_k, tile_size_n, B_tiles, num_threads);
}
