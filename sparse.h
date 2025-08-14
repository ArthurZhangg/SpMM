#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<time.h>
#include<immintrin.h>
#include<omp.h>
// #include "ThreadPool.h"

typedef struct {
    int tile_size_m;
    int tile_size_k;
    int tile_size_n;
} TileInfo;

void Mat2CSR(float **arr, int m, int n, int *row_ptr, int *col, float *val) {
    // memset(row_ptr, 0, (m + 1) * sizeof(int));
    int cnt = 0;
    row_ptr[0] = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (arr[i][j] != 0) {
                // row_ptr[i + 1]++;
                col[cnt] = j;
                val[cnt++] = arr[i][j];
            }
            row_ptr[i + 1] = cnt;
        }
    }
}

void Mat2COO(float **arr, int m, int n, int *row, int *col, float *val, int *num) {
    int k = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (arr[i][j]) {
                row[k] = i;
                col[k] = j;
                val[k++] = arr[i][j]; 
            }
        }
    }
    *num = k;
}

void COO2CSR(int *row, int *col, float *v_i, int nums_i, int rows_i, int *row_ptr, int *col_idx, float *v_o) {
    // 计算row_ptr（前缀和）
    memset(row_ptr, 0, rows_i * sizeof(int));
    for (int i = 0; i < nums_i; i++) {
        row_ptr[row[i] + 1]++;
    }
    for (int i = 0; i < rows_i; i++) {
        row_ptr[i + 1] += row_ptr[i]; 
    }

    for (int i = 0; i < nums_i; i++) {
        v_o[i] = v_i[i];
        col_idx[i] = col[i];
    }
}

// 未分块版
void spmm_csr_dense(int rows_A, int cols_A, 
                    int *row_ptr, int *col_idx, float *values,
                    float **B, int cols_B,
                    float **C, TileInfo *info) {

    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < cols_B; j++) {
            C[i][j] = 0.0f;
        }
    }

    for (int i = 0; i < rows_A; i++) {
        int row_start = row_ptr[i];
        int row_end = row_ptr[i + 1];
        for (int idx = row_start; idx < row_end; idx++) {
            for (int j = 0; j < cols_B; j++) {
                C[i][j] += values[idx] * B[col_idx[idx]][j];
            }
        }
    }
}

// 分块但未做其他优化
void spmm_csr_dense_tiling(int rows_A, int cols_A, 
                    int *row_ptr, int *col, float *values,
                    float **B, int cols_B,
                    float **C, TileInfo *info) {

    int tile_size_m = info->tile_size_m, tile_size_k = info->tile_size_k, tile_size_n = info->tile_size_n;
    int Tm = rows_A / tile_size_m, Tk = cols_A / tile_size_k, Tn = cols_B / tile_size_n;

    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < cols_B; j++) {
            C[i][j] = 0.0f;
        }
    }
    // naive dataflow
    for (int j = 0; j < Tn; j++) {
        for (int k = 0; k < Tk; k++) {
            for (int i = 0; i < Tm; i++) {
                // 分好了块A:[i, k]; 块B:[k, j]
                int rowA_start = i * tile_size_m;
                int rowA_end = (rowA_start + tile_size_m > rows_A) ? rows_A : rowA_start + tile_size_m;
                int reduct_start = k * tile_size_k;
                int reduct_end = (reduct_start + tile_size_k > cols_A) ? cols_A : reduct_start + tile_size_k;
                int colB_start = j * tile_size_n;
                int colB_end = (colB_start + tile_size_n > cols_B) ? cols_B : colB_start + tile_size_n;

                // 切割Sparse矩阵
                for (int ii = rowA_start; ii < rowA_end; ii++) {
                    int tile_row_start = row_ptr[ii];   // 表示这个小块的当前行的第一个非零元素在CSR val中的位置
                    while (col[tile_row_start] < reduct_start) tile_row_start++;
                    for (int kk = tile_row_start; kk < row_ptr[ii + 1] && col[kk] < reduct_end; kk++) {   // kk是在CSR数组的下标
                        for (int jj = colB_start; jj < colB_end; jj++) {
                            C[ii][jj] += values[kk] * B[col[kk]][jj];
                        }
                    }
                }
            }
        }
    }
}

// 分块优化版
void spmm_csr_dense_tiling_opt(int rows_A, int cols_A, 
                    int *row_ptr, int *col, float *values,
                    float **B, int cols_B,
                    float **C, TileInfo *info,
                    float **B_tiles, int num_thread) {

    // 根据硬件缓存调整分块大小
    int tile_size_m = info->tile_size_m;   // L1d优化: 32-64
    int tile_size_k = info->tile_size_k;   // L2优化: 256-512
    int tile_size_n = info->tile_size_n;   // L2优化: 128-256
    
    // 计算分块数量 (向上取整)
    int Tm = (rows_A + tile_size_m - 1) / tile_size_m;
    int Tk = (cols_A + tile_size_k - 1) / tile_size_k;
    int Tn = (cols_B + tile_size_n - 1) / tile_size_n;

    // stationary B
    // float *B_tile = (float*)malloc(tile_size_k * tile_size_n * sizeof(float));
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
        // printf("线程 %d (共 %d 线程) 已启动\n",
        //     omp_get_thread_num(),     // 当前线程ID (0~N-1)
        //     omp_get_num_threads());   // 总线程数
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

                        // SIMD-512
                        // #if defined(__AVX512F__)
                        //     __m512 val_vec = _mm512_set1_ps(val);  // 设置值到 512 位寄存器
                        //     for (int jj = 0; jj <= colB_len - 16; jj += 16) {  // 每次加载 16 个浮点数
                        //         __m512 b_vec = _mm512_loadu_ps(B_row + jj);  // 加载 512 位数据
                        //         __m512 c_vec = _mm512_loadu_ps(C_row + jj);  // 加载 512 位数据
                        //         c_vec = _mm512_fmadd_ps(val_vec, b_vec, c_vec);  // 进行乘法加法操作
                        //         _mm512_storeu_ps(C_row + jj, c_vec);  // 存储 512 位结果
                        //     }
                        //     // 处理尾部元素
                        //     for (int jj = colB_len - (colB_len % 16); jj < colB_len; jj++) {
                        //         C_row[jj] += val * B_row[jj];  // 处理剩余部分
                        //     }
                        // #else
                        //     // 非 SIMD 回退
                        //     for (int jj = 0; jj < colB_len; jj++) {
                        //         C_row[jj] += val * B_row[jj];
                        //     }
                        // #endif

                    }
                }
            }
        }
    }
    free(block_starts);
}

void spmm_csr_dense_tiling5(int rows_A, int cols_A, 
                    int *row_ptr, int *col, float *values,
                    float **B, int cols_B,
                    float **C, TileInfo *info) {

    // 根据硬件缓存调整分块大小
    int tile_size_m = info->tile_size_m;   // L1d优化: 32-64
    int tile_size_k = info->tile_size_k;   // L2优化: 256-512
    int tile_size_n = info->tile_size_n;   // L2优化: 128-256
    
    // 计算分块数量 (向上取整)
    int Tm = (rows_A + tile_size_m - 1) / tile_size_m;
    int Tk = (cols_A + tile_size_k - 1) / tile_size_k;
    int Tn = (cols_B + tile_size_n - 1) / tile_size_n;

    // stationary B
    float **B_tiles = (float**)malloc(32 * sizeof(float*));
    float *B_tile = (float*)malloc(tile_size_k * tile_size_n * sizeof(float));
    int *block_starts = (int*)malloc(rows_A * Tk * sizeof(int));
    // #pragma omp parallel for num_threads(4) schedule(dynamic)
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
    // #pragma omp parallel for num_threads(4) schedule(dynamic)
    for (int j = 0; j < Tn; j++) {
        // printf("线程 %d (共 %d 线程) 已启动\n",
        //     omp_get_thread_num(),     // 当前线程ID (0~N-1)
        //     omp_get_num_threads());   // 总线程数
        const int colB_start = j * tile_size_n;
        const int colB_end = (colB_start + tile_size_n > cols_B) ? cols_B : colB_start + tile_size_n;
        const int colB_len = colB_end - colB_start;
        
        for (int k = 0; k < Tk; k++) {
            const int reduct_start = k * tile_size_k;
            const int reduct_end = (reduct_start + tile_size_k > cols_A) ? cols_A : reduct_start + tile_size_k;
            
            //  *B_tile = (float*)malloc((reduct_end - reduct_start) * colB_len * sizeof(float));
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
                    
                        // SIMD
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
    free(B_tile);
    free(block_starts);
}
