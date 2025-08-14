#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>
#include<pthread.h>
#include "utils.h"



void geMM_naive(float **a, float **b, float **c, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            c[i][j] = 0.0f;
        }
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int t = 0; t < k; t++) {
                c[i][j] += a[i][t] * b[t][j];
            }
        }
    }
}

void geMM_Gustavson(float **a, float **b, float **c, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int t = 0; t < k; t++) {
            for (int j = 0; j < n; j++) {
                c[i][j] += a[i][t] * b[t][j];
            }
        }
    }
}

void geMM_blocking(float **a, float **b, float **c, int m, int k, int n) {
    int block_size = 64;
    for (int i = 0; i < m; i += block_size) {
        for (int j = 0; j < n; j += block_size) {
            for (int l = 0; l < k; l += block_size) {
                for (int ii = i; ii < min(i + block_size, m); ii++) {
                    for (int jj = j; jj < min(j + block_size, n); jj++) {
                        float sum = 0.0f;
                        for (int kk = l; kk < min(l + block_size, k); kk++) {
                            sum += a[ii][kk] * b[kk][jj];
                        }
                        c[ii][jj] += sum;
                    }
                }
            }
        }
    }
}

// 定义线程参数结构体
typedef struct {
    int start_row; // 线程处理的起始行
    int end_row;   // 线程处理的结束行
    float **a, **b, **c;
    int m, k, n;
    int block_size;
} ThreadData;

void *thread_func(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    int start_row = data->start_row;
    int end_row = data->end_row;
    float **a = data->a;
    float **b = data->b;
    float **c = data->c;
    int m = data->m, k = data->k, n = data->n;

    int block_size = 64;
    for (int i = start_row; i < end_row; i += block_size) {
        for (int j = 0; j < n; j += block_size) {
            for (int l = 0; l < k; l += block_size) {
                for (int ii = i; ii < (i + block_size < m ? i + block_size : m); ii++) {
                    for (int kk = l; kk < (l + block_size < k ? l + block_size : k); kk++) {
                        for (int jj = j; jj < (j + block_size < n ? j + block_size : n); jj++) {
                            c[ii][jj] += a[ii][kk] * b[kk][jj];
                        }
                    }
                }
            }
        }
    }
    pthread_exit(NULL);
}

void *thread_func_gust(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    int start_row = data->start_row;
    int end_row = data->end_row;
    float **a = data->a;
    float **b = data->b;
    float **c = data->c;
    int m = data->m, k = data->k, n = data->n;

    for (int i = start_row; i < end_row; i++) {
        for (int l = 0; l < k; l++) {
            for (int j = 0; j < n; j++) {
                c[i][j] += a[i][l] * b[l][j];
            }
        }
    }
    pthread_exit(NULL);
}

void testGemm(float **a, float **b, float **c, int m, int k, int n) {
    int num_threads = 16;
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];

    int rows_per_thread = m / num_threads;
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = (i == num_threads - 1) ? m : (i + 1) * rows_per_thread;
        thread_data[i].a = a, thread_data[i].b = b, thread_data[i].c = c;
        thread_data[i].m = m, thread_data[i].k = k, thread_data[i].n = n;
        pthread_create(&threads[i], NULL, thread_func_gust, &thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}



/*
    优化一：把Weight矩阵Stationary住，通过将其的分块存入缓存实现
*/
// 单线程版本
void spMM(float **A, float **B, float **C, int m, int k, int n) {
    
}