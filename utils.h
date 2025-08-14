#include <stdio.h>

double getTime(struct timeval start, struct timeval end);
void generate_random_matrix(float **matrix, int rows, int cols);
int generateSparseMat(float sparsity, float **arr, int m, int n);
int identical(float **A, float **B, int ma, int mb, int na, int nb);
int min(int a, int b);
void print(float **arr, int m, int n);

// 获取时间
double getTime(struct timeval start, struct timeval end) {
    long seconds = end.tv_sec - start.tv_sec;
    long microseconds = end.tv_usec - start.tv_usec;
    double time_taken = seconds + microseconds / 1e6;
    return time_taken;
}

// 生成随机矩阵
void generate_random_matrix(float **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = (float)rand() / RAND_MAX;
        }
    }
}

// 生成随机稀疏矩阵
int generateSparseMat(float sparsity, float **arr, int m, int n) {
    int ans = 0;
    srand(time(NULL));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float r = (float)rand() / RAND_MAX;
            if (r > sparsity) {
                arr[i][j] = (float)(rand() % 11 - 5);
                ans++;
            }
            else arr[i][j] = 0.0f;
        }
    }
    return ans;
}

// 判断矩阵相同，相同返回1，否则0
int identical(float **A, float **B, int ma, int mb, int na, int nb) {
    if (ma != mb || na != nb) return 0;
    for (int i = 0; i < ma; i++) {
        for (int j = 0; j < na; j++) {
            if (fabs(A[i][j] - B[i][j]) > 1e-6) return 0;
        }
    }
    return 1;
}

// 整数最小值
int min(int a, int b) {
    return (a >= b) ? b : a;
}

// 打印矩阵
void print(float **arr, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", arr[i][j]);
        }
        printf("\n");
    }
    printf("\n");
} 

