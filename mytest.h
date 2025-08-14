#include<string.h>
#include "mm.h"
#include "sparse.h"
// #include "pool.h"

double test_csr(int m, int k, int n, float s, int tile_m, int tile_k, int tile_n, int num_threads) {
    float **spm = (float**)malloc(m * sizeof(float*));
    float **dense = (float**)malloc(k * sizeof(float*));
    float **res = (float**)malloc(m * sizeof(float*));
    // float **tmp = (float**)malloc(m * sizeof(float*));
    for (int i = 0; i < m; i++) {
        spm[i] = (float*)malloc(k * sizeof(float));
        res[i] = (float*)malloc(n * sizeof(float));
        // tmp[i] = (float*)malloc(n * sizeof(float));
    }
    for (int i = 0; i < k; i++) {
        dense[i] = (float*)malloc(n * sizeof(float));
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            res[i][j] = 0.0f;
            // tmp[i][j] = 0.0f;
        }
    }

    
    // 构建Tile信息
    TileInfo *tileinfo = (TileInfo*)malloc(sizeof(TileInfo));
    tileinfo->tile_size_m = tile_m, tileinfo->tile_size_k = tile_k, tileinfo->tile_size_n = tile_n;
    float **B_tiles = (float**)malloc(num_threads * sizeof(float*));
    for (int t = 0; t < num_threads; t++) {
        B_tiles[t] = (float*)malloc(tile_k * tile_n * sizeof(float));
    }
    
    // 初始化线程池
    // ThreadPool pool(std::thread::hardware_concurrency());
    
    // warm-up
    int ans = generateSparseMat(s, spm, m, k);   // ans为非零元素个数
    generate_random_matrix(dense, k, n);
    int *row_ptr = (int*)malloc((m + 1) * sizeof(int));
    int *col = (int*)malloc(ans * sizeof(int));
    float *val = (float*)malloc(ans * sizeof(float));
    for (int t = 0; t < 100; t++) {
        Mat2CSR(spm, m, k, row_ptr, col, val);
        spmm_csr_dense_tiling_opt(m, k, row_ptr, col, val, dense, n, res, tileinfo, B_tiles, num_threads);
    }
    free(row_ptr);
    free(col);
    free(val);
    // 跑100轮，取后30轮的平均时间
    struct timeval start, end;
    double avg = 0.0f;
    for (int t = 0; t < 100; t++) {
        // 生成随机矩阵
        int ans = generateSparseMat(s, spm, m, k);   // ans为非零元素个数
        generate_random_matrix(dense, k, n);
        int *row_ptr = (int*)malloc((m + 1) * sizeof(int));
        int *col = (int*)malloc(ans * sizeof(int));
        float *val = (float*)malloc(ans * sizeof(float));

        gettimeofday(&start, NULL);

        // CSR
        Mat2CSR(spm, m, k, row_ptr, col, val);
        spmm_csr_dense_tiling_opt(m, k, row_ptr, col, val, dense, n, res, tileinfo, B_tiles, num_threads);

        gettimeofday(&end, NULL);
        double time_ = getTime(start, end);
        avg += time_;

        free(row_ptr);
        free(col);
        free(val);
    }

    // free
    for (int i = 0; i < m; i++) {
        free(spm[i]);
        free(res[i]);
    }
    for (int i = 0; i < k; i++) {
        free(dense[i]);
    }
    for (int i = 0; i < num_threads; i++) {
        free(B_tiles[i]);
    }
    free(B_tiles);
    free(spm);
    free(res);
    free(dense);
    free(tileinfo);

    return avg / 100;
}

void findBestTile(int m, int k, int n, float sparsity, int num_thread) {
    
    double bestTime = 100;
    int bs_m, bs_k, bs_n, bs_t;
    for (int tm = 8; tm <= 64; tm <<= 1) {
        for (int tk = 256; tk <= 1024; tk <<= 1) {
            for (int tn = 8; tn <= 64; tn <<= 1) {
                // for (int num_thread = 32; num_thread <= 64; num_thread <<= 1) {
                    double tmp = test_csr(m, k, n, sparsity, tm, tk, tn, num_thread);
                    printf("time:%.5f, tm:%d, tk:%d, tn:%d, num_thread:%d\n", tmp, tm, tk, tn, num_thread);
                    if (tmp <= bestTime) {
                        bestTime = tmp;
                        bs_m = tm, bs_k = tk, bs_n = tn, bs_t = num_thread;
                        // printf("\n-----UPDATED-----\n");
                        // printf("time:%.5f, tm:%d, tk:%d, tn:%d, threads:%d\n", tmp, tm, tk, tn, num_thread);
                    }
                // }
                
            }
        }
    }
    printf("\n--------DONE--------\n");
    printf("-----time:%.5f, tm:%d, tk:%d, tn:%d, threads:%d-----\n", bestTime, bs_m, bs_k, bs_n, bs_t);

    FILE *fp = fopen("results.csv", "a");
    if (fp == NULL) {
        perror("Can't open the file.");
        return;
    }

    fprintf(fp, "%-8d %-8d %-8d %-8d %-8d %-8d %-8d %-10.6lf", m, k, n, bs_m, bs_k, bs_n, bs_t, bestTime);
    fprintf(fp, "\n");
    fclose(fp);
}

void testTile_1(float sparsity, int tm) {    // 连续做4个矩阵乘
    // 做四种尺寸的矩阵乘
    int M = 197;
    int K[4] = {2304, 768, 3072, 768};
    int N[4] = {768, 3072, 768, 768};

    double bestTime = 100;
    double *bestTmp = (double*)malloc(4 * sizeof(double));
    double *tmp = (double*)malloc(4 * sizeof(double));
    double ans = 0.0f;
    int bs_m, bs_k, bs_n, bs_t;
    //for (int tm = 8; tm <= 128; tm <<= 1) {
        for (int tk = 128; tk <= 1024; tk <<= 1) {
            for (int tn = 8; tn <= 1024; tn <<= 1) {
                for (int num_thread = 32; num_thread <= 64; num_thread <<= 1) {
                    // 遍历四种尺寸
                    ans = 0.0f;
                    for (int i = 0; i < 4; i++) {
                        tmp[i] = test_csr(M, K[i], N[i], sparsity, tm, tk, tn, num_thread);
                        ans += tmp[i];
                    }
                    if (ans <= bestTime) {
                        bestTime = ans;
                        for (int i = 0; i < 4; i++) bestTmp[i] = tmp[i];     // 记录不同尺寸的单独时间
                        bs_m = tm, bs_k = tk, bs_n = tn, bs_t = num_thread;
                        printf("\n-----UPDATED-----\n");
                        printf("time:%.5lf, tm:%d, tk:%d, tn:%d, threads:%d\n", ans, tm, tk, tn, num_thread);
                        printf("m1:%.5lf, m2:%.5lf, m3:%.5lf, m4:%.5lf\n", tmp[0], tmp[1], tmp[2], tmp[3]);
                    }
                }
            }
        }
    // }
    printf("\n--------DONE--------\n");
    printf("-----time:%.5f, tm:%d, tk:%d, tn:%d, threads:%d-----\n", bestTime, bs_m, bs_k, bs_n, bs_t);

    FILE *fp = fopen("results.csv", "a");
    if (fp == NULL) {
        perror("Can't open the file.");
        return;
    }
    fprintf(fp, "%-8d %-8d %-8d %-8d %-10.6f %-10.6f %-10.6f %-10.6lf %-10.6f", bs_m, bs_k, bs_n, bs_t, bestTmp[0], bestTmp[1], bestTmp[2], bestTmp[3], bestTime);
    fprintf(fp, "\n");
    fclose(fp);

    free(tmp);
    free(bestTmp);
}