#include "mytest.h"


int main(int argc, char *argv[]) {

    if (argc == 2) {
        float sparsity = atof(argv[1]);
        int tm = atoi(argv[2]);
        testTile_1(sparsity, tm);
    }
    else if (argc >= 6) {
        int m = atoi(argv[1]), k = atoi(argv[2]), n = atoi(argv[3]);
        float sparsity = atof(argv[4]);
        if (argc == 9) {
            int tile_m = atoi(argv[5]), tile_k = atoi(argv[6]), tile_n = atoi(argv[7]);
            int num_threads = atoi(argv[8]);
            double avg_time = test_csr(m, k, n, sparsity, tile_m, tile_k, tile_n, num_threads);
            printf("Avg Time: %.6lf\n", avg_time);
        }
        else if (argc == 6) {
            int num_threads = atoi(argv[5]);
            // for (int i = 0; i < 5; i++) {
                findBestTile(m, k, n, sparsity, num_threads);
            // }
        }
    }
    return 0;
}