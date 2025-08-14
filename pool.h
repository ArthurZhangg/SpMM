#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include "utils.h"

class ThreadPool {
public:
    ThreadPool(size_t num_threads) : stop(false) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] {
                            return this->stop || !this->tasks.empty();
                        });
                        if (this->stop && this->tasks.empty()) return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers) {
            worker.join();
        }
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

void spmm_csr_dense_tiling_with_pool(
    int rows_A, int cols_A, 
    int *row_ptr, int *col, float *values,
    float **B, int cols_B,
    float **C, TileInfo *info,
    ThreadPool& pool  // 传入线程池引用
) {
    int tile_size_m = info->tile_size_m;
    int tile_size_k = info->tile_size_k;
    int tile_size_n = info->tile_size_n;

    int Tm = (rows_A + tile_size_m - 1) / tile_size_m;
    int Tk = (cols_A + tile_size_k - 1) / tile_size_k;
    int Tn = (cols_B + tile_size_n - 1) / tile_size_n;

    float *B_tile = (float*)malloc(tile_size_k * tile_size_n * sizeof(float));
    int *block_starts = (int*)malloc(rows_A * Tk * sizeof(int));

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

    // 使用线程池并行化外层循环
    for (int j = 0; j < Tn; j++) {
        for (int k = 0; k < Tk; k++) {
            pool.enqueue([=, &B, &C] {  // 捕获必要的变量
                const int colB_start = j * tile_size_n;
                const int colB_end = (colB_start + tile_size_n > cols_B) ? cols_B : colB_start + tile_size_n;
                const int colB_len = colB_end - colB_start;
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
            });
        }
    }
    // 等待所有任务完成（简单实现，实际需更复杂的同步）
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    free(B_tile);
    free(block_starts);
}
