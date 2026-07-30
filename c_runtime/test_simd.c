#include "deepspaceslm.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    printf("=== DeepSpace-SLM Bare-Metal C SIMD INT8 Benchmark ===\n");

    const size_t vec_len = 1024;
    const size_t num_iters = 100000;

    int8_t *a = (int8_t*)malloc(vec_len);
    int8_t *b = (int8_t*)malloc(vec_len);

    for (size_t i = 0; i < vec_len; i++) {
        a[i] = (int8_t)((i % 127) - 64);
        b[i] = (int8_t)(((i * 3) % 127) - 64);
    }

    clock_t start = clock();
    int32_t total_sum = 0;
    for (size_t iter = 0; iter < num_iters; iter++) {
        total_sum += dslm_dot_product_int8(a, b, vec_len);
    }
    clock_t end = clock();

    double elapsed_sec = (double)(end - start) / CLOCKS_PER_SEC;
    double total_ops = (double)num_iters * (double)vec_len * 2.0; // 1 mult + 1 add
    double gflops = (total_ops / elapsed_sec) / 1e9;

    printf("Vector Length: %zu elements\n", vec_len);
    printf("Total Iterations: %zu\n", num_iters);
    printf("Total Operations: %.2f Million Ops\n", total_ops / 1e6);
    printf("Execution Time: %.4f seconds\n", elapsed_sec);
    printf("SIMD Throughput: \033[32m%.2f GFLOPs / GOPS\033[0m\n", gflops);
    printf("Sample Dot Product Result: %d\n", total_sum);

    free(a);
    free(b);
    return 0;
}
