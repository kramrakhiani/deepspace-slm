#ifndef DEEPSPACESLM_H
#define DEEPSPACESLM_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t *buffer;
    size_t capacity;
    size_t offset;
} dslm_arena_t;

typedef struct {
    uint32_t num_layers;
    uint32_t hidden_dim;
    uint32_t num_heads;
    uint32_t vocab_size;
    uint32_t max_seq_len;
    uint32_t weight_bits;
    uint64_t total_params;
    uint64_t data_offset;
    uint8_t  checksum_sha256[32];
} dslm_header_t;

typedef struct {
    dslm_header_t header;
    const uint8_t *weight_data;
    uint8_t *raw_data;
    size_t total_size;
    bool is_mmap;
    dslm_arena_t arena;
} dslm_model_t;

dslm_arena_t dslm_arena_init(uint8_t *buf, size_t capacity);
void* dslm_arena_alloc(dslm_arena_t *arena, size_t size);

dslm_model_t* dslm_load_model(const char *filepath);
void dslm_free_model(dslm_model_t *model);

/* SIMD-accelerated 8-bit integer dot product */
int32_t dslm_dot_product_int8(const int8_t *a, const int8_t *b, size_t len);

/* SIMD-accelerated 8-bit integer matrix-vector multiplication (y = W * x) */
void dslm_matvec_int8(
    int32_t *out,
    const int8_t *matrix,
    const int8_t *vec,
    size_t rows,
    size_t cols
);

int dslm_predict_next_token(
    const dslm_model_t *model,
    const int32_t *prompt_ids,
    size_t prompt_len
);

#ifdef __cplusplus
}
#endif

#endif
