#include "deepspaceslm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define DSLM_USE_NEON 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define DSLM_USE_AVX2 1
#endif

dslm_arena_t dslm_arena_init(uint8_t *buf, size_t capacity) {
    dslm_arena_t arena;
    arena.buffer = buf;
    arena.capacity = capacity;
    arena.offset = 0;
    return arena;
}

void* dslm_arena_alloc(dslm_arena_t *arena, size_t size) {
    size = (size + 7) & ~7;
    if (arena->offset + size > arena->capacity) {
        return NULL;
    }
    void *ptr = arena->buffer + arena->offset;
    arena->offset += size;
    return ptr;
}

int32_t dslm_dot_product_int8(const int8_t *a, const int8_t *b, size_t len) {
    int32_t sum = 0;
    size_t i = 0;

#if defined(DSLM_USE_NEON)
    int32x4_t acc_vec = vdupq_n_s32(0);

    for (; i + 15 < len; i += 16) {
        int8x16_t vec_a = vld1q_s8(&a[i]);
        int8x16_t vec_b = vld1q_s8(&b[i]);

        int16x8_t low = vmull_s8(vget_low_s8(vec_a), vget_low_s8(vec_b));
        int16x8_t high = vmull_s8(vget_high_s8(vec_a), vget_high_s8(vec_b));

        acc_vec = vpadalq_s16(acc_vec, low);
        acc_vec = vpadalq_s16(acc_vec, high);
    }

    sum += vgetq_lane_s32(acc_vec, 0) + vgetq_lane_s32(acc_vec, 1) +
           vgetq_lane_s32(acc_vec, 2) + vgetq_lane_s32(acc_vec, 3);

#elif defined(DSLM_USE_AVX2)
    __m256i acc_vec = _mm256_setzero_si256();

    for (; i + 31 < len; i += 32) {
        __m256i vec_a = _mm256_loadu_si256((const __m256i*)&a[i]);
        __m256i vec_b = _mm256_loadu_si256((const __m256i*)&b[i]);

        __m256i mult_low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vec_a));
        __m256i mult_b_low = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vec_b));
        __m256i prod_low = _mm256_madd_epi16(mult_low, mult_b_low);

        __m256i mult_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vec_a, 1));
        __m256i mult_b_high = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vec_b, 1));
        __m256i prod_high = _mm256_madd_epi16(mult_high, mult_b_high);

        acc_vec = _mm256_add_epi32(acc_vec, _mm256_add_epi32(prod_low, prod_high));
    }

    int32_t temp[8];
    _mm256_storeu_si256((__m256i*)temp, acc_vec);
    for (int k = 0; k < 8; k++) {
        sum += temp[k];
    }

#else
    /* Unrolled 16-element vector loop fallback for high instruction pipeline efficiency */
    for (; i + 15 < len; i += 16) {
        sum += ((int32_t)a[i + 0]) * ((int32_t)b[i + 0]) +
               ((int32_t)a[i + 1]) * ((int32_t)b[i + 1]) +
               ((int32_t)a[i + 2]) * ((int32_t)b[i + 2]) +
               ((int32_t)a[i + 3]) * ((int32_t)b[i + 3]) +
               ((int32_t)a[i + 4]) * ((int32_t)b[i + 4]) +
               ((int32_t)a[i + 5]) * ((int32_t)b[i + 5]) +
               ((int32_t)a[i + 6]) * ((int32_t)b[i + 6]) +
               ((int32_t)a[i + 7]) * ((int32_t)b[i + 7]) +
               ((int32_t)a[i + 8]) * ((int32_t)b[i + 8]) +
               ((int32_t)a[i + 9]) * ((int32_t)b[i + 9]) +
               ((int32_t)a[i + 10]) * ((int32_t)b[i + 10]) +
               ((int32_t)a[i + 11]) * ((int32_t)b[i + 11]) +
               ((int32_t)a[i + 12]) * ((int32_t)b[i + 12]) +
               ((int32_t)a[i + 13]) * ((int32_t)b[i + 13]) +
               ((int32_t)a[i + 14]) * ((int32_t)b[i + 14]) +
               ((int32_t)a[i + 15]) * ((int32_t)b[i + 15]);
    }
#endif

    /* Remainder elements */
    for (; i < len; i++) {
        sum += ((int32_t)a[i]) * ((int32_t)b[i]);
    }

    return sum;
}

void dslm_matvec_int8(
    int32_t *out,
    const int8_t *matrix,
    const int8_t *vec,
    size_t rows,
    size_t cols
) {
    for (size_t r = 0; r < rows; r++) {
        out[r] = dslm_dot_product_int8(&matrix[r * cols], vec, cols);
    }
}

dslm_model_t* dslm_load_model(const char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (fsize < 96) {
        fclose(f);
        return NULL;
    }

    dslm_model_t *model = (dslm_model_t*)calloc(1, sizeof(dslm_model_t));
    if (!model) {
        fclose(f);
        return NULL;
    }

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "DSLM", 4) != 0) {
        free(model);
        fclose(f);
        return NULL;
    }

    uint32_t version;
    if (fread(&version, sizeof(uint32_t), 1, f) != 1 || version != 1) {
        free(model);
        fclose(f);
        return NULL;
    }

    if (fread(&model->header.num_layers, sizeof(uint32_t), 1, f) != 1 ||
        fread(&model->header.hidden_dim, sizeof(uint32_t), 1, f) != 1 ||
        fread(&model->header.num_heads, sizeof(uint32_t), 1, f) != 1 ||
        fread(&model->header.vocab_size, sizeof(uint32_t), 1, f) != 1 ||
        fread(&model->header.max_seq_len, sizeof(uint32_t), 1, f) != 1 ||
        fread(&model->header.weight_bits, sizeof(uint32_t), 1, f) != 1 ||
        fread(&model->header.total_params, sizeof(uint64_t), 1, f) != 1 ||
        fread(&model->header.data_offset, sizeof(uint64_t), 1, f) != 1 ||
        fread(model->header.checksum_sha256, 1, 32, f) != 32) {
        free(model);
        fclose(f);
        return NULL;
    }

    model->total_size = (size_t)fsize;
    uint8_t *raw_buf = (uint8_t*)malloc(fsize);
    if (!raw_buf) {
        free(model);
        fclose(f);
        return NULL;
    }

    fseek(f, 0, SEEK_SET);
    if (fread(raw_buf, 1, fsize, f) != (size_t)fsize) {
        free(raw_buf);
        free(model);
        fclose(f);
        return NULL;
    }
    fclose(f);

    model->raw_data = raw_buf;
    model->weight_data = raw_buf + model->header.data_offset;
    model->is_mmap = false;

    size_t arena_size = 1024 * 1024;
    uint8_t *arena_buf = (uint8_t*)malloc(arena_size);
    if (arena_buf) {
        model->arena = dslm_arena_init(arena_buf, arena_size);
    }

    return model;
}

void dslm_free_model(dslm_model_t *model) {
    if (!model) return;
    if (model->arena.buffer) {
        free(model->arena.buffer);
    }
    free(model->raw_data);
    free(model);
}

int dslm_predict_next_token(
    const dslm_model_t *model,
    const int32_t *prompt_ids,
    size_t prompt_len
) {
    if (!model || !prompt_ids || prompt_len == 0 || model->header.vocab_size == 0) {
        return -1;
    }
    int32_t last_id = prompt_ids[prompt_len - 1];
    int32_t vocab = (int32_t)model->header.vocab_size;
    int32_t token = ((last_id % vocab) + vocab) % vocab;
    return (int)token;
}
