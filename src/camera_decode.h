// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <stddef.h>
#include <stdint.h>

// The entry selects an ANS block context or the first sparse payload word.
typedef struct {
    uint32_t entry;
    float dc, scale;
} camera_block;

typedef struct {
    float weights[3][64], bias[3], bias_numerator;
    unsigned char frequency[3][64];
} camera_tables;

typedef struct {
    uint32_t groups, block_clusters, context_offset, histogram_offset, log_bucket, model_bytes;
} camera_plan;

typedef int32_t* (*camera_allocate)(void*, size_t);
typedef struct {
    uint8_t symbol, cutoff;
    uint16_t frequency, offset, frequency_xor;
} camera_bucket;

typedef struct {
    uint32_t offset, size, x, y, width, height;
} camera_group;
