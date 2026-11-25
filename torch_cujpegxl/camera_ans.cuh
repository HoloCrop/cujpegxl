// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "src/camera_decode.h"
// JPEG XL VarDCT context tables, also present in third_party/jxldec/jxl.c.
// The bit reader validates untrusted entropy data before consuming each field.
__constant__ uint32_t coeff_freq_context[63] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 15, 16, 16, 17, 17,
    18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25,
    25, 26, 26, 26, 26, 27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30};
__constant__ uint32_t coeff_num_nonzero_context[63] = {
    0,   31,  62,  62,  93,  93,  93,  93,  123, 123, 123, 123, 152, 152, 152, 152,
    152, 152, 152, 152, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180,
    206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
    206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206};

struct CameraAnsReader {
    const unsigned* data;
    unsigned pos, state, limit;
    bool failed;
    const camera_bucket* buckets;
    unsigned log_bucket, cursor = 1, available = 0;
    uint64_t buffer = 0;
    __device__ unsigned bits(unsigned n) {
        if (pos + n > limit) {
            failed = true;
            return 0;
        }
        if (available < n) {
            buffer |= uint64_t(data[cursor++]) << available;
            available += 32;
        }
        unsigned value = unsigned(buffer) & ((1u << n) - 1);
        pos += n;
        buffer >>= n;
        available -= n;
        return value;
    }
    __device__ unsigned read(unsigned cl) {
        unsigned idx = state & 4095, i = idx >> log_bucket, sub = idx & ((1u << log_bucket) - 1);
        const auto b = buckets[(cl << (12 - log_bucket)) + i];
        bool alias = sub >= b.cutoff;
        unsigned token = alias ? b.symbol : i;
        state = (state >> 12) * (b.frequency ^ (alias ? b.frequency_xor : 0)) + sub +
                (alias ? b.offset : 0);
        if (state < 65536)
            state = (state << 16) | bits(16);
        if (token < 16)
            return token;
        unsigned n = (2 + ((token - 16) >> 2)) & 31;
        return ((4 + (token & 3)) << n) | bits(n);
    }
};
// Each warp advances one serial ANS state. Four groups share the model in
// on-chip memory while the warp scheduler interleaves their dependent reads.
__global__ void decode_groups(const char* data, camera_plan plan, const camera_block* headers,
                              int* output, int* status, int stride, int planes) {
    extern __shared__ unsigned model[];
    __shared__ int rows[4][3][32];
    const auto* source = reinterpret_cast<const unsigned*>(data + plan.histogram_offset);
    for (unsigned k = threadIdx.x; k < plan.model_bytes / 4; k += blockDim.x)
        model[k] = source[k];
    __syncthreads();
    unsigned warp = threadIdx.x / 32, i = blockIdx.x * 4 + warp;
    if (threadIdx.x % 32 != 0 || i >= plan.groups)
        return;
    const auto g = reinterpret_cast<const camera_group*>(data)[i];
    const auto* words = reinterpret_cast<const unsigned*>(data + g.offset);
    CameraAnsReader br{words,          32,    words[0],
                       g.size * 8,     false, reinterpret_cast<const camera_bucket*>(model),
                       plan.log_bucket};
    const auto* clusters =
        reinterpret_cast<const uint8_t*>(model) + (plan.context_offset - plan.histogram_offset);
    unsigned block_contexts = plan.block_clusters;
    auto& nz = rows[warp];
    for (int y = 0; y < g.height; y++)
        for (int x = 0; x < g.width; x++)
            for (int channel_index = 0; channel_index < 3; channel_index++) {
                bool chroma = channel_index != 0;
                if (chroma && (x & 1))
                    continue;
                constexpr int channel_order[] = {1, 0, 2};
                int c = channel_order[channel_index];
                int sx = chroma ? x / 2 : x;
                unsigned pred = y == 0
                                    ? (sx == 0 ? 32 : nz[c][sx - 1])
                                    : (sx == 0 ? nz[c][sx] : (nz[c][sx] + nz[c][sx - 1] + 1) / 2);
                unsigned plane_offset[] = {0, (unsigned)planes, 3u * planes};
                unsigned block = plane_offset[c] + (g.y + y) * (chroma ? stride / 2 : stride) +
                                 (chroma ? g.x / 2 : g.x) + sx;
                unsigned bc = headers[block].entry;
                unsigned nzctx = bc + (pred >= 8 ? 4 + pred / 2 : pred) * block_contexts;
                unsigned left = br.read(clusters[nzctx]);
                if (left > 63) {
                    atomicExch(status, -1);
                    return;
                }
                nz[c][sx] = left;
                unsigned prev = left <= 4;

                for (int k = 1; left && k < 64; k++) {
                    unsigned cl_ctx =
                        (coeff_num_nonzero_context[left - 1] + coeff_freq_context[k - 1]) * 2 +
                        prev;
                    if (cl_ctx >= 458) {
                        atomicExch(status, -2);
                        return;
                    }
                    unsigned u = br.read(clusters[bc * 458 + 37 * block_contexts + cl_ctx]);
                    if (u)
                        output[block * 64 + k] = (u >> 1) ^ -(int)(u & 1);
                    prev = (u != 0);
                    left -= prev;
                }
                if (left || br.failed) {
                    atomicExch(status, -3);
                    return;
                }
            }
    if (br.state != 0x130000)
        atomicExch(status, -4);
}
