// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "frontend_fused.h"

#include <algorithm>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "adaptive_quant_impl.cuh"
#include "dct_impl.cuh"
#include "dct_mma_impl.cuh"
#include "dequant_weight_pairs.h"
#include "exact_div.cuh"
#include "gaborish.h"
#include "gpu_arch.h"
#include "quant_calibration.h"
#include "transform_select_impl.cuh"
#include "vardct_layout.h"
#include "warp_scan.cuh"
#include "xyb_impl.cuh"

namespace cujpegxl {
namespace {

// Tile geometry: one CUDA block processes TB x TB image blocks. The shared XYB
// region carries the two-pixel halo required by gaborish. The pre-erosion map is
// at 4x-subsampled resolution (CELLS x CELLS cells).
constexpr int HALO = 2;
constexpr int TB = 4;
constexpr int TILE_PX = TB * 8;
constexpr int PAD = TILE_PX + 2 * HALO;
constexpr int CELLS = 2 * TB;
constexpr int TILE_SAMPLES = TILE_PX * TILE_PX;
// The DCT horizontal pass has the lanes of a warp read different tile rows at
// the same column; the odd row stride maps them to distinct shared-memory banks.
constexpr int TILE_STRIDE = TILE_PX + 1;
constexpr int TILE_FLOATS = TILE_PX * TILE_STRIDE;
constexpr int PAD_SAMPLES = PAD * PAD;
constexpr int FRONTEND_THREADS = 256;
// The pre-erosion takes one cell row per warp and one tile column per lane.
static_assert(FRONTEND_THREADS == 32 * CELLS && TILE_PX == 32);
// The erosion takes four lanes per cell.
static_assert(FRONTEND_THREADS == 4 * CELLS * CELLS);
// The raw X and B planes stay resident from the load to the X/B gaborish at
// RESIDENT_X_B_OFFSET; the buffers local to each phase lie below them. With
// HALF2_FRONTEND they are one plane of half2 (X, B) pairs.
constexpr std::size_t LOAD_AQ_LOCAL_PEAK =
    TILE_FLOATS + PAD_SAMPLES + 2 * CELLS * CELLS + 6 * TB * TB;
// The selection's horizontal-pass rows of each candidate size N start at sharp
// Y, which they overlap, with vector v at v * (N + 1): the lanes of a warp
// reading consecutive vectors hit distinct banks. The selection words follow:
// the 8x8 and 16x16 costs, the 32x32 partial distortions and rates, and the
// region layout. Then the rate terms of the quantized magnitudes below
// RATE_TERMS (see accumulate_coeff_cost).
constexpr int SELECT_ROWS16_OFFSET = 32 * (32 + 1);
constexpr int SELECT_ROWS8_OFFSET = SELECT_ROWS16_OFFSET + 64 * (16 + 1);
constexpr int SELECT_WORDS_OFFSET = SELECT_ROWS8_OFFSET + 128 * (8 + 1);
constexpr std::size_t SELECT_WORDS = 16 + 4 + 2 * 2 + 1;
constexpr int RATE_TERMS = 128;
constexpr std::size_t RATE_TERMS_OFFSET = SELECT_WORDS_OFFSET + SELECT_WORDS;
constexpr std::size_t REGION_LOCAL_PEAK = RATE_TERMS_OFFSET + RATE_TERMS;
// The X/B horizontal-pass rows and coefficients, padded by region_row_index.
constexpr int REGION_ROWS_FLOATS = TILE_SAMPLES + TILE_SAMPLES / 32;
constexpr std::size_t RESIDENT_X_B_OFFSET = std::max(LOAD_AQ_LOCAL_PEAK, REGION_LOCAL_PEAK);
// The X/B pass holds sharp X and B below the raw planes, its horizontal-pass
// rows in the raw planes once gaborish has consumed them, and its coefficients
// in the sharp planes once the horizontal pass has consumed those.
// Warps that pair X and B lanes read distinct banks: the horizontal pass reads
// 16 consecutive rows per channel, so sharp B starts 16 banks after sharp X;
// the DCT16 vertical pass reads and writes 16 vectors per channel on 16 banks
// with a gap of 8 after each 8, so the B rows and coefficients start 8 (mod 16)
// banks after the X ones.
constexpr std::size_t SHARP_B_OFFSET = TILE_FLOATS + 16;
static_assert(SHARP_B_OFFSET % 32 == 16 && SHARP_B_OFFSET + TILE_FLOATS <= RESIDENT_X_B_OFFSET);
constexpr int ROWS_B_SKEW = 8;
static_assert((PAD_SAMPLES + ROWS_B_SKEW) % 16 == 8);
static_assert(REGION_ROWS_FLOATS + ROWS_B_SKEW <= PAD_SAMPLES);
constexpr std::size_t COEFFS_B_OFFSET = SHARP_B_OFFSET + 8;
static_assert(COEFFS_B_OFFSET % 16 == 8 && REGION_ROWS_FLOATS <= SHARP_B_OFFSET &&
              COEFFS_B_OFFSET + REGION_ROWS_FLOATS <= RESIDENT_X_B_OFFSET);
constexpr std::size_t WORKSPACE_FLOATS = RESIDENT_X_B_OFFSET + 2 * PAD_SAMPLES;
constexpr std::size_t WORKSPACE_BYTES = WORKSPACE_FLOATS * sizeof(float);
// Resident blocks per SM. sm_61 has 96 KB of shared memory per SM, and 4 blocks
// leave 64 registers per thread. sm_87 has 164 KB, minus 1 KB reserved per
// block, and 5 blocks leave 48 registers per thread.
constexpr PerArch<int> FRONTEND_BLOCKS_PER_SM{.default_entry = 4, .sm87 = 5};
static_assert(FRONTEND_BLOCKS_PER_SM.default_entry * WORKSPACE_BYTES <= 96 * 1024);
static_assert(FRONTEND_BLOCKS_PER_SM.sm87 * (WORKSPACE_BYTES + 1024) <= 164 * 1024);

// Loads the N floats at `src` and computes F of their 1D DCT frequencies (N
// times A, see dct_impl.cuh) into out[i], frequency i * N / F + part: all of
// them, or for F = N / 2 the even (part 0) or odd (part 1) ones.
template <int N, int F, int SIZE>
__device__ __forceinline__ void dct_vector(const float* __restrict__ src, int part,
                                           float (&out)[SIZE]) {
    static_assert((F == N || 2 * F == N) && F <= SIZE);
    float in[N];
#pragma unroll
    for (int t{0}; t < N; ++t) {
        in[t] = src[t];
    }
    float freqs[F];
    if constexpr (F == N) {
        dct1d(in);
#pragma unroll
        for (int i{0}; i < F; ++i) {
            freqs[i] = in[i];
        }
    } else if (part == 0) {
        dct1d_even(in, freqs);
    } else {
        dct1d_odd(in, freqs);
    }
#pragma unroll
    for (int i{0}; i < F; ++i) {
        out[i] = freqs[i];
    }
}

// Sum over aligned groups of G lanes, valid on every lane.
template <int G>
__device__ inline float lane_group_sum(float v) {
#pragma unroll
    for (int off{1}; off < G; off <<= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

// Full-mask warp sums of a, b and c as one reduce-scatter: the result is the sum
// of a on lane 0, of b on lane 8 and of c on lane 16. Each sum pairs lanes like
// a halving tree v[i] += v[i + s] for s = 16..1 (xor partners form the same
// pairs, and the adds are commutative and non-contractible), so callers can
// reproduce a shared-memory tree reduction bit for bit.
__device__ inline float warp_reduce_sum3(float a, float b, float c, int lane) {
    // At s = 16 the lower half keeps a and the upper half c, while b is summed
    // on every lane; at s = 8 the lanes with bit 3 set take over b.
    const bool upper{(lane & 16) != 0};
    float v{__fadd_rn(upper ? c : a, __shfl_xor_sync(0xffffffffu, upper ? a : c, 16))};
    b = __fadd_rn(b, __shfl_xor_sync(0xffffffffu, b, 16));
    const bool keeps_b{(lane & 8) != 0};
    v = __fadd_rn(keeps_b ? b : v, __shfl_xor_sync(0xffffffffu, keeps_b ? v : b, 8));
    for (int off{4}; off > 0; off >>= 1) {
        v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, off));
    }
    return v;
}

// Neighbour k (row-major in the 3x3 neighbourhood) of pre-erosion cell (cx, cy),
// clamped to the tile (the seam approximation).
__device__ inline float erosion_neighbour(const float* __restrict__ pre, int cx, int cy, int k) {
    const int x{min(max(cx + k % 3 - 1, 0), CELLS - 1)};
    const int y{min(max(cy + k / 3 - 1, 0), CELLS - 1)};
    return pre[y * CELLS + x];
}

// FuzzyErosion of all pre-erosion cells: the weighted sum of the four smallest
// values in each cell's 3x3 neighbourhood. Four lanes share a cell, each sorting
// up to three of the nine neighbours, and two butterfly merges give all four
// lanes the cell's four smallest values. These do not depend on how the
// neighbours are split, and the weighted sum keeps its order.
__device__ void erode_cells(const float* __restrict__ pre, float* __restrict__ eroded, int tid,
                            float km0, float km1, float km2, float km3) {
    const int cell{tid / 4};
    const int part{tid % 4};
    const int cx{cell % CELLS};
    const int cy{cell / CELLS};
    float m0{erosion_neighbour(pre, cx, cy, part)};
    float m1{erosion_neighbour(pre, cx, cy, part + 4)};
    float m2{part == 0 ? erosion_neighbour(pre, cx, cy, 8) : INFINITY};
    float m3{INFINITY};
    compare_swap(m0, m1);
    compare_swap(m1, m2);
    compare_swap(m0, m1);
    merge_min4_xor(m0, m1, m2, m3, 1);
    merge_min4_xor(m0, m1, m2, m3, 2);
    if (part == 0) {
        eroded[cell] = km0 * m0 + km1 * m1 + km2 * m2 + km3 * m3;
    }
}

// rate_terms[i] = 1 + log2f(1 + i) for 0 < i < RATE_TERMS, and 0 for i = 0.
__device__ inline void init_rate_terms(float* __restrict__ rate_terms, int tid) {
    if (tid < RATE_TERMS) {
        rate_terms[tid] = tid == 0 ? 0.0f : 1.0f + log2f(1.0f + static_cast<float>(tid));
    }
}

struct LumaCostWeight {
    DequantWeightPair weight;
    float threshold;
};

template<int N> struct LumaCostWeights { LumaCostWeight values[N*N]; };

template<int N>
constexpr LumaCostWeights<N> luma_cost_weights(const DequantWeightPairs<N*N>& weights) {
    LumaCostWeights<N> result{};
    for (int i=0; i<N*N; ++i) result.values[i] = {weights.w[1][i], ac_deadzone(N,1).threshold(i)};
    return result;
}

__device__ LumaCostWeights<8> costs8{luma_cost_weights<8>(DCT8_DEQUANT_PAIRS)};
__device__ LumaCostWeights<16> costs16{luma_cost_weights<16>(DCT16_DEQUANT_PAIRS)};
__device__ LumaCostWeights<32> costs32{luma_cost_weights<32>(DCT32_DEQUANT_PAIRS)};

// Rate-distortion contribution of one Y candidate coefficient quantized with the
// nearest DCT8 weight; the LLF corner is excluded.
__device__ inline void accumulate_coeff_cost(float coeff, int fx, int fy, int n, float qgsf,
                                             float inv_qgsf, const float* __restrict__ rate_terms,
                                             float& dist, float& rate) {
    const int m{n / 8};
    if (fx < m && fy < m) {
        return;
    }
    const int index = fx * n + fy;
    const auto entry = n == 8 ? costs8.values[index] : n == 16 ? costs16.values[index] : costs32.values[index];
    const DequantWeightPair w = entry.weight;
    const float steps = div_rn(coeff * qgsf, w.weight, w.inv_weight);
    const float q = fabsf(steps) < entry.threshold ? 0.0f : rintf(steps);
    const float err{div_rn(q * w.weight, qgsf, inv_qgsf) - coeff};
    dist += err * err;
    const float mag{fabsf(q)};
    if (mag < static_cast<float>(RATE_TERMS)) {
        rate += rate_terms[static_cast<int>(mag)];
    } else {
        rate += 1.0f + log2f(1.0f + mag);
    }
}

__device__ inline float candidate_cost(int n, float dist, float rate, float lambda) {
    return static_cast<float>(n * n) * dist + lambda * rate;
}

constexpr std::uint32_t WHOLE32_LAYOUT{0x20202020u};

__device__ inline int quadrant_side(std::uint32_t layout, int q) {
    return static_cast<int>((layout >> (8 * q)) & 0xffu);
}

struct alignas(16) Half8 {
    __half2 h[4];
};

struct alignas(8) Half4 {
    __half2 h[2];
};

__device__ inline void store_half8(__half* dst, const float* v) {
    *reinterpret_cast<Half8*>(dst) =
        Half8{{__floats2half2_rn(v[0], v[1]), __floats2half2_rn(v[2], v[3]),
               __floats2half2_rn(v[4], v[5]), __floats2half2_rn(v[6], v[7])}};
}

__device__ inline void store_half4(__half* dst, const float* v) {
    *reinterpret_cast<Half4*>(dst) =
        Half4{{__floats2half2_rn(v[0], v[1]), __floats2half2_rn(v[2], v[3])}};
}

// Orin and Ada use tensor cores for the 32x32 Y candidate and the X/B
// transforms, and half2 arithmetic for adaptive quantization and X/B gaborish.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 870 || __CUDA_ARCH__ == 890)
constexpr bool DCT_MMA{true};
constexpr bool HALF2_FRONTEND{true};
#else
constexpr bool DCT_MMA{false};
constexpr bool HALF2_FRONTEND{false};
#endif

// Not const: a const table is placed in the constant bank, whose loads
// serialize over the lanes' distinct addresses.
[[maybe_unused]] __device__ Dct32MmaBasis DCT32_BASIS{DCT32_MMA_BASIS};
// The region transform bases of DCT16 (0) and DCT8 (1) quadrants.
[[maybe_unused]] __device__ RegionMmaBasis REGION_BASES[2]{REGION_MMA_BASIS_DCT16,
                                                           REGION_MMA_BASIS_DCT8};

// Coefficient (fx, fy) held in c[m] by lane `lane` of 32x32 warp `part` after
// the vertical pass.
__device__ inline int dct32_lane_fx(int lane, int m) {
    return DCT_MMA ? dct32_mma_fx(lane, m) : lane;
}

__device__ inline int dct32_lane_fy(int part, int lane, int m) {
    return DCT_MMA ? dct32_mma_fy(part, lane, m) : 2 * m + part;
}

// Each warp takes one candidate size: warps 0-3 the 8x8 candidates of tile
// column block w, warps 4-5 the 16x16 candidates of tile column half w - 4,
// warps 6-7 the even and odd frequencies of the 32x32 candidate. A lane
// transforms one N-vector per pass: a tile row, then the rows of one fx. With
// DCT_MMA, warps 6-7 instead compute coefficient rows fy in [0, 16) and
// [16, 32): the vertical pass in the horizontal-pass slot, the horizontal pass
// in the vertical-pass slot.
//
// Returns the region layout: byte q holds the side (8, 16 or 32) of the
// transforms covering quadrant q, or 0 if the quadrant lies outside the image.
// Only the chosen transforms' Y coefficients are stored to y_plane. Inlined, it
// keeps the kernel's range of tid, so the lane and warp arithmetic stays
// unsigned.
__device__ __forceinline__ std::uint32_t select_transforms(float* __restrict__ select,
                                                           std::size_t rbx, std::size_t rby,
                                                           std::size_t bw, int tile_bw, int tile_bh,
                                                           float qgsf, float lambda,
                                                           std::int8_t* __restrict__ acs,
                                                           __half* __restrict__ y_plane, int tid) {
    const int lane = tid % 32;
    const int column = tid / 32;
    float* rows = select + SELECT_ROWS8_OFFSET;
    if (column < 4) {
        float values[8];
        dct_vector<8,8>(select + lane*TILE_STRIDE + column*8, 0, values);
        float* dst = rows + (4*column + lane/8)*8*9 + lane%8;
#pragma unroll
        for(int f=0; f<8; ++f) dst[f*9]=values[f];
    }
    __syncthreads();
    if(column < tile_bw && lane/8 < tile_bh) {
        float values[8];
        dct_vector<8,8>(rows + (32*column+lane)*9, 0, values);
#pragma unroll
        for(int f=0; f<8; ++f) values[f]*=DCT2D_SCALE<8>;
        if(lane%8==0) acs[(rby+lane/8)*bw+rbx+column]=8;
        store_half8(y_plane+covered_plane_slot(8,rbx+column,rby+lane/8,bw,8*(lane%8)), values);
    }
    return 0x08080808u;
}

// Shared-memory index of region entry k (a horizontal-pass row entry or a
// coefficient): one padding float after every 32 entries. A pass's N-vectors
// stay contiguous, and the lanes of a warp reading 32 consecutive vectors at
// the same position hit distinct banks.
constexpr int region_row_index(int k) {
    return k + k / 32;
}

// Entry k of a region's X or B rows and coefficients: the transforms in
// quadrant order, so quadrant q holds k in [256 q, 256 q + 256) unless the
// region is one DCT32, DCT8 sub-block (dx, dy) of a quadrant holds the 64
// entries from 64 (2 dy + dx), and entry fx * N + i of a transform is its
// frequency fx on row i (rows) or its coefficient (fx, i).
//
// Both passes give each warp one task. A whole DCT32 gives warp w < 4 channel
// w / 2 and the even or odd frequencies (w % 2), one vector per lane. Otherwise
// warp w takes quadrant w / 2: a DCT16 puts X on lanes 0-15 and B on 16-31
// with the even or odd frequencies (w % 2); the DCT8s put column w % 2 (rows)
// or channel w % 2 (coefficients) on all lanes.

// Entry of coefficient (fx, fy) of the side-n transforms of a quadrant, relative
// to the quadrant's first entry: for DCT8s, frequency (fx % 8, fy % 8) of block
// (fx / 8, fy / 8).
constexpr int quadrant_entry(int n, int fx, int fy) {
    return n == 16 ? 16 * fx + fy : 64 * (2 * (fy / 8) + fx / 8) + 8 * (fx % 8) + fy % 8;
}

// The X (0) and B (1) buffers of the X/B pass. Selecting a channel per lane
// from an array of pointers would place the array on the stack, and the
// accesses through the loaded pointers would lose their address space.
template <typename T>
struct ChannelPair {
    T* x;
    T* b;
    __device__ T* operator[](int c) const { return c == 0 ? x : b; }
};

// Horizontal pass of the X and B transforms of the region into rows.
[[maybe_unused]] __device__ __forceinline__ void region_rows(ChannelPair<const float> sharp,
                                                             ChannelPair<float> rows,
                                                             std::uint32_t layout, int tile_bw,
                                                             int tile_bh, int warp, int lane) {
    if (layout == WHOLE32_LAYOUT) {
        if (warp < 4) {
            const int c{warp / 2};
            const int part{warp % 2};
            float out[16];
            dct_vector<32, 16>(sharp[c] + lane * TILE_STRIDE, part, out);
#pragma unroll
            for (int m{0}; m < 16; ++m) {
                rows[c][region_row_index((2 * m + part) * 32 + lane)] = out[m];
            }
        }
        return;
    }
    const int q{warp / 2};
    const int sub{warp % 2};
    const int n{quadrant_side(layout, q)};
    const int c{lane / 16};
    const int r{lane % 16};
    const float* src{sharp[c] + (16 * (q / 2) + r) * TILE_STRIDE + 16 * (q % 2)};
    if (n == 16) {
        float out[8];
        dct_vector<16, 8>(src, sub, out);
#pragma unroll
        for (int m{0}; m < 8; ++m) {
            rows[c][region_row_index(256 * q + (2 * m + sub) * 16 + r)] = out[m];
        }
    } else if (n == 8) {
        if (2 * (q % 2) + sub < tile_bw && 2 * (q / 2) + r / 8 < tile_bh) {
            float out[8];
            dct_vector<8, 8>(src + 8 * sub, 0, out);
            const int k0{256 * q + 64 * (2 * (r / 8) + sub) + r % 8};
#pragma unroll
            for (int f{0}; f < 8; ++f) {
                rows[c][region_row_index(k0 + 8 * f)] = out[f];
            }
        }
    }
}

// Vertical pass of the X and B transforms of the region from rows into
// coefficients.
[[maybe_unused]] __device__ __forceinline__ void region_coeffs(ChannelPair<const float> rows,
                                                               std::uint32_t layout, int tile_bw,
                                                               int tile_bh,
                                                               ChannelPair<float> coeffs, int warp,
                                                               int lane) {
    if (layout == WHOLE32_LAYOUT) {
        if (warp < 4) {
            const int c{warp / 2};
            const int part{warp % 2};
            float out[16];
            dct_vector<32, 16>(rows[c] + region_row_index(32 * lane), part, out);
#pragma unroll
            for (int m{0}; m < 16; ++m) {
                coeffs[c][region_row_index(32 * lane + 2 * m + part)] = out[m] * DCT2D_SCALE<32>;
            }
        }
        return;
    }
    const int q{warp / 2};
    const int sub{warp % 2};
    const int n{quadrant_side(layout, q)};
    if (n == 16) {
        const int c{lane / 16};
        const int k0{256 * q + 16 * (lane % 16)};
        float out[8];
        dct_vector<16, 8>(rows[c] + region_row_index(k0), sub, out);
#pragma unroll
        for (int m{0}; m < 8; ++m) {
            coeffs[c][region_row_index(k0 + 2 * m + sub)] = out[m] * DCT2D_SCALE<16>;
        }
    } else if (n == 8) {
        const int block{lane / 8};
        if (2 * (q % 2) + block % 2 < tile_bw && 2 * (q / 2) + block / 2 < tile_bh) {
            const int k0{256 * q + 8 * lane};
            float out[8];
            dct_vector<8, 8>(rows[sub] + region_row_index(k0), 0, out);
#pragma unroll
            for (int f{0}; f < 8; ++f) {
                coeffs[sub][region_row_index(k0 + f)] = out[f] * DCT2D_SCALE<8>;
            }
        }
    }
}

// Shared-memory index of y[i] of lane `lane` in the tensor-core X/B pass: of
// warp half `part` of a whole DCT32, or of side n in quadrant q.
constexpr int whole32_mma_index(int part, int lane, int i) {
    return region_row_index(32 * dct32_mma_fx(lane, i) + dct32_mma_fy(part, lane, i));
}

template <int N>
constexpr int quadrant_mma_index(int q, int lane, int i) {
    return region_row_index(
        256 * q + quadrant_entry(N, region_mma_fx(N, lane, i), region_mma_fy(N, lane, i)));
}

// Whether index(a, lane, i) = index(a, lane, 0) + index(0, 0, i) for every
// a < A, lane and i < I: a lane's stores then take one base address and
// immediate offsets.
template <int A, int I, typename Index>
constexpr bool splits_into_base_and_offsets(Index index) {
    for (int a{0}; a < A; ++a) {
        for (int lane{0}; lane < 32; ++lane) {
            for (int i{0}; i < I; ++i) {
                if (index(a, lane, i) != index(a, lane, 0) + index(0, 0, i)) {
                    return false;
                }
            }
        }
    }
    return true;
}

static_assert(splits_into_base_and_offsets<2, 16>(whole32_mma_index));
static_assert(splits_into_base_and_offsets<4, 8>(quadrant_mma_index<16>));
static_assert(splits_into_base_and_offsets<4, 8>(quadrant_mma_index<8>));

template <int N>
__device__ __forceinline__ void store_quadrant_mma(float* __restrict__ coeffs, int q, int lane,
                                                   const float (&y)[8]) {
    float* dst{coeffs + quadrant_mma_index<N>(q, lane, 0)};
#pragma unroll
    for (int i{0}; i < 8; ++i) {
        dst[quadrant_mma_index<N>(0, 0, i)] = y[i];
    }
}

// The X and B transforms of the region on the tensor cores, from the sharp
// planes into coefficients. A whole DCT32 gives warp w < 4 channel w / 2 and
// coefficient rows fy in [16 (w % 2), 16 (w % 2) + 16); otherwise warp w takes
// quadrant w / 2 and channel w % 2.
[[maybe_unused]] __device__ __forceinline__ void region_coeffs_mma(ChannelPair<const float> sharp,
                                                                   std::uint32_t layout,
                                                                   ChannelPair<float> coeffs,
                                                                   int warp, int lane) {
    if (layout == WHOLE32_LAYOUT) {
        if (warp < 4) {
            const int c{warp / 2};
            const int part{warp % 2};
            std::uint32_t u[2][4];
            const float offset{
                dct32_mma_vertical(DCT32_BASIS, sharp[c], TILE_STRIDE, part, lane, u)};
            float y[16];
            dct32_mma_horizontal(DCT32_BASIS, u, part, lane, offset, y);
            float* dst{coeffs[c] + whole32_mma_index(part, lane, 0)};
#pragma unroll
            for (int i{0}; i < 16; ++i) {
                dst[whole32_mma_index(0, 0, i)] = y[i];
            }
        }
        return;
    }
    const int q{warp / 2};
    const int c{warp % 2};
    const int n{quadrant_side(layout, q)};
    if (n == 0) {
        return;
    }
    float y[8];
    region_mma(REGION_BASES[n == 16 ? 0 : 1], n,
               sharp[c] + 16 * (q / 2) * TILE_STRIDE + 16 * (q % 2), TILE_STRIDE, lane, y);
    if (n == 16) {
        store_quadrant_mma<16>(coeffs[c], q, lane, y);
    } else {
        store_quadrant_mma<8>(coeffs[c], q, lane, y);
    }
}

// Stores the region's X and B coefficients to the covered-block planes, eight
// consecutive entries per thread: channel tid / 128, entries from
// 8 (tid % 128).
__device__ __forceinline__ void store_region_coeffs(ChannelPair<const float> coeffs,
                                                    std::uint32_t layout, std::size_t rbx,
                                                    std::size_t rby, std::size_t bw, int tile_bw,
                                                    int tile_bh, ChannelPair<__half> dst, int tid) {
    const int c{tid / 128};
    const int k{8 * (tid % 128)};
    const float* src{coeffs[c] + region_row_index(k)};
    if (layout == WHOLE32_LAYOUT) {
        store_half8(dst[c] + covered_plane_slot(32, rbx, rby, bw, static_cast<std::size_t>(k)),
                    src);
        return;
    }
    const int q{k / 256};
    const int n{quadrant_side(layout, q)};
    const std::size_t qbx{rbx + 2 * static_cast<std::size_t>(q % 2)};
    const std::size_t qby{rby + 2 * static_cast<std::size_t>(q / 2)};
    if (n == 16) {
        store_half8(
            dst[c] + covered_plane_slot(16, qbx, qby, bw, static_cast<std::size_t>(k % 256)), src);
    } else if (n == 8) {
        const int block{(k % 256) / 64};
        if (2 * (q % 2) + block % 2 < tile_bw && 2 * (q / 2) + block / 2 < tile_bh) {
            store_half8(dst[c] + covered_plane_slot(8, qbx + static_cast<std::size_t>(block % 2),
                                                    qby + static_cast<std::size_t>(block / 2), bw,
                                                    static_cast<std::size_t>(k % 64)),
                        src);
        }
    }
}

// The adaptive field histogram bins a value by its float exponent and top
// mantissa bits, 2^(23 - FIELD_BIN_SHIFT) bins per octave from FIELD_MIN up to
// FIELD_MAX; values outside the range fall into the end bins.
constexpr std::uint32_t FIELD_BIN_SHIFT{16};
constexpr float FIELD_MIN{1.0f / 65536.0f};
constexpr float FIELD_MAX{65536.0f};
constexpr std::uint32_t FIELD_BIN_FIRST{(127u - 16u) << (23u - FIELD_BIN_SHIFT)};
static_assert(ADAPTIVE_FIELD_BINS == 32u << (23u - FIELD_BIN_SHIFT));

__device__ inline std::uint32_t field_bin(float qf) {
    const std::int32_t bin{static_cast<std::int32_t>(__float_as_uint(qf) >> FIELD_BIN_SHIFT) -
                           static_cast<std::int32_t>(FIELD_BIN_FIRST)};
    return bin < 0 ? 0u : min(static_cast<std::uint32_t>(bin), ADAPTIVE_FIELD_BINS - 1);
}

// The number of field values below `v` in the histogram's cumulative counts
// `below` (below[b] values in the bins before b), linear within a bin in the
// float representation.
__device__ inline float field_values_below(const std::uint32_t* below, float v) {
    if (!(v > FIELD_MIN)) {
        return 0.0f;
    }
    const std::uint32_t bits{__float_as_uint(v)};
    const std::uint32_t bin{(bits >> FIELD_BIN_SHIFT) - FIELD_BIN_FIRST};
    if (bin >= ADAPTIVE_FIELD_BINS) {
        return static_cast<float>(below[ADAPTIVE_FIELD_BINS]);
    }
    constexpr std::uint32_t FRACTION_MASK{(1u << FIELD_BIN_SHIFT) - 1};
    const float fraction{static_cast<float>(bits & FRACTION_MASK) *
                         (1.0f / static_cast<float>(1u << FIELD_BIN_SHIFT))};
    return static_cast<float>(below[bin]) +
           static_cast<float>(below[bin + 1] - below[bin]) * fraction;
}

constexpr std::uint32_t CALIBRATE_THREADS{1024};
static_assert(ADAPTIVE_FIELD_BINS == 4 * CALIBRATE_THREADS);

// The smallest positive float in (lo, hi], given by their representations, at
// which the monotone `reaches` holds, to one representation step; `reaches`
// must hold at hi. Called by the whole block with block-uniform arguments.
template <typename Reaches>
__device__ float first_reaching(std::uint32_t lo, std::uint32_t hi, Reaches reaches) {
    while (hi - lo > 1) {
        const std::uint32_t step{(hi - lo + CALIBRATE_THREADS - 1) / CALIBRATE_THREADS};
        const std::uint32_t x{min(hi, lo + (threadIdx.x + 1) * step)};
        const std::uint32_t failing{
            static_cast<std::uint32_t>(__syncthreads_count(!reaches(__uint_as_float(x))))};
        lo += failing * step;
        hi = min(hi, lo + step);
    }
    return __uint_as_float(hi);
}

// The median and median absolute deviation of the adaptive field, estimated from
// its histogram, and the calibration they give. libjxl takes the order
// statistics of rank count / 2 exactly; the estimates are within a bin of them.
__global__ void __launch_bounds__(CALIBRATE_THREADS)
    calibrate_field_kernel(const std::uint32_t* __restrict__ field_histogram, float quant_dc_float,
                           QuantCalibration* __restrict__ calibration) {
    __shared__ std::uint32_t below[ADAPTIVE_FIELD_BINS + 1];
    __shared__ std::uint32_t warp_sums[CALIBRATE_THREADS / 32];
    const std::uint32_t tid{threadIdx.x};
    const std::uint32_t lane{tid % 32};
    const std::uint32_t warp{tid / 32};

    const uint4 counts{reinterpret_cast<const uint4*>(field_histogram)[tid]};
    const std::uint32_t sum{counts.x + counts.y + counts.z + counts.w};
    const std::uint32_t inclusive{warp_inclusive_sum(sum, lane)};
    if (lane == 31) {
        warp_sums[warp] = inclusive;
    }
    __syncthreads();
    if (warp == 0) {
        warp_sums[lane] = warp_inclusive_sum(warp_sums[lane], lane);
    }
    __syncthreads();
    const std::uint32_t first{(warp == 0 ? 0u : warp_sums[warp - 1]) + inclusive - sum};
    below[4 * tid] = first;
    below[4 * tid + 1] = first + counts.x;
    below[4 * tid + 2] = first + counts.x + counts.y;
    below[4 * tid + 3] = first + counts.x + counts.y + counts.z;
    if (tid == CALIBRATE_THREADS - 1) {
        below[ADAPTIVE_FIELD_BINS] = first + sum;
    }
    __syncthreads();

    const float rank{static_cast<float>(below[ADAPTIVE_FIELD_BINS] / 2) + 0.5f};
    const float median{first_reaching(0u, __float_as_uint(FIELD_MAX), [&](float v) {
        return field_values_below(below, v) >= rank;
    })};
    const float deviation{first_reaching(0u, __float_as_uint(2.0f * FIELD_MAX), [&](float t) {
        return field_values_below(below, median + t) - field_values_below(below, median - t) >=
               rank;
    })};
    if (tid == 0) {
        *calibration = calibrate_quant(quant_dc_float, median, deviation);
    }
}

constexpr std::uint32_t FIELD_THREADS{256};

__global__ void __launch_bounds__(FIELD_THREADS)
    quantize_field_kernel(const float* __restrict__ adaptive_field, std::uint32_t blocks,
                          const QuantCalibration* __restrict__ calibration,
                          std::int32_t* __restrict__ quant_field) {
    const std::uint32_t i{blockIdx.x * FIELD_THREADS + threadIdx.x};
    if (i < blocks) {
        quant_field[i] = quant_field_value(adaptive_field[i], calibration->inv_global_scale);
    }
}

struct Nv12View {
    cudaTextureObject_t luma, chroma;
};
struct Rgb8View {
    const std::uint8_t* pixels;
    std::size_t pitch;
};
// Match centered horizontal 4:2:2 chroma sampling before the RGB-to-XYB transform.
__device__ void rgb422_to_xyb(Rgb8View image, int x, int y, int width,
                              float& out_x, float& out_y, float& out_b) {
    const auto* row = reinterpret_cast<const uchar3*>(image.pixels + size_t(y) * image.pitch);
    const auto sample = [&](int column) { return row[min(max(column, 0), width - 1)]; };
    const auto pixel=sample(x);
    const int yy=(19595*pixel.x+38470*pixel.y+7471*pixel.z+32768)>>16;
    const int pair=(x&~15)+2*(x&7);
    const auto cb=[](uchar3 v){return (32768*v.z-11059*v.x-21709*v.y+8421375)>>16;};
    const auto cr=[](uchar3 v){return (32768*v.x-27439*v.y-5329*v.z+8421375)>>16;};
    const auto a=sample(pair), b=sample(pair+1);
    const int bias=1;
    out_x=(x&8)?0.f:(((cb(a)+cb(b)+bias)/2)-128)*(1.f/255.f);
    out_b=(x&8)?0.f:(((cr(a)+cr(b)+bias)/2)-128)*(1.f/255.f);
    out_y=(yy-128)*(1.f/255.f);
}

template <typename View>
__global__ void __launch_bounds__(FRONTEND_THREADS, FRONTEND_BLOCKS_PER_SM[COMPILED_ARCH])
    frontend_fused_kernel(View image, std::size_t width, std::size_t height, std::size_t bw,
                          std::size_t bh, float km0, float km1, float km2, float km3, float mul_pb,
                          float add_pb, float qgsf, __half* __restrict__ coeffs,
                          std::int8_t* __restrict__ acs, float* __restrict__ adaptive_field,
                          std::uint32_t* __restrict__ field_histogram) {
    extern __shared__ __align__(16) unsigned char workspace_bytes[];
    float* workspace{reinterpret_cast<float*>(workspace_bytes)};
    float* sharp_y{workspace};
    float* raw_x{workspace + RESIDENT_X_B_OFFSET};
    float* raw_b{raw_x + PAD_SAMPLES};
    __half2* raw_xb{reinterpret_cast<__half2*>(raw_x)};

    const std::size_t tbx0{static_cast<std::size_t>(blockIdx.x) * TB};
    const std::size_t tby0{static_cast<std::size_t>(blockIdx.y) * TB};
    // The tile's block columns and rows that lie inside the image.
    const int tile_bw{static_cast<int>(min(bw - tbx0, std::size_t{TB}))};
    const int tile_bh{static_cast<int>(min(bh - tby0, std::size_t{TB}))};
    const int tid{static_cast<int>(threadIdx.x)};

    // With fixed DCT8 and no Gaborish filter, only the tile's own pixels are needed.
    const int lane = tid % TILE_PX;
    const int warp = tid / TILE_PX;
    const int gx = static_cast<int>(tbx0 * 8) + lane;
    const int gy0 = static_cast<int>(tby0 * 8) + warp;
    const float chroma_x_max{fmaf(static_cast<float>(width - 1), 0.5f, 0.25f)};
    const float chroma_y_max{fmaf(static_cast<float>(height - 1), 0.5f, 0.25f)};
#pragma unroll
    for (int i = tid; i < TILE_SAMPLES; i += FRONTEND_THREADS) {
        const int gy = gy0 + (i - tid) / TILE_PX;
        float xo{}, yo{}, bo{};
        if constexpr (std::is_same_v<View, Rgb8View>) {
            int x = min(max(int(gx), 0), int(width) - 1), y = min(max(int(gy), 0), int(height) - 1);
            rgb422_to_xyb(image, x, y, int(width), xo, yo, bo);
        } else {
            // Point sampling at an integral coordinate reads that texel.
            const float yv{tex2D<float>(image.luma, gx, gy)};
            // The pixel centre in chroma texels, (g + 0.5) / 2 in one rounding.
            const float2 c{tex2D<float2>(image.chroma, fminf(fmaf(gx, 0.5f, 0.25f), chroma_x_max),
                                         fminf(fmaf(gy, 0.5f, 0.25f), chroma_y_max))};
            nv12_pixel_to_xyb(yv, c.x, c.y, xo, yo, bo);
        }
        sharp_y[(i / TILE_PX) * TILE_STRIDE + lane] = yo;
        if constexpr (HALF2_FRONTEND) {
            raw_xb[i] = __floats2half2_rn(xo, bo);
        } else {
            raw_x[i] = xo;
            raw_b[i] = bo;
        }
    }
    if (adaptive_field != nullptr && tid < TB * TB && tid % TB < tile_bw && tid / TB < tile_bh) {
        adaptive_field[(tby0 + tid / TB) * bw + tbx0 + tid % TB] = mul_pb;
    }
    if (adaptive_field != nullptr && blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
        field_histogram[field_bin(mul_pb)] = static_cast<std::uint32_t>(bw * bh);
    }
    __syncthreads();

    const std::size_t cplane{bw * bh * COEFFS_PER_BLOCK};
    const std::uint32_t layout{select_transforms(workspace, tbx0, tby0, bw, tile_bw, tile_bh, qgsf,
                                                 TRANSFORM_SELECT_LAMBDA, acs, coeffs + cplane,
                                                 tid)};
    // Sharp B overlaps the selection rows.
    __syncthreads();

    float* sharp_x{workspace};
    float* sharp_b{workspace + SHARP_B_OFFSET};
#pragma unroll
    for (int i = tid; i < TILE_SAMPLES; i += FRONTEND_THREADS) {
        const int index = (i / TILE_PX) * TILE_STRIDE + lane;
        if constexpr (HALF2_FRONTEND) {
            const float2 pair = __half22float2(raw_xb[i]);
            sharp_x[index] = pair.x;
            sharp_b[index] = pair.y;
        } else {
            sharp_x[index] = raw_x[i];
            sharp_b[index] = raw_b[i];
        }
    }
    __syncthreads();
    if constexpr (DCT_MMA) {
        // The coefficients go to the raw planes, which gaborish has consumed.
        region_coeffs_mma({sharp_x, sharp_b}, layout, {raw_x, raw_b}, warp, lane);
        __syncthreads();
        store_region_coeffs({raw_x, raw_b}, layout, tbx0, tby0, bw, tile_bw, tile_bh,
                            {coeffs, coeffs + 2 * cplane}, tid);
    } else {
        float* rows_b{raw_b + ROWS_B_SKEW};
        float* coeffs_b{workspace + COEFFS_B_OFFSET};
        region_rows({sharp_x, sharp_b}, {raw_x, rows_b}, layout, tile_bw, tile_bh, warp, lane);
        __syncthreads();
        region_coeffs({raw_x, rows_b}, layout, tile_bw, tile_bh, {sharp_x, coeffs_b}, warp, lane);
        __syncthreads();
        store_region_coeffs({sharp_x, coeffs_b}, layout, tbx0, tby0, bw, tile_bw, tile_bh,
                            {coeffs, coeffs + 2 * cplane}, tid);
    }
}

}  // namespace

Nv12Textures::Nv12Textures(const std::uint8_t* luma, std::size_t luma_pitch,
                           const std::uint8_t* chroma, std::size_t chroma_pitch, std::size_t width,
                           std::size_t height, cudaStream_t stream)
    : stream_{stream} {
    cudaResourceDesc luma_resource{};
    luma_resource.resType = cudaResourceTypePitch2D;
    luma_resource.res.pitch2D.devPtr = const_cast<std::uint8_t*>(luma);
    luma_resource.res.pitch2D.desc = cudaCreateChannelDesc<unsigned char>();
    luma_resource.res.pitch2D.width = width;
    luma_resource.res.pitch2D.height = height;
    luma_resource.res.pitch2D.pitchInBytes = luma_pitch;

    cudaTextureDesc luma_texture{};
    luma_texture.addressMode[0] = cudaAddressModeClamp;
    luma_texture.addressMode[1] = cudaAddressModeClamp;
    luma_texture.filterMode = cudaFilterModePoint;
    luma_texture.readMode = cudaReadModeNormalizedFloat;
    luma_texture.normalizedCoords = 0;

    cudaResourceDesc chroma_resource{};
    chroma_resource.resType = cudaResourceTypePitch2D;
    chroma_resource.res.pitch2D.devPtr = const_cast<std::uint8_t*>(chroma);
    chroma_resource.res.pitch2D.desc = cudaCreateChannelDesc<uchar2>();
    chroma_resource.res.pitch2D.width = (width + 1) / 2;
    chroma_resource.res.pitch2D.height = (height + 1) / 2;
    chroma_resource.res.pitch2D.pitchInBytes = chroma_pitch;

    cudaTextureDesc chroma_texture{};
    chroma_texture.addressMode[0] = cudaAddressModeClamp;
    chroma_texture.addressMode[1] = cudaAddressModeClamp;
    chroma_texture.filterMode = cudaFilterModeLinear;
    chroma_texture.readMode = cudaReadModeNormalizedFloat;
    chroma_texture.normalizedCoords = 0;

    if (cudaEventCreateWithFlags(&released_, cudaEventDisableTiming | cudaEventBlockingSync) !=
        cudaSuccess) {
        released_ = nullptr;
        return;
    }
    if (cudaCreateTextureObject(&luma_, &luma_resource, &luma_texture, nullptr) != cudaSuccess) {
        return;
    }
    if (cudaCreateTextureObject(&chroma_, &chroma_resource, &chroma_texture, nullptr) !=
        cudaSuccess) {
        cudaDestroyTextureObject(luma_);
        return;
    }
    valid_ = true;
}

Nv12Textures::~Nv12Textures() {
    if (valid_) {
        cudaEventRecord(released_, stream_);
        cudaEventSynchronize(released_);
        cudaDestroyTextureObject(chroma_);
        cudaDestroyTextureObject(luma_);
    }
    if (released_ != nullptr) {
        cudaEventDestroy(released_);
    }
}

template <typename View>
bool encode_view(View image, std::size_t width, std::size_t height, float distance, __half* coeffs,
                 std::int8_t* acs, float* adaptive_field, std::uint32_t* field_histogram,
                 std::int32_t* quant_field, QuantCalibration* calibration, cudaStream_t stream, bool initialize_field) {
    const std::size_t bw{padded_to_blocks(width) / 8};
    const std::size_t bh{padded_to_blocks(height) / 8};
    const std::uint32_t blocks{static_cast<std::uint32_t>(bw * bh)};

    const float scale{K_AC_QUANT / distance};
    const float quant_dc_float{initial_quant_dc(distance)};
    // The transform selection runs before the adaptive field exists, at the
    // quantizer of a uniform field of the distance's AC quantizer.
    const QuantCalibration uniform{calibrate_quant(quant_dc_float, scale, 0.0f)};
    const float qgsf{static_cast<float>(quant_field_value(scale, uniform.inv_global_scale)) *
                     uniform.global_scale_float};

    // PerBlockModulations exponent -> multiplicative field mapping.
    const float base_level{0.48f * scale};
    float dampen{1.0f};
    if (distance >= 2.0f) {
        dampen = 1.0f - (distance - 2.0f) / (14.0f - 2.0f);
        if (dampen < 0.0f) {
            dampen = 0.0f;
        }
    }
    const float mul_pb{scale * dampen};
    const float add_pb{(1.0f - dampen) * base_level};

    // FuzzyErosion weights (butteraugli-target dependent), normalized to kTotal.
    const float efm{distance < 2.0f ? (2.0f - distance) * 0.5f : 0.0f};
    float km0{0.125f};
    float km1{0.10f - efm * 0.10f};
    float km2{0.09f - efm * 0.09f};
    float km3{0.06f - efm * 0.06f};
    const float norm{0.29959705784054957f / (km0 + km1 + km2 + km3)};
    km0 *= norm;
    km1 *= norm;
    km2 *= norm;
    km3 *= norm;

    if (initialize_field &&
        cudaMemsetAsync(field_histogram, 0, ADAPTIVE_FIELD_BINS * sizeof(std::uint32_t), stream) !=
            cudaSuccess) {
        return false;
    }
    const dim3 block{FRONTEND_THREADS};
    const dim3 grid{static_cast<unsigned int>((bw + TB - 1) / TB),
                    static_cast<unsigned int>((bh + TB - 1) / TB)};
    frontend_fused_kernel<<<grid, block, WORKSPACE_BYTES, stream>>>(
        image, width, height, bw, bh, km0, km1, km2, km3, mul_pb, add_pb, qgsf, coeffs, acs,
        initialize_field ? adaptive_field : nullptr, field_histogram);
    if (!initialize_field) {
        return cudaGetLastError() == cudaSuccess;
    }
    calibrate_field_kernel<<<1, CALIBRATE_THREADS, 0, stream>>>(field_histogram, quant_dc_float,
                                                                calibration);
    quantize_field_kernel<<<(blocks + FIELD_THREADS - 1) / FIELD_THREADS, FIELD_THREADS, 0,
                            stream>>>(adaptive_field, blocks, calibration, quant_field);
    return cudaGetLastError() == cudaSuccess;
}

bool encode_frontend(const Nv12Textures& image, std::size_t width, std::size_t height,
                     float distance, __half* coeffs, std::int8_t* acs, float* adaptive_field,
                     std::uint32_t* field_histogram, std::int32_t* quant_field,
                     QuantCalibration* calibration, cudaStream_t stream, bool initialize_field) {
    return image.valid() &&
           encode_view(Nv12View{image.luma(), image.chroma()}, width, height, distance, coeffs, acs,
                       adaptive_field, field_histogram, quant_field, calibration, stream,
                       initialize_field);
}
bool encode_rgb_frontend(const std::uint8_t* rgb, std::size_t pitch, std::size_t width,
                         std::size_t height, float distance, __half* coeffs, std::int8_t* acs,
                         float* adaptive_field, std::uint32_t* field_histogram,
                         std::int32_t* quant_field, QuantCalibration* calibration,
                         cudaStream_t stream, bool initialize_field) {
    return encode_view(Rgb8View{rgb, pitch}, width, height, distance, coeffs, acs, adaptive_field,
                       field_histogram, quant_field, calibration, stream, initialize_field);
}
}  // namespace cujpegxl
