#include "src/trained_block_contexts.h"
// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "frontend_quantize.h"

#include <cmath>
#include <cstddef>
#include <limits>

#include <cuda_runtime.h>

#include "cfl.h"
#include "coeff_order.h"
#include "dequant_weight_pairs.h"
#include "exact_div.cuh"
#include "quant_calibration.h"
#include "round_to_int.h"
#include "vardct_layout.h"

namespace cujpegxl {
namespace {

std::size_t ceil_div(std::size_t a, std::size_t b) {
    return (a + b - 1) / b;
}

// The per-coefficient quantization tables hold the coefficients of the DCT8,
// DCT16 and DCT32 transforms back to back.
constexpr std::uint32_t DCT16_ENTRY_OFFSET = 64;
constexpr std::uint32_t DCT32_ENTRY_OFFSET = DCT16_ENTRY_OFFSET + 256;
constexpr std::uint32_t TABLE_ENTRIES = DCT32_ENTRY_OFFSET + 1024;

__host__ __device__ constexpr std::uint32_t table_entry(std::uint32_t side, std::uint32_t raw) {
    return (side == 32 ? DCT32_ENTRY_OFFSET : (side == 16 ? DCT16_ENTRY_OFFSET : 0u)) + raw;
}

struct alignas(sizeof(float4)) WeightPairTable {
    DequantWeightPair w[3][TABLE_ENTRIES];
};

template <std::size_t COEFFS>
constexpr void copy_weight_pairs(const DequantWeightPairs<COEFFS>& pairs, std::uint32_t side,
                                 WeightPairTable* table) {
    for (std::size_t c{0}; c < 3; ++c) {
        for (std::uint32_t raw{0}; raw < COEFFS; ++raw) {
            table->w[c][table_entry(side, raw)] = pairs.w[c][raw];
        }
    }
}

constexpr WeightPairTable make_weight_pair_table() {
    WeightPairTable table{};
    copy_weight_pairs(DCT8_DEQUANT_PAIRS, 8, &table);
    copy_weight_pairs(DCT16_DEQUANT_PAIRS, 16, &table);
    copy_weight_pairs(DCT32_DEQUANT_PAIRS, 32, &table);
    return table;
}

// The dead-zone threshold of each coefficient, [0] for X and B (which share
// it) and [1] for Y. A coefficient of the low-frequency corner (fx, fy < N/8),
// carried as DC, has an infinite threshold: it quantizes to 0, with a zero Y
// roundtrip.
struct alignas(sizeof(float2)) ThresholdTable {
    float t[2][TABLE_ENTRIES];
};

constexpr ThresholdTable make_threshold_table() {
    ThresholdTable table{};
    for (const std::uint32_t side : {8u, 16u, 32u}) {
        for (std::uint32_t raw{0}; raw < side * side; ++raw) {
            const bool llf{raw / side < side / 8 && raw % side < side / 8};
            for (std::uint32_t luma{0}; luma < 2; ++luma) {
                table.t[luma][table_entry(side, raw)] =
                    llf ? std::numeric_limits<float>::infinity()
                        : ac_deadzone(side, luma).threshold(raw);
            }
        }
    }
    return table;
}

// In global memory: the lanes of a warp read different entries, which
// constant memory would serialize.
__device__ const WeightPairTable WEIGHT_PAIRS{make_weight_pair_table()};
__device__ const ThresholdTable THRESHOLDS{make_threshold_table()};

// The quantization parameters of a lane's two adjacent slots, from table
// entries `entry` and `entry + 1` (`entry` even).
struct LaneWeightPairs {
    DequantWeightPair slot[2];
};

__device__ inline LaneWeightPairs lane_weight_pairs(std::uint32_t channel, std::uint32_t entry) {
    const float4 w{__ldg(reinterpret_cast<const float4*>(&WEIGHT_PAIRS.w[channel][entry]))};
    return {{{w.x, w.y}, {w.z, w.w}}};
}

struct LaneThresholds {
    float slot[2];
};

__device__ inline LaneThresholds lane_thresholds(std::uint32_t luma, std::uint32_t entry) {
    const float2 t{__ldg(reinterpret_cast<const float2*>(&THRESHOLDS.t[luma][entry]))};
    return {{t.x, t.y}};
}

__device__ inline int quantize_deadzone(float steps, float threshold) {
    return fabsf(steps) < threshold ? 0 : round_to_int(steps);
}

// Choose the smallest number of unit coefficients that recovers at least as
// much squared-error reduction as scalar quantization in this channel and group.
__device__ inline int allocate_units(float value, const DequantWeightPair weight, float threshold,
                                     float quant, float unit) {
    const float steps = div_rn(value * quant, weight.weight, weight.inv_weight);
    return isfinite(threshold) ? round_to_int(steps) : 0;
}

#include "spectral_quant_tables.cuh"
// LlfDcTerms of every covered block of a DCT16 (entries 0-3) and a DCT32
// (entries 4-19), row-major in (out_y, out_x).
constexpr std::uint32_t LLF_DC_TERM_ENTRIES = 20;

__host__ __device__ constexpr std::uint32_t llf_dc_term_entry(std::uint32_t m, std::uint32_t out_x,
                                                              std::uint32_t out_y) {
    return (m == 2 ? 0 : 4) + out_y * m + out_x;
}

// The FP64 weights of dc_from_llf_at combined into one FP32 weight per corner
// coefficient, w[fx * m + fy] = m * x[fx] * y[fy] * scale[fx] * scale[fy], so
// that a DC is m*m FP32 multiply-adds.
struct LlfDcTerms {
    float w[16];
};

__global__ void build_llf_dc_terms_kernel(LlfDcTerms* __restrict__ table) {
    const std::uint32_t entry{threadIdx.x};
    const std::uint32_t m{entry < 4 ? 2u : 4u};
    const std::uint32_t local{entry < 4 ? entry : entry - 4};
    const LlfDcWeights w{llf_dc_weights(m, local % m, local / m)};
    for (std::uint32_t fx{0}; fx < m; ++fx) {
        for (std::uint32_t fy{0}; fy < m; ++fy) {
            table[entry].w[fx * m + fy] = static_cast<float>(static_cast<double>(m) * w.x[fx] *
                                                             w.y[fy] * w.scale[fx] * w.scale[fy]);
        }
    }
}

// Built by a device kernel so that the weights carry the device math library's
// FP64 cos/sqrt, which may round differently from the host's. Lazily
// initialized on the device active at first use (the single target device). The
// kernel runs on the legacy default stream; the wait orders it before the
// kernels of the non-blocking streams.
const LlfDcTerms* llf_dc_terms_device() {
    static const LlfDcTerms* const table{[]() -> const LlfDcTerms* {
        LlfDcTerms* t{nullptr};
        if (cudaMalloc(&t, LLF_DC_TERM_ENTRIES * sizeof(LlfDcTerms)) != cudaSuccess) {
            return nullptr;
        }
        build_llf_dc_terms_kernel<<<1, LLF_DC_TERM_ENTRIES>>>(t);
        if (cudaGetLastError() != cudaSuccess || cudaStreamSynchronize(nullptr) != cudaSuccess) {
            cudaFree(t);
            return nullptr;
        }
        return t;
    }()};
    return table;
}

// 8x8 block positions per side of a 64x64 color tile.
constexpr std::uint32_t TILE_SIDE = 8;
constexpr std::uint32_t BLOCK_SLOTS = COEFFS_PER_BLOCK;
constexpr std::uint32_t TILE_POSITIONS = TILE_SIDE * TILE_SIDE;
constexpr std::uint32_t TILE_SLOTS = TILE_POSITIONS * BLOCK_SLOTS;
constexpr std::uint32_t HALVES_PER_VECTOR = sizeof(uint4) / sizeof(__half);
constexpr std::uint32_t ROW_VECTORS = TILE_SIDE * BLOCK_SLOTS / HALVES_PER_VECTOR;
constexpr std::uint32_t TILE_VECTORS = 3 * TILE_SIDE * ROW_VECTORS;
constexpr std::uint32_t QUANT_THREADS = 512;
// Bounded by the shared-memory tile; sets the register budget.
constexpr std::uint32_t QUANT_BLOCKS_PER_SM = 3;
constexpr std::uint32_t QUANT_WARPS = QUANT_THREADS / 32;
constexpr std::uint32_t QUANT_STEPS = TILE_POSITIONS / QUANT_WARPS;
constexpr std::uint32_t FETCH_VECTORS = TILE_VECTORS / QUANT_THREADS;
static_assert(BLOCK_SLOTS == 64 && TILE_POSITIONS == 64);
static_assert(TILE_POSITIONS % QUANT_WARPS == 0 && TILE_VECTORS % QUANT_THREADS == 0);
// One CfL partial-sum thread per slot and tile row.
static_assert(QUANT_THREADS == TILE_SIDE * BLOCK_SLOTS);

// A tile's order statistics are summed in shared memory as one 8-bit counter
// per CoeffOrderStats word, four per packed word: a tile adds at most
// TILE_POSITIONS to any of them.
constexpr std::uint32_t ORDER_STATS_WORDS = sizeof(CoeffOrderStats) / sizeof(std::uint32_t);
constexpr std::uint32_t PACKED_ORDER_STATS_VECTORS = (ORDER_STATS_WORDS + 15) / 16;
constexpr std::uint32_t PACKED_ORDER_STATS_WORDS = 4 * PACKED_ORDER_STATS_VECTORS;
static_assert(TILE_POSITIONS <= 0xff && PACKED_ORDER_STATS_VECTORS <= QUANT_THREADS);

__device__ inline void count_order_stat(std::uint32_t* packed_stats, std::uint32_t word) {
    atomicAdd(&packed_stats[word / 4], std::uint32_t{1} << (word % 4 * 8));
}

// Counts the non-zero flags of a lane's two slots, words `base + 2 * lane` and
// `base + 2 * lane + 1`, in one atomic; `base` is a multiple of 4, so the
// lane's byte shifts depend on the lane only.
__device__ inline void count_lane_nonzeros(std::uint32_t* packed_stats, std::uint32_t base,
                                           std::uint32_t lane, bool nonzero0, bool nonzero1) {
    const std::uint32_t shift{lane % 2 * 16};
    const std::uint32_t increment{(nonzero0 ? std::uint32_t{1} << shift : 0u) |
                                  (nonzero1 ? std::uint32_t{1} << (shift + 8) : 0u)};
    if (increment != 0) {
        atomicAdd(&packed_stats[base / 4 + lane / 2], increment);
    }
}

// A thread's share of a tile's coefficients, loaded into registers in one batch
// so that all global loads are in flight before the shared-memory store.
// Threads below TILE_POSITIONS also carry the signals of their position.
struct TileFetch {
    uint4 coeffs[FETCH_VECTORS];
    std::int32_t acs;
    std::int32_t quant_field;
};

__device__ inline TileFetch fetch_tile(const __half* __restrict__ coeffs,
                                       const std::int8_t* __restrict__ acs,
                                       const std::int32_t* __restrict__ quant_field,
                                       std::uint32_t bw, std::uint32_t bh, std::uint32_t cplane,
                                       std::uint32_t tile_bx, std::uint32_t tile_by) {
    TileFetch f{};
    const std::uint32_t cols{min(TILE_SIDE, bw - tile_bx)};
    const std::uint32_t rows{min(TILE_SIDE, bh - tile_by)};
#pragma unroll
    for (std::uint32_t i{0}; i < FETCH_VECTORS; ++i) {
        const std::uint32_t v{i * QUANT_THREADS + threadIdx.x};
        const std::uint32_t c{v / (TILE_SIDE * ROW_VECTORS)};
        const std::uint32_t row{(v / ROW_VECTORS) % TILE_SIDE};
        const std::uint32_t in_row{v % ROW_VECTORS};
        if (row < rows && in_row * HALVES_PER_VECTOR / BLOCK_SLOTS < cols) {
            const std::uint32_t src{c * cplane + ((tile_by + row) * bw + tile_bx) * BLOCK_SLOTS +
                                    in_row * HALVES_PER_VECTOR};
            f.coeffs[i] = *reinterpret_cast<const uint4*>(&coeffs[src]);
        }
    }
    const std::uint32_t lx{threadIdx.x % TILE_SIDE};
    const std::uint32_t ly{threadIdx.x / TILE_SIDE};
    if (threadIdx.x < TILE_POSITIONS && lx < cols && ly < rows) {
        const std::uint32_t blk{(tile_by + ly) * bw + tile_bx + lx};
        f.acs = acs[blk];
        f.quant_field = quant_field[blk];
    }
    return f;
}

// Slots of covered block `covered_ordinal` of a transform of side `side` that
// lie in its low-frequency corner (raw index fx*N+fy with fx, fy < N/8),
// carried as DC and excluded from AC.
__device__ inline std::uint64_t llf_slot_mask(std::uint32_t side, std::uint32_t covered_ordinal) {
    const std::uint32_t m{side / 8};
    std::uint64_t mask{0};
    for (std::uint32_t fx{0}; fx < m; ++fx) {
        for (std::uint32_t fy{0}; fy < m; ++fy) {
            const std::uint32_t raw{fx * side + fy};
            if (raw / BLOCK_SLOTS == covered_ordinal) {
                mask |= std::uint64_t{1} << (raw % BLOCK_SLOTS);
            }
        }
    }
    return mask;
}

// The transform owning a tile position, in tile-local position indices.
struct TilePosition {
    std::uint8_t owner;
    std::uint8_t side;
    std::uint8_t covered_ordinal;
};

// First CoeffOrderStats word of the non-zero counts of a position's slots.
__device__ inline std::uint32_t position_stats_base(const TilePosition& p, std::uint32_t c) {
    return static_cast<std::uint32_t>(coeff_order_offset(p.side, static_cast<int>(c))) +
           p.covered_ordinal * BLOCK_SLOTS;
}

// One CUDA block per 64x64 color tile. A DCT16/DCT32 never straddles a color
// tile (its alignment divides the tile side), so the tile's positions hold
// exactly the coefficients of the first-blocks whose CfL tile it is, and the
// whole tile is staged in shared memory from one coalesced read. Then:
// - Each position's owner is resolved from the transform signals by warp
//   shuffles (a 4x4-aligned DCT32 or 2x2-aligned DCT16 owner lies in the same
//   4 tile rows, i.e. the same warp), with its quantizer strength and
//   low-frequency slot mask.
// - Y is quantized first, since it does not depend on the color map; each
//   thread keeps its Y roundtrip values in registers for the X/B residuals.
// - The CfL regression sums are reduced per slot and tile row, then over the
//   block, in a fixed order.
// - Each position derives its DC from its transform's LLF corner in the staged
//   tile and quantizes it.
// - X and B are quantized against the tile's color map.
// - The non-zero counts per coefficient (CoeffOrderStats) are summed per step
//   in the tile's packed counters, then added to `order_stats`.
// A warp quantizes the 64 slots of one position per step (two per lane), so
// the non-zero counts of a step all go to one first-block.
// One strength per 16x8 luma pair ensures the same spatial support for 4:2:2 chroma.
__device__ const float field_step[3][64]={{2.769470214844e-03f,3.461837768555e-03f,4.615783691406e-03f,5.538940429688e-03f,7.385253906250e-03f,7.385253906250e-03f,7.385253906250e-03f,8.308410644531e-03f,3.115653991699e-03f,3.461837768555e-03f,4.615783691406e-03f,6.346702575684e-03f,7.385253906250e-03f,7.385253906250e-03f,7.385253906250e-03f,8.308410644531e-03f,4.154205322266e-03f,4.154205322266e-03f,6.750583648682e-03f,7.385253906250e-03f,7.385253906250e-03f,7.385253906250e-03f,8.308410644531e-03f,8.308410644531e-03f,6.462097167969e-03f,7.616043090820e-03f,8.308410644531e-03f,8.308410644531e-03f,8.308410644531e-03f,8.308410644531e-03f,9.231567382812e-03f,1.015472412109e-02f,9.231567382812e-03f,9.231567382812e-03f,8.308410644531e-03f,8.308410644531e-03f,9.231567382812e-03f,9.231567382812e-03f,9.231567382812e-03f,1.015472412109e-02f,1.015472412109e-02f,1.015472412109e-02f,1.015472412109e-02f,9.231567382812e-03f,9.231567382812e-03f,1.015472412109e-02f,1.292419433594e-02f,1.200103759766e-02f,1.015472412109e-02f,1.015472412109e-02f,1.107788085938e-02f,1.107788085938e-02f,1.107788085938e-02f,1.292419433594e-02f,1.200103759766e-02f,1.200103759766e-02f,1.107788085938e-02f,1.107788085938e-02f,1.107788085938e-02f,1.107788085938e-02f,1.107788085938e-02f,1.200103759766e-02f,1.200103759766e-02f,1.200103759766e-02f},{2.769470214844e-03f,3.461837768555e-03f,4.615783691406e-03f,4.615783691406e-03f,5.192756652832e-03f,5.538940429688e-03f,7.385253906250e-03f,9.000778198242e-03f,3.461837768555e-03f,3.461837768555e-03f,4.615783691406e-03f,5.192756652832e-03f,5.538940429688e-03f,6.923675537109e-03f,8.077621459961e-03f,9.520053863525e-03f,4.615783691406e-03f,4.615783691406e-03f,5.192756652832e-03f,5.538940429688e-03f,6.923675537109e-03f,7.789134979248e-03f,9.000778198242e-03f,9.520053863525e-03f,5.192756652832e-03f,5.192756652832e-03f,5.538940429688e-03f,6.923675537109e-03f,8.308410644531e-03f,8.077621459961e-03f,8.885383605957e-03f,9.231567382812e-03f,5.538940429688e-03f,5.538940429688e-03f,6.923675537109e-03f,7.385253906250e-03f,8.250713348389e-03f,9.000778198242e-03f,9.231567382812e-03f,9.346961975098e-03f,6.923675537109e-03f,8.308410644531e-03f,7.789134979248e-03f,9.693145751953e-03f,9.808540344238e-03f,9.808540344238e-03f,8.769989013672e-03f,8.308410644531e-03f,7.385253906250e-03f,8.077621459961e-03f,8.250713348389e-03f,9.000778198242e-03f,9.231567382812e-03f,9.346961975098e-03f,1.096248626709e-02f,9.231567382812e-03f,8.077621459961e-03f,7.789134979248e-03f,7.789134979248e-03f,8.077621459961e-03f,8.308410644531e-03f,8.654594421387e-03f,8.308410644531e-03f,8.308410644531e-03f},{2.769470214844e-03f,4.154205322266e-03f,4.615783691406e-03f,6.462097167969e-03f,8.308410644531e-03f,8.308410644531e-03f,8.308410644531e-03f,8.308410644531e-03f,4.154205322266e-03f,4.154205322266e-03f,5.538940429688e-03f,7.616043090820e-03f,9.231567382812e-03f,9.231567382812e-03f,7.385253906250e-03f,1.015472412109e-02f,5.538940429688e-03f,5.538940429688e-03f,7.269859313965e-03f,8.308410644531e-03f,8.308410644531e-03f,9.231567382812e-03f,9.231567382812e-03f,1.015472412109e-02f,7.385253906250e-03f,8.250713348389e-03f,9.231567382812e-03f,9.231567382812e-03f,1.015472412109e-02f,9.231567382812e-03f,8.308410644531e-03f,1.107788085938e-02f,1.015472412109e-02f,1.015472412109e-02f,9.231567382812e-03f,9.231567382812e-03f,9.231567382812e-03f,1.015472412109e-02f,1.384735107422e-02f,1.200103759766e-02f,1.015472412109e-02f,1.015472412109e-02f,1.107788085938e-02f,1.107788085938e-02f,1.292419433594e-02f,1.200103759766e-02f,1.200103759766e-02f,1.200103759766e-02f,1.015472412109e-02f,1.200103759766e-02f,1.107788085938e-02f,1.015472412109e-02f,1.661682128906e-02f,1.661682128906e-02f,1.200103759766e-02f,1.200103759766e-02f,1.384735107422e-02f,1.477050781250e-02f,1.661682128906e-02f,1.292419433594e-02f,1.200103759766e-02f,1.200103759766e-02f,1.200103759766e-02f,1.200103759766e-02f}};


template<unsigned Feature>
__device__ inline float field_feature(const float2 (&x)[4],const float2 (&step)[4],float scale,unsigned lane){
    if constexpr(Feature<5){
        float v=0;
        if constexpr(Feature==0)v=128.f+127.5f*(x[0].x+x[1].x);
        if constexpr(Feature==1)v=127.5f*fabsf(x[0].x-x[1].x);
        if constexpr(Feature==2)v=255.f*x[2].x;
        if constexpr(Feature==3)v=255.f*x[3].x;
        if constexpr(Feature==4){const float u=255.f*x[2].x,vv=255.f*x[3].x;v=fmaxf(fabsf(.344136f*u+2.116136f*vv),fabsf(2.116136f*u+.714136f*vv));}
        return __shfl_sync(~0u,v,0);
    }else if constexpr(Feature==15 || Feature==21){
        constexpr float q=Feature==15?3.f:8.f;
        unsigned count=0;
        #pragma unroll
        for(unsigned c=0;c<4;c++){
            count+=(lane!=0 && fabsf(x[c].x*q)>.5f*(step[c].x/scale));
            count+=fabsf(x[c].y*q)>.5f*(step[c].y/scale);
        }
        for(unsigned i=16;i;i>>=1)count+=__shfl_xor_sync(~0u,count,i);
        return float(count);
    }else{
        float sum=0;
        #pragma unroll
        for(unsigned c=0;c<4;c++){
            if constexpr(Feature<14){if((c<2?0u:c-1)!=(Feature-5)/3)continue;}
            #pragma unroll
            for(unsigned h=0;h<2;h++){
                const unsigned pos=2*lane+h,frequency=pos/8+pos%8;
                const float value=pos?(h?x[c].y:x[c].x):0.f;
                if constexpr(Feature<14){
                    if((frequency>3)+(frequency>7)==(Feature-5)%3)sum+=value*value*(2040.f*2040.f);
                }else{
                    constexpr unsigned qi=(Feature-14)/3,type=(Feature-14)%3;
                    constexpr float q=qi==0?3.f:qi==1?5.f:8.f;
                    const float st=(h?step[c].y:step[c].x)/scale;
                    const float rounded=rintf(value*q/st),error=(value-rounded*(st/q))*2040.f;
                    if constexpr(type==0)sum+=error*error;
                    if constexpr(type==1)sum+=float(rounded!=0.f);
                    if constexpr(type==2){if(frequency>=6)sum+=error*error;}
                }
            }
        }
        for(unsigned i=16;i;i>>=1)sum+=__shfl_down_sync(~0u,sum,i);
        return __shfl_sync(~0u,sum,0);
    }
}
__device__ inline int tree_field(const float2 (&x)[4],const float2 (&step)[4],float scale,unsigned lane){
const float f15=field_feature<15>(x,step,scale,lane);
if(f15<=1.050000000e+01f){const float f6=field_feature<6>(x,step,scale,lane);
if(f6<=1.978176575e+02f){const float f2=field_feature<2>(x,step,scale,lane);
if(f2<=-1.786743164e+01f){return 5;}else{const float f5=field_feature<5>(x,step,scale,lane);
if(f5<=9.558771515e+01f){
if(f6<=3.635887527e+01f){return 5;}else{return 6;}}else{const float f14=field_feature<14>(x,step,scale,lane);
if(f14<=2.701420898e+02f){return 6;}else{return 8;}}}}else{
if(f15<=3.500000000e+00f){return 8;}else{const float f0=field_feature<0>(x,step,scale,lane);
if(f0<=8.492672729e+01f){
if(f15<=6.500000000e+00f){return 8;}else{const float f2=field_feature<2>(x,step,scale,lane);
if(f2<=-5.077743530e-01f){return 8;}else{return 6;}}}else{return 6;}}}}else{const float f20=field_feature<20>(x,step,scale,lane);
if(f20<=3.507633209e+02f){const float f5=field_feature<5>(x,step,scale,lane);
if(f5<=7.911918701e+03f){const float f6=field_feature<6>(x,step,scale,lane);
if(f6<=2.642878571e+02f){const float f10=field_feature<10>(x,step,scale,lane);
if(f10<=1.941021252e+01f){return 6;}else{return 5;}}else{
if(f20<=2.109317932e+02f){const float f2=field_feature<2>(x,step,scale,lane);
if(f2<=-1.159515381e+01f){return 8;}else{return 6;}}else{const float f2=field_feature<2>(x,step,scale,lane);
if(f2<=-6.091833115e-01f){
if(f5<=3.016278931e+03f){return 8;}else{
if(f15<=1.650000000e+01f){return 8;}else{const float f10=field_feature<10>(x,step,scale,lane);
if(f10<=1.173091221e+01f){
if(f5<=4.878249756e+03f){return 8;}else{return 6;}}else{return 6;}}}}else{return 6;}}}}else{
if(f5<=2.732155176e+04f){const float f14=field_feature<14>(x,step,scale,lane);
if(f14<=7.451598816e+02f){const float f6=field_feature<6>(x,step,scale,lane);
if(f6<=2.954295959e+02f){return 5;}else{return 6;}}else{return 6;}}else{return 5;}}}else{const float f12=field_feature<12>(x,step,scale,lane);
if(f12<=1.377213013e+03f){const float f7=field_feature<7>(x,step,scale,lane);
if(f7<=4.019602783e+03f){const float f4=field_feature<4>(x,step,scale,lane);
if(f4<=5.707022476e+01f){const float f0=field_feature<0>(x,step,scale,lane);
if(f0<=3.192565918e+00f){
if(f0<=2.725646973e+00f){const float f14=field_feature<14>(x,step,scale,lane);
if(f14<=1.943198792e+03f){return 5;}else{return 6;}}else{const float f5=field_feature<5>(x,step,scale,lane);
if(f5<=6.874454956e+02f){return 6;}else{return 5;}}}else{const float f5=field_feature<5>(x,step,scale,lane);
if(f5<=1.093496643e+03f){
if(f5<=7.571152039e+02f){
if(f4<=1.333308518e+00f){const float f6=field_feature<6>(x,step,scale,lane);
if(f6<=2.498671875e+03f){return 6;}else{return 5;}}else{return 5;}}else{
if(f0<=3.659484863e+00f){return 5;}else{const float f6=field_feature<6>(x,step,scale,lane);
if(f6<=2.844111450e+03f){return 6;}else{return 5;}}}}else{return 5;}}}else{const float f10=field_feature<10>(x,step,scale,lane);
if(f10<=6.403425598e+01f){const float f6=field_feature<6>(x,step,scale,lane);
if(f6<=1.002671875e+03f){return 5;}else{const float f5=field_feature<5>(x,step,scale,lane);
if(f5<=2.533038770e+04f){return 6;}else{return 5;}}}else{const float f14=field_feature<14>(x,step,scale,lane);
if(f14<=1.034189026e+03f){const float f5=field_feature<5>(x,step,scale,lane);
if(f5<=3.918430908e+03f){return 5;}else{return 4;}}else{
if(f10<=1.079052658e+02f){const float f6=field_feature<6>(x,step,scale,lane);
if(f6<=2.093102539e+03f){return 5;}else{const float f17=field_feature<17>(x,step,scale,lane);
if(f17<=1.189271790e+03f){return 6;}else{return 5;}}}else{return 5;}}}}}else{const float f6=field_feature<6>(x,step,scale,lane);
if(f6<=5.704579297e+04f){const float f4=field_feature<4>(x,step,scale,lane);
if(f4<=5.411822081e+00f){
if(f6<=1.271064893e+04f){return 5;}else{return 4;}}else{const float f21=field_feature<21>(x,step,scale,lane);
if(f21<=1.105000000e+02f){return 5;}else{const float f0=field_feature<0>(x,step,scale,lane);
if(f0<=9.113383102e+01f){
if(f6<=1.305148486e+04f){return 5;}else{const float f5=field_feature<5>(x,step,scale,lane);
if(f5<=5.664555469e+04f){return 4;}else{return 5;}}}else{return 5;}}}}else{return 5;}}}else{return 6;}}}
}
__global__ void select_spatial_field(const __half* coeffs,unsigned bw,unsigned bh,int* field,int* base_field,const QuantCalibration* calibration){
    const unsigned lane=threadIdx.x%32,p=(blockIdx.x*blockDim.x+threadIdx.x)/32,width=(bw+1)/2;
    if(p>=width*bh)return;
    const unsigned l=p/width*bw+2*(p%width),r=min(l+1,(p/width+1)*bw-1),plane=bw*bh*64;
    const unsigned blocks[4]={plane+64*l,plane+64*r,64*l,2*plane+64*l};
    float2 x[4],step[4];
    #pragma unroll
    for(unsigned k=0;k<4;k++){
        x[k]=__half22float2(*reinterpret_cast<const __half2*>(coeffs+blocks[k]+2*lane));
        step[k]=__ldg(reinterpret_cast<const float2*>(field_step[k<2?1:k==2?0:2]+2*lane));
    }
    const int value=tree_field(x,step,calibration->global_scale_float,lane);
    if(lane==0){
        
        const float cb=255.f*x[2].x, cr=255.f*x[3].x;
        const float chroma_peak=fmaxf(1.402f*cr,fmaxf(-.344136f*cb-.714136f*cr,1.772f*cb));
        const float peak=128.f+255.f*fmaxf(x[0].x,x[1].x)+chroma_peak;
        field[l]=field[r]=value;
        base_field[l]=base_field[r]=int(peak<16.f);
    }
}
__device__ __align__(8) const float quant_multiplier[2][3][64]={{{3.6107989502e+02f,2.8886392212e+02f,2.1664793396e+02f,1.8053994751e+02f,1.3540495300e+02f,1.3540495300e+02f,1.3540495300e+02f,1.2035996246e+02f,3.2095989990e+02f,2.3109112549e+02f,2.1664793396e+02f,1.5756213379e+02f,1.3540495300e+02f,1.3540495300e+02f,1.3540495300e+02f,1.2035996246e+02f,2.4071992493e+02f,2.4071992493e+02f,1.4813534546e+02f,1.3540495300e+02f,1.3540495300e+02f,1.3540495300e+02f,1.2035996246e+02f,1.2035996246e+02f,1.5474851990e+02f,1.7506904602e+02f,1.2035996246e+02f,1.2035996246e+02f,1.2035996246e+02f,1.2035996246e+02f,1.0832396698e+02f,9.8476333618e+01f,1.0832396698e+02f,1.0832396698e+02f,1.2035996246e+02f,1.2035996246e+02f,1.0832396698e+02f,1.0832396698e+02f,1.0832396698e+02f,9.8476333618e+01f,9.8476333618e+01f,9.8476333618e+01f,9.8476333618e+01f,1.0832396698e+02f,1.0832396698e+02f,9.8476333618e+01f,7.7374259949e+01f,8.3326126099e+01f,9.8476333618e+01f,9.8476333618e+01f,9.0269973755e+01f,9.0269973755e+01f,7.2215980530e+01f,7.7374259949e+01f,8.3326126099e+01f,8.3326126099e+01f,9.0269973755e+01f,9.0269973755e+01f,7.2215980530e+01f,9.0269973755e+01f,4.5134986877e+01f,8.3326126099e+01f,8.3326126099e+01f,8.3326126099e+01f},{3.6107989502e+02f,2.5676791382e+02f,1.9257594299e+02f,1.9257594299e+02f,1.9257594299e+02f,1.8053994751e+02f,1.3540495300e+02f,1.1110150146e+02f,2.3109112549e+02f,2.3109112549e+02f,2.1664793396e+02f,1.9257594299e+02f,1.8053994751e+02f,1.4443196106e+02f,1.4148437500e+02f,1.0504141998e+02f,1.9257594299e+02f,1.9257594299e+02f,1.9257594299e+02f,1.8053994751e+02f,1.6506509399e+02f,1.2838395691e+02f,1.2697315216e+02f,1.0504141998e+02f,1.9257594299e+02f,1.9257594299e+02f,1.8053994751e+02f,1.6506509399e+02f,1.3755424500e+02f,1.2379882050e+02f,1.1254438019e+02f,1.0832396698e+02f,1.6047994995e+02f,1.8053994751e+02f,1.4443196106e+02f,1.3540495300e+02f,1.3851617432e+02f,1.2697315216e+02f,1.0832396698e+02f,9.5099227905e+01f,1.4443196106e+02f,1.3755424500e+02f,1.2838395691e+02f,1.1790364075e+02f,1.1651654053e+02f,1.0195196533e+02f,1.0135575867e+02f,1.2035996246e+02f,1.3540495300e+02f,1.2379882050e+02f,1.2120164490e+02f,1.1110150146e+02f,1.0832396698e+02f,1.0698663330e+02f,9.1220184326e+01f,1.0832396698e+02f,1.2379882050e+02f,1.1411907959e+02f,1.1411907959e+02f,1.2379882050e+02f,1.0698663330e+02f,1.0270716858e+02f,1.2035996246e+02f,1.2035996246e+02f},{3.6107989502e+02f,2.4071992493e+02f,2.1664793396e+02f,1.2379882050e+02f,1.2035996246e+02f,1.6047994995e+02f,1.2035996246e+02f,1.6047994995e+02f,2.4071992493e+02f,2.4071992493e+02f,1.8053994751e+02f,1.3130177307e+02f,1.0832396698e+02f,1.4443196106e+02f,1.3540495300e+02f,9.8476333618e+01f,1.8053994751e+02f,1.8053994751e+02f,1.3755424500e+02f,1.2035996246e+02f,1.2035996246e+02f,1.0832396698e+02f,1.0832396698e+02f,9.8476333618e+01f,1.3540495300e+02f,1.2120164490e+02f,1.0832396698e+02f,1.0832396698e+02f,9.8476333618e+01f,1.0832396698e+02f,1.2035996246e+02f,7.2215980530e+01f,9.8476333618e+01f,9.8476333618e+01f,1.0832396698e+02f,1.0832396698e+02f,1.0832396698e+02f,6.5650894165e+01f,7.2215980530e+01f,8.3326126099e+01f,9.8476333618e+01f,9.8476333618e+01f,9.0269973755e+01f,9.0269973755e+01f,7.7374259949e+01f,8.3326126099e+01f,8.3326126099e+01f,7.4067672729e+01f,7.8781066895e+01f,6.6660903931e+01f,9.0269973755e+01f,9.8476333618e+01f,6.0179981232e+01f,6.0179981232e+01f,7.4067672729e+01f,8.3326126099e+01f,5.7772781372e+01f,6.7702476501e+01f,3.0089990616e+01f,7.7374259949e+01f,4.1663063049e+01f,8.3326126099e+01f,8.3326126099e+01f,8.3326126099e+01f}},{{7.2215979004e+02f,5.7772784424e+02f,4.3329586792e+02f,3.6107989502e+02f,2.7080990601e+02f,2.7080990601e+02f,2.7080990601e+02f,2.4071992493e+02f,6.4191979980e+02f,4.6218225098e+02f,4.3329586792e+02f,3.1512426758e+02f,2.7080990601e+02f,2.7080990601e+02f,2.7080990601e+02f,2.4071992493e+02f,4.8143984985e+02f,4.8143984985e+02f,2.9627069092e+02f,2.7080990601e+02f,2.7080990601e+02f,2.7080990601e+02f,2.4071992493e+02f,2.4071992493e+02f,3.0949703979e+02f,3.5013809204e+02f,2.4071992493e+02f,2.4071992493e+02f,2.4071992493e+02f,2.4071992493e+02f,2.1664793396e+02f,1.9695266724e+02f,2.1664793396e+02f,2.1664793396e+02f,2.4071992493e+02f,2.4071992493e+02f,2.1664793396e+02f,2.1664793396e+02f,2.1664793396e+02f,1.9695266724e+02f,1.9695266724e+02f,1.9695266724e+02f,1.9695266724e+02f,2.1664793396e+02f,2.1664793396e+02f,1.9695266724e+02f,1.5474851990e+02f,1.6665225220e+02f,1.9695266724e+02f,1.9695266724e+02f,1.8053994751e+02f,1.8053994751e+02f,1.4443196106e+02f,1.5474851990e+02f,1.6665225220e+02f,1.6665225220e+02f,1.8053994751e+02f,1.8053994751e+02f,1.4443196106e+02f,1.8053994751e+02f,4.5134986877e+01f,1.6665225220e+02f,1.6665225220e+02f,1.6665225220e+02f},{3.6107989502e+02f,2.5676791382e+02f,1.9257594299e+02f,1.9257594299e+02f,1.9257594299e+02f,1.8053994751e+02f,1.3540495300e+02f,1.1110150146e+02f,2.3109112549e+02f,2.3109112549e+02f,2.1664793396e+02f,1.9257594299e+02f,1.8053994751e+02f,1.4443196106e+02f,1.4148437500e+02f,1.0504141998e+02f,1.9257594299e+02f,1.9257594299e+02f,1.9257594299e+02f,1.8053994751e+02f,1.6506509399e+02f,1.2838395691e+02f,1.2697315216e+02f,1.0504141998e+02f,1.9257594299e+02f,1.9257594299e+02f,1.8053994751e+02f,1.6506509399e+02f,1.3755424500e+02f,1.2379882050e+02f,1.1254438019e+02f,1.0832396698e+02f,1.6047994995e+02f,1.8053994751e+02f,1.4443196106e+02f,1.3540495300e+02f,1.3851617432e+02f,1.2697315216e+02f,1.0832396698e+02f,9.5099227905e+01f,1.4443196106e+02f,1.3755424500e+02f,1.2838395691e+02f,1.1790364075e+02f,1.1651654053e+02f,1.0195196533e+02f,1.0135575867e+02f,1.2035996246e+02f,1.3540495300e+02f,1.2379882050e+02f,1.2120164490e+02f,1.1110150146e+02f,1.0832396698e+02f,1.0698663330e+02f,9.1220184326e+01f,1.0832396698e+02f,1.2379882050e+02f,1.1411907959e+02f,1.1411907959e+02f,1.2379882050e+02f,1.0698663330e+02f,1.0270716858e+02f,1.2035996246e+02f,1.2035996246e+02f},{7.2215979004e+02f,4.8143984985e+02f,4.3329586792e+02f,2.4759764099e+02f,2.4071992493e+02f,3.2095989990e+02f,2.4071992493e+02f,3.2095989990e+02f,4.8143984985e+02f,4.8143984985e+02f,3.6107989502e+02f,2.6260354614e+02f,2.1664793396e+02f,2.8886392212e+02f,2.7080990601e+02f,1.9695266724e+02f,3.6107989502e+02f,3.6107989502e+02f,2.7510848999e+02f,2.4071992493e+02f,2.4071992493e+02f,2.1664793396e+02f,2.1664793396e+02f,1.9695266724e+02f,2.7080990601e+02f,2.4240328979e+02f,2.1664793396e+02f,2.1664793396e+02f,1.9695266724e+02f,2.1664793396e+02f,2.4071992493e+02f,1.4443196106e+02f,1.9695266724e+02f,1.9695266724e+02f,2.1664793396e+02f,2.1664793396e+02f,2.1664793396e+02f,1.3130178833e+02f,1.4443196106e+02f,1.6665225220e+02f,1.9695266724e+02f,1.9695266724e+02f,1.8053994751e+02f,1.8053994751e+02f,1.5474851990e+02f,1.6665225220e+02f,1.6665225220e+02f,7.4067672729e+01f,1.5756213379e+02f,1.3332180786e+02f,1.8053994751e+02f,1.9695266724e+02f,1.2035996246e+02f,1.2035996246e+02f,7.4067672729e+01f,1.6665225220e+02f,1.1554556274e+02f,1.3540495300e+02f,3.0089990616e+01f,1.5474851990e+02f,4.1663063049e+01f,1.6665225220e+02f,1.6665225220e+02f,1.6665225220e+02f}}};
__device__ __align__(8) const float chroma_inv[2][64]={{7.2215979004e+02f,2.2811557007e+02f,2.1214961243e+02f,1.8964807129e+02f,1.3712590027e+02f,1.4081460571e+02f,1.4264520264e+02f,1.1789535522e+02f,2.3038980103e+02f,2.1941040039e+02f,2.1079553223e+02f,1.6848484802e+02f,1.2325584412e+02f,1.1852501678e+02f,1.0936337280e+02f,1.2037637329e+02f,2.2659053040e+02f,2.1048873901e+02f,1.7411337280e+02f,1.3561181641e+02f,1.0533621979e+02f,1.0538849640e+02f,9.3695007324e+01f,1.1109341431e+02f,2.1666471863e+02f,1.9845298767e+02f,1.4460250854e+02f,1.0534151459e+02f,1.0457655334e+02f,1.0550712585e+02f,9.3522132874e+01f,7.9115791321e+01f,1.6882223511e+02f,1.4544364929e+02f,1.0283811951e+02f,8.6970939636e+01f,7.4721153259e+01f,8.1689849854e+01f,9.0286003113e+01f,9.8159446716e+01f,1.3175651550e+02f,9.5366271973e+01f,6.6314155579e+01f,6.9854713440e+01f,7.2366744995e+01f,5.0729827881e+01f,7.6256607056e+01f,8.3558532715e+01f,9.1278747559e+01f,6.2493820190e+01f,3.8279674530e+01f,6.0167034149e+01f,6.9566223145e+01f,5.6408222198e+01f,7.2739799500e+01f,4.7983161926e+01f,6.3299499512e+01f,5.3632858276e+01f,4.3171596527e+01f,5.0822952271e+01f,3.2316390991e+01f,4.9857997894e+01f,4.2448711395e+01f,5.3439815521e+01f},{7.2215979004e+02f,2.5154756165e+02f,2.1891224670e+02f,1.6083686829e+02f,1.1039530182e+02f,9.6756652832e+01f,6.5517570496e+01f,6.9253509521e+01f,2.4804789734e+02f,2.3298129272e+02f,1.9180757141e+02f,1.3991273499e+02f,9.3590995789e+01f,6.8379631042e+01f,5.0920440674e+01f,4.1576850891e+01f,2.2662725830e+02f,2.0396838379e+02f,1.5608744812e+02f,1.0781995392e+02f,5.7586578369e+01f,5.2389999390e+01f,4.1494895935e+01f,5.1863807678e+01f,2.0458273315e+02f,1.8102304077e+02f,1.4094828796e+02f,8.0987930298e+01f,4.0059825897e+01f,3.7344280243e+01f,5.0120288849e+01f,4.0566909790e+01f,1.6878147888e+02f,1.3176893616e+02f,6.9518943787e+01f,4.4017936707e+01f,3.8556613922e+01f,3.1927715302e+01f,5.0236148834e+01f,3.9211746216e+01f,1.1810781860e+02f,6.0094188690e+01f,3.6702716827e+01f,3.9917049408e+01f,3.5501998901e+01f,3.0797500610e+01f,4.8957843781e+01f,4.6943195343e+01f,6.6377388000e+01f,3.5867759705e+01f,2.9097810745e+01f,3.5081604004e+01f,4.1252777100e+01f,3.7643917084e+01f,3.7040130615e+01f,3.8227081299e+01f,3.6678325653e+01f,3.2796375275e+01f,2.2037559509e+01f,2.3518154144e+01f,3.5474678040e+01f,4.5963588715e+01f,2.8717782974e+01f,4.5476078033e+01f}};
__device__ __align__(8) const float luma_scale[64]={1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f};
// Common signed rounding thresholds, trained on the four development renders.
__device__ const signed char rounding_shift[720]={0,-36,0,0,-36,0,0,0,0,0,0,0,0,0,35,0,0,35,0,35,0,0,0,0,0,33,30,0,-36,0,0,0,0,0,0,0,0,34,34,0,34,0,0,0,0,0,0,0,0,0,34,0,0,35,36,35,35,0,0,0,0,34,0,0,0,0,0,0,0,0,0,0,0,34,0,0,0,0,0,0,0,0,0,0,36,0,35,0,0,35,36,0,0,36,-38,-38,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-36,0,0,0,0,0,0,0,0,0,0,0,-36,-36,0,-36,0,0,-35,0,0,-34,0,0,-36,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-35,0,0,0,0,0,0,35,0,0,0,0,0,0,0,0,-38,0,35,0,0,0,0,0,0,0,0,0,0,0,35,0,0,0,0,0,0,0,0,0,0,36,0,35,0,0,35,36,36,0,36,-38,-38,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-36,-35,0,-36,0,0,0,0,0,0,0,0,0,-36,0,-36,-36,0,-35,0,0,-35,0,0,-35,-36,0,0,0,0,0,0,0,0,0,0,-36,-35,0,-36,0,0,-35,0,0,0,0,0,0,-36,0,-37,0,0,-36,0,0,-36,0,0,-35,-34,0,-32,0,0,0,0,0,0,0,0,-37,-35,0,0,0,0,0,0,0,0,0,36,0,-38,0,0,0,36,36,0,36,0,-38,0,-37,-34,0,0,0,0,-35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,37,0,0,37,0,-36,0,0,-37,0,-37,0,0,0,0,0,0,0,0,-35,36,0,0,-37,0,0,0,0,0,0,0,35,35,0,35,36,0,0,-35,0,-35,0,0,-36,0,0,0,0,0,0,0,0,0,0,0,-36,-37,0,0,0,0,0,0,0,-35,0,35,35,-37,36,-37,-37,0,-37,0,0,-36,0,0,-35,-36,0,-37,0,0,0,0,0,0,0,36,0,0,0,0,-38,0,0,0,0,0,0,36,-38,0,0,36,36,37,36,-38,37,-37,-37,0,-37,0,0,0,0,0,0,0,0,0,0,0,0,0,38,0,-37,0,0,-37,0,-33,0,0,-38,0,0,-38,-37,38,36,-37,38,-37,-37,0,-36,-37,0,0,-37,0,0,0,0,0,0,0,36,0,0,0,-38,0,0,0,38,0,0,40,36,36,0,35,36,37,0,0,0,0,0,0,0,0,0,0,0,0,-36,0,0,0,0,0,0,0,0,0,-38,37,0,-39,38,0,0,0,-37,0,0,0,0,37,0,0,0,-37,0,0,0,0,0,0,40,0,0,0,0,0,0,0,0,0,0,0,-38,40,0,-38,0,0,0,36,0,0,0,0,0,0,36,-38,38,-37,-37,0,-37,-37,0,0,40,0,0,40,0,0,0,0,0,0,38,0,-38,40,0,-37,0,0,0,0,-37,0,0,-38,36,38,36,-37,38,36,-37,0,-37,-37,0,0,40,0,0,-38,0,-36,0};
__device__ __align__(8) float precision_scale[2][3][64]={{{1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,8.0000001192e-01f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.3333333731e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,8.0000001192e-01f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,8.0000001192e-01f,1.0000000000e+00f,5.0000000000e-01f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f},{1.0000000000e+00f,8.8888889551e-01f,8.8888889551e-01f,8.8888889551e-01f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,8.0000001192e-01f,8.0000001192e-01f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.1428571939e+00f,1.0000000000e+00f,8.8888889551e-01f,8.8888889551e-01f,1.0000000000e+00f,1.0000000000e+00f,1.1428571939e+00f,1.0000000000e+00f,1.1428571939e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.1428571939e+00f,1.1428571939e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,8.8888889551e-01f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.1428571939e+00f,1.1428571939e+00f,1.0000000000e+00f,8.8888889551e-01f,1.0000000000e+00f,1.1428571939e+00f,1.0000000000e+00f,1.1428571939e+00f,1.1428571939e+00f,1.0000000000e+00f,8.8888889551e-01f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,8.8888889551e-01f,8.8888889551e-01f,1.0000000000e+00f,8.8888889551e-01f,8.8888889551e-01f,1.0000000000e+00f,1.0000000000e+00f},{1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,8.0000001192e-01f,1.0000000000e+00f,1.3333333731e+00f,1.0000000000e+00f,1.3333333731e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.3333333731e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,8.0000001192e-01f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,6.6666668653e-01f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,8.8888889551e-01f,8.0000001192e-01f,8.0000001192e-01f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,8.8888889551e-01f,1.0000000000e+00f,8.0000001192e-01f,1.0000000000e+00f,5.0000000000e-01f,1.0000000000e+00f,5.0000000000e-01f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f}},{{2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,1.6000000238e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.6666667461e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,1.6000000238e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,1.6000000238e+00f,2.0000000000e+00f,5.0000000000e-01f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f},{1.0000000000e+00f,8.8888889551e-01f,8.8888889551e-01f,8.8888889551e-01f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,8.0000001192e-01f,8.0000001192e-01f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.1428571939e+00f,1.0000000000e+00f,8.8888889551e-01f,8.8888889551e-01f,1.0000000000e+00f,1.0000000000e+00f,1.1428571939e+00f,1.0000000000e+00f,1.1428571939e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.1428571939e+00f,1.1428571939e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,8.8888889551e-01f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.1428571939e+00f,1.1428571939e+00f,1.0000000000e+00f,8.8888889551e-01f,1.0000000000e+00f,1.1428571939e+00f,1.0000000000e+00f,1.1428571939e+00f,1.1428571939e+00f,1.0000000000e+00f,8.8888889551e-01f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,1.0000000000e+00f,8.8888889551e-01f,8.8888889551e-01f,1.0000000000e+00f,8.8888889551e-01f,8.8888889551e-01f,1.0000000000e+00f,1.0000000000e+00f},{2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,1.6000000238e+00f,2.0000000000e+00f,2.6666667461e+00f,2.0000000000e+00f,2.6666667461e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.6666667461e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,1.6000000238e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,1.3333333731e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,8.8888889551e-01f,1.6000000238e+00f,1.6000000238e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f,8.8888889551e-01f,2.0000000000e+00f,1.6000000238e+00f,2.0000000000e+00f,5.0000000000e-01f,2.0000000000e+00f,5.0000000000e-01f,2.0000000000e+00f,2.0000000000e+00f,2.0000000000e+00f}}};
__device__ __align__(8) unsigned char bright_stride[3][64]={{2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2},{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},{2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,1,2,2,2,1,2,1,2,2,2}};
__device__ unsigned char block_context_table[]={CUJPEGXL_BLOCK_CONTEXTS};
__device__ unsigned char level_map[2][64]={{0,1,0,0,0,0,0,0,2,3,4,0,0,0,0,0,5,6,0,0,0,0,0,0,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},{0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};
__device__ unsigned char level_values[][128]={{0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,16,16,16,16,16,16,16,16},{0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,16,16,16,16,16,16,16},{0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,14,14,14,14,14,14,14,14,14,14,0,14,14,14,14,14,14,16,16,16,16,16,16,16},{0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,12,12,12,12,0,0,12,12,0,0,12,12,0,0,12,0,12,0,0,0,0,0,0,14,14,0,0,0,0,0,0,0,0,0,16,0,0,0,0,0},{0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,16,16,16,16,16,16,16,16},{0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,14,14,14,14,14,14,14,0,14,14,14,14,14,14,0,14,14,16,16,16,16,16,16,16},{0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,12,0,12,0,12,12,0,12,12,12,12,12,0,12,0,12,0,0,0,14,14,0,14,14,14,0,0,0,0,0,14,0,0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,16,16,16,16,16,16,16,16},{0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,12,12,12,0,12,12,12,12,0,12,12,12,12,12,12,12,14,0,14,0,14,14,0,14,0,14,0,0,0,0,14,0,0,0,16,16,0,0,16,0}};
__global__ void quantize_square(
    const __half* __restrict__ coeffs, const int* __restrict__ quant_field,
    const QuantCalibration* calibration, unsigned bw, unsigned bh, unsigned tiles_x,
    std::int8_t* ytox, std::int8_t* ytob, short* ac, int* dc, int* nz_exact,
    int* nz_grid, CoeffOrderStats* order_stats, int sample_phase) {
    __shared__ unsigned counters[3*16];
    __shared__ int tile_dc[3][64];
    __shared__ unsigned char tile_count[3][64];
    const unsigned tid=threadIdx.x, warp=tid/32, lane=tid%32, entry=2*lane;
    const unsigned tile=blockIdx.x, bx0=8*(tile%tiles_x), by0=8*(tile/tiles_x);
    const unsigned blocks=bw*bh, plane=blocks*64;
    const bool collect=sample_phase<0 || tile%16==unsigned(sample_phase);
    if(tid==0){ytox[tile]=0;ytob[tile]=0;}
    if(collect){if(tid<48)counters[tid]=0;__syncthreads();}
    const float gsf=calibration->global_scale_float, dcscale=gsf*float(calibration->quant_dc);
    for(unsigned p=warp;p<64;p+=blockDim.x/32){
        const unsigned bx=bx0+p%8,by=by0+p/8;
        if(bx>=bw || by>=bh)continue;
        const unsigned blk=by*bw+bx;
        const unsigned ix=blk*64+entry;
        const float2 y=__half22float2(*reinterpret_cast<const __half2*>(coeffs+plane+ix));
        const float mean=128.f+255.f*__shfl_sync(~0u,y.x,0);
        const unsigned brightness=(mean>=8)+(mean>=16)+(mean>=32)+(mean>=96);
        const float y0=entry?y.x*2040.f:0.f, y1=y.y*2040.f;
        float energy=y0*y0+y1*y1;
        for(unsigned i=16;i;i>>=1)energy+=__shfl_xor_sync(~0u,energy,i);
        const unsigned group=unsigned(energy>64*36)+unsigned(energy>64*196)+unsigned(energy>64*1024);
        const int base_field=quant_field[blk];const bool dark=nz_grid[blk]!=0;


        #pragma unroll
        for(unsigned c=0;c<3;++c){
            const float scale=base_field*gsf;
            const float2 v=c==1?y:__half22float2(*reinterpret_cast<const __half2*>(coeffs+c*plane+ix));
            if(lane==0)tile_dc[c][p]=round_to_int(v.x*(DC_INV_QUANT[c]*dcscale));
            unsigned nz=0;
            if(c==1 || (bx&1)==0){
                
                short val[2];
                if(c!=1){
                    const auto factor=__ldg(reinterpret_cast<const float2*>(chroma_inv[c/2]+entry));
                    const float bias=c==0?.94534993f:.95006490f;
                    #pragma unroll
                    for(unsigned h=0;h<2;++h){
                        const float x=((h?v.y:v.x)*scale)*(h?factor.y:factor.x);
                        const float a=fabsf(x);const int low=int(a);
                        const auto level=[&](int n){return n==0?0.f:n==1?bias:float(n)-.145f/float(n);};
                        const int mag=low+int(2.f*a>=level(low)+level(low+1));
                        val[h]=entry+h==0?0:(x<0.f?-mag:mag);
                    }
                }else{
                const float2 factor=__ldg(reinterpret_cast<const float2*>(quant_multiplier[dark][c]+entry));
                const uchar2 stride=__ldg(reinterpret_cast<const uchar2*>(bright_stride[c]+entry));
                const float2 base=__ldg(reinterpret_cast<const float2*>(tr_base_half[c]+entry));
                const uint4 suffix=__ldg(reinterpret_cast<const uint4*>(tr_suffix[c]+entry));
                float x[2];unsigned mask[2];
                for(unsigned h=0;h<2;++h){

                    x[h]=(h?v.y:v.x)*(scale*(h?factor.y:factor.x));
                    x[h]*=__ldg(luma_scale+entry+h);
                    mask[h]=__ballot_sync(~0u,entry+h!=0 && fabsf(x[h])>=(h?base.y:base.x));
                }
                for(unsigned h=0;h<2;++h){
                    const unsigned pos=entry+h;
                    unsigned n=__popc(mask[0]&(h?suffix.z:suffix.x))+__popc(mask[1]&(h?suffix.w:suffix.y));
                    unsigned level=(n>=1)+(n>=2)+(n>=4)+(n>=8)+(n>=16);
                    const float a=fabsf(x[h]);const unsigned m=unsigned(a);const float frac=64*(a-m);
                    const unsigned t=__ldg(&warp_threshold[c][group][min(m,4u)][level][pos]);
                    int mag=m+(frac>=(t>>8))-(m!=0 && frac<(t&255));
                    const unsigned frequency=pos/8+pos%8;
                    const unsigned band=(frequency>=3)+(frequency>=6)+(frequency>=9);
                    const int rule=__ldg(rounding_shift+((((brightness*4+group)*3+c)*4+band)*3+min(mag,2)));
                    const int phase=__float2int_rd(8.f*(a-float(mag)));
                    if(rule>0 && phase>=rule-32)++mag;
                    if(rule<0 && mag && phase<=rule+32)--mag;
                    if (!dark) {
                        mag *= h ? stride.y : stride.x;
                        if (c != 1 && a < 8.f) {
                            const int table = __ldg(level_map[c/2] + pos);
                            if (table) mag = __ldg(level_values[table-1] + int(a*16.f));
                        }
                    }
                    val[h]=pos==0?0:(x[h]<0?-mag:mag);
                }
                }
                *reinterpret_cast<short2*>(ac+c*plane+ix)=make_short2(val[0],val[1]);
                nz=__popc(__ballot_sync(~0u,val[0]!=0))+__popc(__ballot_sync(~0u,val[1]!=0));
                if(collect)count_lane_nonzeros(counters,64*c,lane,val[0]!=0,val[1]!=0);
            }
            if(lane==0)tile_count[c][p]=nz;
        }
    }
    __syncthreads();
    if(tid<192){
        const unsigned c=tid/64,p=tid%64;
        const unsigned bx=bx0+p%8,by=by0+p/8;
        if(bx<bw&&by<bh){
            const int ydc=tile_dc[1][p],cb=tile_dc[0][p&~1u],cr=tile_dc[2][p&~1u];
            const int qcol=c==1?bx:(bx&~31u)+(bx%32)/2;
            const int strength=quant_field[by*bw+qcol];
            const unsigned luma_bin=(ydc>-332)+(ydc>-324)+(ydc>-305)+(ydc>-230)+(ydc>-128)+(ydc>-39)+(ydc>53),cb_bin=(cb>-5),cr_bin=0,field_bin=(strength>4)+(strength>5)+(strength>6);
            const unsigned channel=c<2?c^1:2;
            const unsigned key=(((channel*4+field_bin)*2+cb_bin)*1+cr_bin)*8+luma_bin;
            const unsigned block_context=__ldg(block_context_table+key);
            const unsigned dst=c*blocks+by*bw+bx,nz=tile_count[c][p];
            dc[dst]=tile_dc[c][p];nz_grid[dst]=int(nz);
            nz_exact[dst]=int(nz|(block_context<<16));
        }
    }
    if(collect){
        if(tid<192){
            const unsigned c=tid/64, k=tid%64;
            const unsigned n=(counters[tid/4]>>(8*(tid%4)))&255;
            if(n)atomicAdd(reinterpret_cast<unsigned*>(order_stats)+coeff_order_offset(8,c)+k,n);
        }
        if(tid==0)atomicAdd(&order_stats->blocks[coeff_order_side_index(8)],min(8u,bw-bx0)*min(8u,bh-by0));
    }
}

}  // namespace

void select_pair_precision(const __half* coeffs, unsigned bw,unsigned bh,int* field,int* base_field,const QuantCalibration* calibration,cudaStream_t stream){
    select_spatial_field<<<(((bw+1)/2)*bh+3)/4,128,0,stream>>>(coeffs,bw,bh,field,base_field,calibration);
}

bool quantize_scan_residual(const __half* coeffs, const std::int8_t* acs,
                            const std::int32_t* quant_field, const QuantCalibration* calibration,
                            std::size_t width, std::size_t height, std::int8_t* ytox_map,
                            std::int8_t* ytob_map, std::int16_t* ac, std::int32_t* dc,
                            std::int32_t* nz_exact, std::int32_t* nz_grid,
                            CoeffOrderStats* order_stats, cudaStream_t stream, int sample_phase) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::uint32_t tiles_x{static_cast<std::uint32_t>(ceil_div(bw, TILE_SIDE))};
    const std::uint32_t tiles{tiles_x * static_cast<std::uint32_t>(ceil_div(bh, TILE_SIDE))};
    if (sample_phase < 0 &&
        cudaMemsetAsync(order_stats, 0, sizeof(CoeffOrderStats), stream) != cudaSuccess) {
        return false;
    }
    quantize_square<<<tiles,256,0,stream>>>(coeffs,quant_field,calibration,bw,bh,tiles_x,ytox_map,ytob_map,ac,dc,nz_exact,nz_grid,order_stats,sample_phase);
    return cudaGetLastError() == cudaSuccess;
}

bool quantize_residual(const __half* coeffs, const std::int8_t* acs,
                       const std::int32_t* quant_field, const QuantCalibration* calibration,
                       std::size_t width, std::size_t height, std::int8_t* ytox_map,
                       std::int8_t* ytob_map, std::int16_t* ac, std::int32_t* dc,
                       std::int32_t* nz_exact, std::int32_t* nz_grid, CoeffOrderStats* order_stats,
                       cudaStream_t stream) {
    return quantize_scan_residual(coeffs, acs, quant_field, calibration, width, height, ytox_map,
                                  ytob_map, ac, dc, nz_exact, nz_grid, order_stats, stream, -1);
}

}  // namespace cujpegxl
