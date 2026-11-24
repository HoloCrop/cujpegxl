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
        const float y0=entry?y.x*2040.f:0.f, y1=y.y*2040.f;
        float energy=y0*y0+y1*y1;
        for(unsigned i=16;i;i>>=1)energy+=__shfl_xor_sync(~0u,energy,i);
        const unsigned group=unsigned(energy>64*36)+unsigned(energy>64*196)+unsigned(energy>64*1024);
        const float scale=float(quant_field[blk])*gsf;


        #pragma unroll
        for(unsigned c=0;c<3;++c){
            const float2 v=c==1?y:__half22float2(*reinterpret_cast<const __half2*>(coeffs+c*plane+ix));
            if(lane==0)tile_dc[c][p]=round_to_int(v.x*(DC_INV_QUANT[c]*dcscale));
            unsigned nz=0;
            if(c==1 || (bx&1)==0){
                const auto weights=lane_weight_pairs(c,entry);
                const float2 base=__ldg(reinterpret_cast<const float2*>(tr_base_half[c]+entry));
                const uint4 suffix=__ldg(reinterpret_cast<const uint4*>(tr_suffix[c]+entry));
                float x[2];unsigned mask[2];short val[2];
                for(unsigned h=0;h<2;++h){
                    const auto wt=weights.slot[h];
                    x[h]=div_rn((h?v.y:v.x)*scale,wt.weight,wt.inv_weight);
                    mask[h]=__ballot_sync(~0u,entry+h!=0 && fabsf(x[h])>=(h?base.y:base.x));
                }
                for(unsigned h=0;h<2;++h){
                    const unsigned pos=entry+h;
                    unsigned n=__popc(mask[0]&(h?suffix.z:suffix.x))+__popc(mask[1]&(h?suffix.w:suffix.y));
                    unsigned level=(n>=1)+(n>=2)+(n>=4)+(n>=8)+(n>=16);
                    const float a=fabsf(x[h]);const unsigned m=unsigned(a);const float frac=64*(a-m);
                    const unsigned t=__ldg(&warp_threshold[c][group][min(m,4u)][level][pos]);
                    const int mag=m+(frac>=(t>>8))-(m!=0 && frac<(t&255));
                    val[h]=pos==0?0:(x[h]<0?-mag:mag);
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
            const int ydc=tile_dc[1][p];const int lumaclass=(ydc>-332)+(ydc>-313)+(ydc>-104)+(ydc>25);
            const unsigned dst=c*blocks+by*bw+bx,nz=tile_count[c][p];
            dc[dst]=tile_dc[c][p];nz_grid[dst]=int(nz);
            nz_exact[dst]=int(nz|(((c<2?c^1:2)*5+lumaclass)<<16));
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
