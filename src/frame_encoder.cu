// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "frame_encoder.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <numeric>
#include <optional>
#include <thread>

#include <cuda_runtime.h>

#include <cuda_fp16.h>

#include "ac_clustering.h"
#include "coeff_order.h"
#include "entropy.h"
#include "frontend_fused.h"
#include "frontend_quantize.h"
#include "prefix_build.h"
#include "quant_calibration.h"
#include "src/bitstream/container.h"
#include "src/bitstream/frame_assembly.h"
#include "src/bitstream/token_stream.h"
#include "vardct_layout.h"

namespace cujpegxl {
namespace {
__global__ void age_histograms(unsigned* hist, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        hist[i] /= 2;
}

__global__ void finish_groups(const unsigned* layout, unsigned ng, unsigned* sizes,
                              const unsigned char* body, unsigned char* result) {
    unsigned g = blockIdx.y;
    const unsigned* bits = layout + 2 * (ng + 1);
    unsigned count = (bits[g] + 7) / 8, offset = 0;
    for (unsigned i = 0; i < g; ++i)
        offset += (bits[i] + 7) / 8;
    if (threadIdx.x == 0 && blockIdx.x == 0)
        sizes[g] = count;
    body += 8 * layout[ng + 1 + g];
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x)
        result[offset + i] = body[i];
}
using Clock = std::chrono::steady_clock;

constexpr EncodeResult ENCODE_FAILED{EncodeStatus::FAILED, 0};

constexpr std::size_t TEXTURE_PITCH_ALIGNMENT{32};

double us_since(Clock::time_point start) {
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

std::size_t aligned_plane_pitch(std::size_t width) {
    return (width + 511) & ~std::size_t{511};
}

struct Plane {
    const std::uint8_t* data;
    std::size_t pitch;
};

// `plane` itself if a texture can sample its pitch, else its `row_bytes` x `rows`
// copy into `aligned` at aligned_plane_pitch(width), enqueued on `stream`.
// Nullopt on a CUDA error.
std::optional<Plane> texture_plane(Plane plane, std::size_t row_bytes, std::size_t rows,
                                   std::size_t width, std::uint8_t* aligned, cudaStream_t stream) {
    if (plane.pitch % TEXTURE_PITCH_ALIGNMENT == 0) {
        return plane;
    }
    const Plane copy{aligned, aligned_plane_pitch(width)};
    if (cudaMemcpy2DAsync(aligned, copy.pitch, plane.data, plane.pitch, row_bytes, rows,
                          cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        return std::nullopt;
    }
    return copy;
}

// Grow-only page-locked host buffer of `T`, kept across frames.
template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    ~PinnedBuffer() {
        if (data_ != nullptr) {
            cudaFreeHost(data_);
        }
    }

    // Makes room for `count` elements; the contents are not kept.
    bool reserve(std::size_t count) {
        if (count <= capacity_) {
            return true;
        }
        if (data_ != nullptr) {
            cudaFreeHost(data_);
            data_ = nullptr;
            capacity_ = 0;
        }
        // Headroom, so that frames of slowly growing size do not reallocate
        // every time.
        const std::size_t grown{count + count / 4};
        void* p{nullptr};
        if (cudaHostAlloc(&p, grown * sizeof(T), cudaHostAllocPortable) != cudaSuccess) {
            return false;
        }
        data_ = static_cast<T*>(p);
        capacity_ = grown;
        return true;
    }

    T* data() const { return data_; }

private:
    T* data_{nullptr};
    std::size_t capacity_{0};
};

// Grow-only device allocation, kept across frames. Persistent scratch keeps the
// frames in flight independent: per-frame stream-ordered allocations would let
// the pool hand a block freed by one frame's stream to another's, ordering the
// two.
class DeviceArena {
public:
    DeviceArena() = default;
    DeviceArena(const DeviceArena&) = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;
    ~DeviceArena() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    // Makes room for `bytes`; the contents are not kept.
    bool reserve(std::size_t bytes) {
        if (bytes <= capacity_) {
            return true;
        }
        if (data_ != nullptr) {
            cudaFree(data_);
            data_ = nullptr;
            capacity_ = 0;
        }
        void* p{nullptr};
        if (cudaMalloc(&p, bytes) != cudaSuccess) {
            return false;
        }
        data_ = static_cast<std::uint8_t*>(p);
        capacity_ = bytes;
        return true;
    }

    std::uint8_t* data() const { return data_; }

private:
    std::uint8_t* data_{nullptr};
    std::size_t capacity_{0};
};

// Hands out consecutive aligned ranges from address `base`; with base 0 it only
// measures their total size.
class ArenaCarver {
public:
    explicit ArenaCarver(std::uintptr_t base) : base_{base} {}

    template <typename T>
    T* take(std::size_t count) {
        constexpr std::size_t ALIGNMENT{256};
        offset_ = (offset_ + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
        T* const p{reinterpret_cast<T*>(base_ + offset_)};
        offset_ += count * sizeof(T);
        return p;
    }

    std::size_t size() const { return offset_; }

private:
    std::uintptr_t base_;
    std::size_t offset_{0};
};

struct FrameBuffers {
    std::uint8_t* aligned_luma{nullptr};
    std::uint8_t* aligned_chroma{nullptr};
    unsigned* lengths;
    unsigned char* body;
    __half* coeffs{nullptr};
    std::int8_t* acs{nullptr};
    std::int8_t* ytox_map{nullptr};
    std::int8_t* ytob_map{nullptr};
    float* adaptive_field{nullptr};
    std::uint32_t* field_histogram{nullptr};
    std::int32_t* quant_field{nullptr};
    QuantCalibration* calibration{nullptr};
    std::int16_t* ac{nullptr};
    std::int32_t* dc{nullptr};
    std::int32_t* nz_exact{nullptr};
    std::int32_t* nz_grid{nullptr};
    std::uint32_t* context_hist{nullptr};
    std::uint8_t* cluster_scratch{nullptr};
    std::uint64_t* item_sizes{nullptr};
    std::uint32_t* token_records{nullptr};
    std::uint8_t* layout_scratch{nullptr};
    std::uint64_t* chunk_base{nullptr};
    std::uint32_t* layout{nullptr};
    std::uint32_t* ac_counters{nullptr};
    std::uint32_t* dc_group_hist{nullptr};
    std::uint8_t* dc_stats_scratch{nullptr};
    std::size_t* fb_pos{nullptr};
    std::size_t* fbc{nullptr};
    std::uint8_t* dc_depth{nullptr};
    std::uint16_t* dc_bits{nullptr};
    std::uint8_t* am_depth{nullptr};
    std::uint16_t* am_bits{nullptr};
    std::uint8_t* pre{nullptr};
    std::uint32_t* pre_off{nullptr};
    std::uint32_t* pre_bits{nullptr};
    std::uint8_t* mid{nullptr};
    std::uint32_t* mid_off{nullptr};
    std::uint32_t* mid_bits{nullptr};
    std::uint8_t* dc_body{nullptr};
    std::uint32_t* dc_sizes{nullptr};
    std::uint32_t* dc_offsets{nullptr};
    std::uint8_t* dc_scratch{nullptr};
};

FrameBuffers carve_frame_buffers(ArenaCarver& carver, std::size_t width, std::size_t height) {
    const std::size_t blocks{(width / 8) * (height / 8)};
    const std::size_t color_tiles{(((width / 8) + 7) / 8) * (((height / 8) + 7) / 8)};
    const std::size_t num_ac{bitstream::ac_group_count(width, height)};
    const std::size_t num_dc{bitstream::dc_group_count(width, height)};
    const std::size_t codes{num_dc * bitstream::HISTOGRAM_STRIDE};
    return FrameBuffers{
        .aligned_luma = carver.take<std::uint8_t>(aligned_plane_pitch(width) * height),
        .aligned_chroma = carver.take<std::uint8_t>(aligned_plane_pitch(width) * (height / 2)),
        .lengths = carver.take<unsigned>(num_ac),
        .body = carver.take<unsigned char>(4 * ac_max_tokens(width, height)),
        .coeffs = carver.take<__half>(3 * blocks * COEFFS_PER_BLOCK),
        .acs = carver.take<std::int8_t>(blocks),
        .ytox_map = carver.take<std::int8_t>(color_tiles),
        .ytob_map = carver.take<std::int8_t>(color_tiles),
        .adaptive_field = carver.take<float>(blocks),
        .field_histogram = carver.take<std::uint32_t>(ADAPTIVE_FIELD_BINS),
        .quant_field = carver.take<std::int32_t>(blocks),
        .calibration = carver.take<QuantCalibration>(1),
        .ac = carver.take<std::int16_t>(3 * blocks * COEFFS_PER_BLOCK),
        .dc = carver.take<std::int32_t>(3 * blocks),
        .nz_exact = carver.take<std::int32_t>(3 * blocks),
        .nz_grid = carver.take<std::int32_t>(3 * blocks),
        .context_hist = carver.take<std::uint32_t>(AC_NUM_CONTEXTS * AC_CONTEXT_ALPHABET),
        .cluster_scratch = carver.take<std::uint8_t>(ac_cluster_scratch_bytes()),
        .item_sizes = carver.take<std::uint64_t>(3 * blocks),
        .token_records = carver.take<std::uint32_t>(ac_max_tokens(width, height)),
        .layout_scratch = carver.take<std::uint8_t>(ac_token_layout_scratch_bytes(width, height)),
        .chunk_base = carver.take<std::uint64_t>(ac_token_chunks(width, height)),
        .layout = carver.take<std::uint32_t>(3 * num_ac + 2),
        .ac_counters = carver.take<std::uint32_t>(ac_counter_words(width, height)),
        .dc_group_hist = carver.take<std::uint32_t>(2 * num_dc * AC_HISTOGRAM_SIZE),
        .dc_stats_scratch = carver.take<std::uint8_t>(dc_group_stats_scratch_bytes(width, height)),
        .fb_pos = carver.take<std::size_t>(num_dc * DC_FIRST_BLOCK_STRIDE),
        .fbc = carver.take<std::size_t>(num_dc),
        .dc_depth = carver.take<std::uint8_t>(codes),
        .dc_bits = carver.take<std::uint16_t>(codes),
        .am_depth = carver.take<std::uint8_t>(codes),
        .am_bits = carver.take<std::uint16_t>(codes),
        .pre = carver.take<std::uint8_t>(num_dc * DC_BLOB_STRIDE),
        .pre_off = carver.take<std::uint32_t>(num_dc),
        .pre_bits = carver.take<std::uint32_t>(num_dc),
        .mid = carver.take<std::uint8_t>(num_dc * DC_BLOB_STRIDE),
        .mid_off = carver.take<std::uint32_t>(num_dc),
        .mid_bits = carver.take<std::uint32_t>(num_dc),
        .dc_body = carver.take<std::uint8_t>(section_capacities(width, height).dc_groups),
        .dc_sizes = carver.take<std::uint32_t>(num_dc),
        .dc_offsets = carver.take<std::uint32_t>(num_dc),
        .dc_scratch = carver.take<std::uint8_t>(dc_encode_scratch_bytes(width, height)),
    };
}

// A frame in flight: its parameters and result, and the resources it uses. The
// device work runs on `stream`, with the DC path on `dc_stream`.
struct Slot {
    Slot() = default;
    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;
    ~Slot() {
        textures.reset();
        release();
    }

    // Creates the streams and events on the current device, once.
    bool init();
    void release();
    // Waits until the slot's streams and copies have no queued work. False on a
    // CUDA error.
    bool wait_idle();

    bool ready{false};
    cudaStream_t stream{nullptr};
    cudaStream_t dc_stream{nullptr};
    cudaEvent_t input_ready{nullptr};
    cudaEvent_t forked{nullptr};
    cudaEvent_t layout_copied{nullptr};
    cudaEvent_t dc_sizes_copied{nullptr};
    cudaEvent_t ac_complete{nullptr};
    std::array<cudaEvent_t, 2> stream_idle{};

    DeviceArena arena;
    FrameBuffers device{};
    PinnedBuffer<QuantCalibration> calibration;
    PinnedBuffer<AcClustering> clustering;
    PinnedBuffer<CoeffOrders> coeff_orders;
    PinnedBuffer<unsigned> ac_lengths;
    PinnedBuffer<std::uint32_t> dc_sizes;
    std::optional<Nv12Textures> textures;

    std::size_t width{0};
    std::size_t height{0};
    std::uint8_t* out_file{nullptr};
    std::size_t out_capacity{0};
    std::vector<StageTiming>* stats{nullptr};
    uint64_t model_generation{};
    Clock::time_point entropy_start{};
    EncodeResult result{};
};

bool Slot::init() {
    if (ready) {
        return true;
    }
    release();
    const auto create_event = [](cudaEvent_t& e) {
        return cudaEventCreateWithFlags(&e, cudaEventDisableTiming) == cudaSuccess;
    };
    // Events the host waits on put the waiting thread to sleep instead of spinning.
    const auto create_host_event = [](cudaEvent_t& e) {
        return cudaEventCreateWithFlags(&e, cudaEventDisableTiming | cudaEventBlockingSync) ==
               cudaSuccess;
    };
    // The main stream gets the greatest priority, so that the block scheduler
    // dispatches the AC path's short kernels ahead of pending blocks of the DC
    // path's large grids. Non-blocking, so that the slots' streams do not
    // synchronize with each other through the legacy default stream.
    int least_priority{0};
    int greatest_priority{0};
    ready = cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority) == cudaSuccess &&
            cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest_priority) ==
                cudaSuccess &&
            cudaStreamCreateWithPriority(&dc_stream, cudaStreamNonBlocking, least_priority) ==
                cudaSuccess &&
            create_event(input_ready) && create_event(forked) && create_host_event(layout_copied) &&
            create_host_event(dc_sizes_copied) && create_host_event(ac_complete);
    for (cudaEvent_t& e : stream_idle) {
        ready = ready && create_host_event(e);
    }
    return ready;
}

bool Slot::wait_idle() {
    const std::array<cudaStream_t, 2> streams{stream, dc_stream};
    for (std::size_t i{0}; i < streams.size(); ++i) {
        if (cudaEventRecord(stream_idle[i], streams[i]) != cudaSuccess ||
            cudaEventSynchronize(stream_idle[i]) != cudaSuccess) {
            return false;
        }
    }
    return true;
}

void Slot::release() {
    const auto destroy_event = [](cudaEvent_t& e) {
        if (e != nullptr) {
            cudaEventDestroy(e);
            e = nullptr;
        }
    };
    for (cudaEvent_t* e : {&input_ready, &forked, &layout_copied, &dc_sizes_copied, &ac_complete}) {
        destroy_event(*e);
    }
    for (cudaEvent_t& e : stream_idle) {
        destroy_event(e);
    }
    for (cudaStream_t* s : {&stream, &dc_stream}) {
        if (*s != nullptr) {
            cudaStreamDestroy(*s);
            *s = nullptr;
        }
    }
    ready = false;
}

}  // namespace

SectionCapacities section_capacities(std::size_t width, std::size_t height) {
    const std::size_t blocks{(padded_to_blocks(width) / 8) * (padded_to_blocks(height) / 8)};
    const std::size_t num_ac{bitstream::ac_group_count(width, height)};
    const std::size_t num_dc{bitstream::dc_group_count(width, height)};
    return SectionCapacities{3 * blocks * 384 + num_ac * 64 + 4096,
                             48 * blocks + 2 * num_dc * DC_BLOB_STRIDE + num_dc * 64 + 4096,
                             std::size_t{1} << 16};
}

std::size_t max_encoded_size(std::size_t width, std::size_t height) {
    // The container framing, codestream headers, FrameHeader and DcGlobal take
    // well under FIXED_HEADERS_BOUND; a TOC entry takes at most 4 bytes.
    constexpr std::size_t FIXED_HEADERS_BOUND{1024};
    constexpr std::size_t TOC_ENTRY_BOUND{4};
    const std::size_t num_sections{bitstream::dc_group_count(width, height) +
                                   bitstream::ac_group_count(width, height) + 2};
    const SectionCapacities caps{section_capacities(width, height)};
    return caps.ac_groups + caps.dc_groups + caps.ac_global + FIXED_HEADERS_BOUND +
           TOC_ENTRY_BOUND * num_sections;
}

// Frames update a shared device model in submission order. Each slot retains
// its model snapshot until the driver has serialized its self-contained file.
struct FrameEncoder::State {
    State(int device_ordinal, GpuArch arch, std::size_t pipeline_depth)
        : device{device_ordinal},
          arch{arch},
          depth{pipeline_depth},
          slots{std::make_unique<Slot[]>(pipeline_depth)},
          driver{[this] { drive(); }} {}
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    ~State() {
        {
            const std::lock_guard<std::mutex> lock{mutex};
            stop = true;
        }
        submitted_cv.notify_all();
        driver.join();
    }
    bool enqueue_frame(Slot& slot, const std::uint8_t* luma, std::size_t luma_pitch,
                       const std::uint8_t* chroma, std::size_t chroma_pitch,
                       cudaStream_t input_stream, float distance);
    void drive();
    EncodeResult code_frame(Slot& slot);

    const int device;
    const GpuArch arch;
    const std::size_t depth;
    const std::unique_ptr<Slot[]> slots;
    DeviceArena entropy_model;
    DeviceArena order_model;
    uint64_t generation{0};

    std::vector<std::uint8_t> ac_global;
    std::size_t described_groups{0};
    uint64_t described_generation{UINT64_MAX};
    std::vector<std::uint32_t> section_sizes;

    // Frame counts since construction; submitted and received change only on
    // the caller thread and completed changes only on the
    // driver thread.
    std::mutex mutex;
    std::condition_variable submitted_cv;
    std::condition_variable completed_cv;
    std::uint64_t submitted{0};
    std::uint64_t completed{0};
    std::uint64_t received{0};
    bool stop{false};
    std::thread driver;
};

bool FrameEncoder::State::enqueue_frame(Slot& slot, const std::uint8_t* luma,
                                        std::size_t luma_pitch, const std::uint8_t* chroma,
                                        std::size_t chroma_pitch, cudaStream_t input_stream,
                                        float distance) {
    // The stages after the front end cover the frame padded to whole blocks.
    const std::size_t width{padded_to_blocks(slot.width)};
    const std::size_t height{padded_to_blocks(slot.height)};
    const std::size_t num_ac{bitstream::ac_group_count(width, height)};
    const std::size_t num_dc{bitstream::dc_group_count(width, height)};

    ArenaCarver measure{0};
    carve_frame_buffers(measure, width, height);
    if (!slot.arena.reserve(measure.size()) || !slot.ac_lengths.reserve(num_ac) ||
        !slot.calibration.reserve(1) || !slot.clustering.reserve(1) ||
        !slot.coeff_orders.reserve(1) || !slot.dc_sizes.reserve(num_dc)) {
        return false;
    }
    ArenaCarver carver{reinterpret_cast<std::uintptr_t>(slot.arena.data())};
    slot.device = carve_frame_buffers(carver, width, height);
    const FrameBuffers& d{slot.device};

    // Only `slot.stream` reads the image; the other streams fork from it later.
    if (cudaEventRecord(slot.input_ready, input_stream) != cudaSuccess ||
        cudaStreamWaitEvent(slot.stream, slot.input_ready, 0) != cudaSuccess) {
        return false;
    }
    // The frames' main streams run in submission order: the serial host stage
    // needs the oldest frame first, and frames sharing the GPU would all
    // finish late together.
    if (submitted != 0 &&
        cudaStreamWaitEvent(slot.stream, slots[(submitted - 1) % depth].ac_complete, 0) !=
            cudaSuccess) {
        return false;
    }

    if (!order_model.reserve(sizeof(CoeffOrderStats) + sizeof(CoeffOrders)))
        return false;
    auto* order_history = reinterpret_cast<CoeffOrderStats*>(order_model.data());
    auto* orders = reinterpret_cast<CoeffOrders*>(order_history + 1);
    const auto frontend_start = Clock::now();
    bool ready;
    if (chroma == nullptr) {
        ready = encode_rgb_frontend(luma, luma_pitch, slot.width, slot.height, distance, d.coeffs,
                                    d.acs, d.adaptive_field, d.field_histogram, d.quant_field,
                                    d.calibration, slot.stream);
    } else {
        const std::optional<Plane> luma_src{texture_plane(
            {luma, luma_pitch}, slot.width, slot.height, width, d.aligned_luma, slot.stream)};
        const std::optional<Plane> chroma_src{
            texture_plane({chroma, chroma_pitch}, (slot.width + 1) / 2 * 2, (slot.height + 1) / 2,
                          width, d.aligned_chroma, slot.stream)};
        if (!luma_src || !chroma_src) {
            return false;
        }
        slot.textures.emplace(luma_src->data, luma_src->pitch, chroma_src->data, chroma_src->pitch,
                              slot.width, slot.height, slot.stream);

        ready = encode_frontend(*slot.textures, slot.width, slot.height, distance, d.coeffs, d.acs,
                                d.adaptive_field, d.field_histogram, d.quant_field, d.calibration,
                                slot.stream);
    }
    if (!ready || !quantize_scan_residual(d.coeffs, d.acs, d.quant_field, d.calibration, width,
                                          height, d.ytox_map, d.ytob_map, d.ac, d.dc, d.nz_exact,
                                          d.nz_grid, order_history, slot.stream,
                                          submitted == 0 ? -1 : int(submitted % 16))) {
        return false;
    }
    if (slot.stats != nullptr) {
        // Instrumentation-only wait, so that the front end's wall time is its own.
        if (cudaStreamSynchronize(slot.stream) != cudaSuccess) {
            return false;
        }
        slot.stats->push_back(StageTiming{"frontend", 0, us_since(frontend_start), 0.0});
    }
    slot.entropy_start = Clock::now();

    // The DC path (DcGroup histograms, prefix codes and sections) runs on its own
    // stream alongside the AC path.
    if (cudaEventRecord(slot.forked, slot.stream) != cudaSuccess ||
        cudaStreamWaitEvent(slot.dc_stream, slot.forked, 0) != cudaSuccess ||
        !dc_group_stats(d.dc, d.quant_field, d.acs, d.ytox_map, d.ytob_map, width, height,
                        d.dc_stats_scratch, d.dc_group_hist, d.fb_pos, d.fbc, slot.dc_stream) ||
        !build_dc_group_blobs_device(width, height, d.dc_group_hist,
                                     d.dc_group_hist + num_dc * AC_HISTOGRAM_SIZE, d.fbc,
                                     d.dc_depth, d.dc_bits, d.am_depth, d.am_bits, d.pre, d.pre_off,
                                     d.pre_bits, d.mid, d.mid_off, d.mid_bits, slot.dc_stream) ||
        !dc_encode_groups(d.dc, width, height, d.quant_field, d.acs, d.ytox_map, d.ytob_map,
                          d.fb_pos, d.fbc, d.dc_depth, d.dc_bits, d.am_depth, d.am_bits, d.pre,
                          d.pre_off, d.pre_bits, d.mid, d.mid_off, d.mid_bits, d.dc_scratch,
                          d.dc_body, section_capacities(width, height).dc_groups, d.dc_sizes,
                          d.dc_offsets, slot.dc_stream) ||
        cudaMemcpyAsync(slot.dc_sizes.data(), d.dc_sizes, num_dc * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, slot.dc_stream) != cudaSuccess ||
        cudaEventRecord(slot.dc_sizes_copied, slot.dc_stream) != cudaSuccess) {
        return false;
    }

    constexpr size_t bins = AC_NUM_CONTEXTS * AC_CONTEXT_ALPHABET;
    if (!entropy_model.reserve(sizeof(AcClustering) +
                               (AC_MAX_CLUSTERS * bitstream::HISTOGRAM_STRIDE + bins) *
                                   sizeof(unsigned)))
        return false;
    auto model = reinterpret_cast<AcClustering*>(entropy_model.data());
    auto table = reinterpret_cast<unsigned*>(model + 1);
    auto history = table + AC_MAX_CLUSTERS * bitstream::HISTOGRAM_STRIDE;
    const bool refresh =
        submitted == 2 || submitted == 4 || submitted == 8 || (submitted && submitted % 16 == 0);
    if ((submitted == 0 || refresh) && !ac_coeff_orders(order_history, orders, slot.stream))
        return false;
    if (submitted == 0) {
        if (cudaMemsetAsync(order_history, 0, sizeof(*order_history), slot.stream) != cudaSuccess)
            return false;
        if (cudaMemsetAsync(history, 0, bins * sizeof(unsigned), slot.stream) != cudaSuccess)
            return false;
        if (!ac_build_histogram(d.ac, d.acs, d.nz_grid, d.nz_exact, orders, width, height,
                                d.context_hist, d.item_sizes, d.token_records, d.ac_counters, arch,
                                slot.stream) ||
            !ac_cluster_contexts(d.context_hist, d.cluster_scratch, model, slot.stream))
            return false;
        if (!build_scan_prefix(model, table, slot.stream))
            return false;
        generation = submitted;
    }
    // Adapt quickly at startup, then halve the history at each 16-image refresh.
    if (refresh) {
        if (!ac_cluster_contexts(history, d.cluster_scratch, model, slot.stream))
            return false;
        if (!build_scan_prefix(model, table, slot.stream))
            return false;
        generation = submitted;
        constexpr size_t order_words = sizeof(CoeffOrderStats) / sizeof(unsigned);
        age_histograms<<<(order_words + 255) / 256, 256, 0, slot.stream>>>(
            reinterpret_cast<unsigned*>(order_history), order_words);
        age_histograms<<<(bins + 255) / 256, 256, 0, slot.stream>>>(history, bins);
    }
    slot.model_generation = generation;
    if (!ac_code_items(d.ac, d.acs, d.nz_grid, d.nz_exact, orders, width, height, history,
                       d.item_sizes, d.token_records, d.ac_counters, arch, slot.stream, model,
                       table, submitted % 16) ||
        !ac_token_layout(d.item_sizes, width, height, d.layout_scratch, d.ac_counters, d.chunk_base,
                         d.layout, reinterpret_cast<uint64_t*>(d.body), slot.stream) ||
        cudaMemcpyAsync(slot.calibration.data(), d.calibration, sizeof(QuantCalibration),
                        cudaMemcpyDeviceToHost, slot.stream) != cudaSuccess ||
        cudaMemcpyAsync(slot.clustering.data(), model, sizeof(AcClustering), cudaMemcpyDeviceToHost,
                        slot.stream) != cudaSuccess ||
        // The host reads only the Lehmer codes of the orders.
        cudaMemcpyAsync(slot.coeff_orders.data()->lehmer, orders->lehmer,
                        sizeof(CoeffOrders::lehmer), cudaMemcpyDeviceToHost,
                        slot.stream) != cudaSuccess ||
        cudaEventRecord(slot.layout_copied, slot.stream) != cudaSuccess) {
        return false;
    }

    auto st = slot.stream;
    auto packed = reinterpret_cast<unsigned*>(d.body);
    if (!prefix_pack(d.acs, d.token_records, width, height, d.item_sizes, d.chunk_base, d.layout, 0,
                     num_ac, packed, d.ac_counters, st))
        return false;
    finish_groups<<<dim3(8, num_ac), 128, 0, st>>>(
        d.layout, num_ac, d.lengths, d.body, reinterpret_cast<unsigned char*>(d.token_records));
    if (cudaMemcpyAsync(slot.ac_lengths.data(), d.lengths, 4 * num_ac, cudaMemcpyDeviceToHost,
                        st) != cudaSuccess ||
        cudaEventRecord(slot.ac_complete, st) != cudaSuccess)
        return false;
    return true;
}

void FrameEncoder::State::drive() {
    // A failure shows in the frames' CUDA calls.
    cudaSetDevice(device);
    std::unique_lock<std::mutex> lock{mutex};
    for (;;) {
        submitted_cv.wait(lock, [this] { return stop || completed != submitted; });
        if (completed == submitted) {
            return;
        }
        const std::uint64_t frame{completed};
        lock.unlock();
        Slot& slot{slots[frame % depth]};
        const EncodeResult result{code_frame(slot)};
        lock.lock();
        slot.result = result;
        ++completed;
        completed_cv.notify_one();
    }
}

EncodeResult FrameEncoder::State::code_frame(Slot& slot) {
    const std::size_t width{slot.width};
    const std::size_t height{slot.height};
    const std::size_t num_ac{bitstream::ac_group_count(width, height)};
    const std::size_t num_dc{bitstream::dc_group_count(width, height)};
    const SectionCapacities caps{section_capacities(width, height)};
    StageTiming entropy{"entropy", 0, 0.0, 0.0};
    StageTiming assembly{"assembly", 0, 0.0, 0.0};

    if (cudaEventSynchronize(slot.layout_copied) != cudaSuccess) {
        return ENCODE_FAILED;
    }
    entropy.gpu_us =
        std::chrono::duration<double, std::micro>(Clock::now() - slot.entropy_start).count();
    const Clock::time_point entropy_cpu_start{Clock::now()};
    const QuantCalibration& calibration{*slot.calibration.data()};
    const std::vector<std::uint8_t> dc_global{bitstream::build_dc_global(
        bitstream::QuantParams{calibration.global_scale, calibration.quant_dc})};

    if (described_generation != slot.model_generation || described_groups != num_ac) {
        const auto description = bitstream::describe_scan_histograms(*slot.clustering.data());
        ac_global = bitstream::build_scan_ac_global(*slot.coeff_orders.data(), num_ac, description);
        described_groups = num_ac;
        described_generation = slot.model_generation;
    }
    auto s = slot.stream;
    auto body = slot.device.token_records;
    auto& lengths = slot.ac_lengths;
    if (cudaEventSynchronize(slot.ac_complete) != cudaSuccess ||
        cudaEventSynchronize(slot.dc_sizes_copied) != cudaSuccess)
        return ENCODE_FAILED;
    entropy.cpu_us = us_since(entropy_cpu_start);
    auto assembly_start = Clock::now();
    const size_t ac_total = std::accumulate(lengths.data(), lengths.data() + num_ac, size_t{0});
    const size_t dc_total =
        std::accumulate(slot.dc_sizes.data(), slot.dc_sizes.data() + num_dc, size_t{0});
    if (dc_total > caps.dc_groups)
        return ENCODE_FAILED;
    section_sizes.clear();
    section_sizes.push_back(dc_global.size());
    section_sizes.insert(section_sizes.end(), slot.dc_sizes.data(), slot.dc_sizes.data() + num_dc);
    section_sizes.push_back(ac_global.size());
    section_sizes.insert(section_sizes.end(), lengths.data(), lengths.data() + num_ac);
    auto head = bitstream::build_codestream_head(width, height, section_sizes);
    size_t total = head.size() + dc_global.size() + dc_total + ac_global.size() + ac_total;
    auto framing = bitstream::container_framing(total);
    total += framing.size();
    if (total > slot.out_capacity)
        return {EncodeStatus::BUFFER_TOO_SMALL, total};
    auto p = std::copy(framing.begin(), framing.end(), slot.out_file);
    p = std::copy(head.begin(), head.end(), p);
    p = std::copy(dc_global.begin(), dc_global.end(), p);
    if (cudaMemcpyAsync(p, slot.device.dc_body, dc_total, cudaMemcpyDeviceToHost, s) != cudaSuccess)
        return ENCODE_FAILED;
    p = std::copy(ac_global.begin(), ac_global.end(), p + dc_total);
    if (cudaMemcpyAsync(p, body, ac_total, cudaMemcpyDeviceToHost, s) != cudaSuccess ||
        cudaStreamSynchronize(s) != cudaSuccess)
        return ENCODE_FAILED;
    if (slot.stats != nullptr) {
        assembly.cpu_us = us_since(assembly_start);
        slot.stats->push_back(entropy);
        slot.stats->push_back(assembly);
    }
    return {EncodeStatus::OK, total};
}

FrameEncoder::FrameEncoder(std::int32_t device_ordinal, GpuArch arch,
                           std::size_t /*worker_threads*/, std::size_t pipeline_depth)
    : state_{std::make_unique<State>(device_ordinal, arch, pipeline_depth)} {
    assert(pipeline_depth >= 1);
}

FrameEncoder::~FrameEncoder() = default;

std::size_t FrameEncoder::pipeline_depth() const {
    return state_->depth;
}

std::size_t FrameEncoder::in_flight() const {
    const std::lock_guard<std::mutex> lock{state_->mutex};
    return static_cast<std::size_t>(state_->submitted - state_->received);
}

bool FrameEncoder::submit(const std::uint8_t* luma, std::size_t luma_pitch,
                          const std::uint8_t* chroma, std::size_t chroma_pitch,
                          cudaStream_t input_stream, std::size_t width, std::size_t height,
                          float distance, std::uint8_t* out_file, std::size_t out_capacity,
                          std::vector<StageTiming>* stats) {
    assert(in_flight() < pipeline_depth());
    State& s{*state_};
    if (bitstream::ac_group_count(width, height) <= 1) {
        return false;
    }
    Slot& slot{s.slots[s.submitted % s.depth]};
    // The slot's previous frame has completed, but after a failure its device
    // work may still be running.
    if (cudaSetDevice(s.device) != cudaSuccess || !slot.init() || !slot.wait_idle()) {
        return false;
    }
    slot.width = width;
    slot.height = height;
    slot.out_file = out_file;
    slot.out_capacity = out_capacity;
    slot.stats = stats;
    if (!s.enqueue_frame(slot, luma, luma_pitch, chroma, chroma_pitch, input_stream, distance)) {
        return false;
    }
    {
        const std::lock_guard<std::mutex> lock{s.mutex};
        ++s.submitted;
    }
    s.submitted_cv.notify_all();
    return true;
}

EncodeResult FrameEncoder::receive() {
    assert(in_flight() > 0);
    State& s{*state_};
    std::unique_lock<std::mutex> lock{s.mutex};
    s.completed_cv.wait(lock, [&s] { return s.completed != s.received; });
    const EncodeResult result{s.slots[s.received % s.depth].result};
    ++s.received;
    return result;
}

}  // namespace cujpegxl
