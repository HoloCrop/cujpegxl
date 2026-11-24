// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_FRAME_ENCODER_H_
#define CUJPEGXL_SRC_FRAME_ENCODER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_runtime_api.h>

#include "gpu_arch.h"

namespace cujpegxl {

// Per-stage instrumentation record. `gpu_us`/`cpu_us` are measured wall time of
// the device and host portions of a stage; because the device sub-calls are
// synchronous, blocking wall time attributes cleanly to each. `bytes_moved` is
// the analytic DRAM traffic of the stage.
struct StageTiming {
    const char* name{""};
    std::size_t bytes_moved{0};
    double gpu_us{0.0};
    double cpu_us{0.0};
};

enum class EncodeStatus : std::uint8_t { OK, BUFFER_TOO_SMALL, FAILED };

struct EncodeResult {
    EncodeStatus status{EncodeStatus::FAILED};
    // The bytes written on OK; the capacity the file needs on BUFFER_TOO_SMALL.
    std::size_t file_size{0};
};

// Capacities of the coded sections: the device DcGroups buffer (an encode fails
// when the DcGroups outgrow it) and bounds of the AcGroups and AcGlobal, so
// that together they bound the coded body of a frame.
struct SectionCapacities {
    std::size_t ac_groups{0};
    std::size_t dc_groups{0};
    std::size_t ac_global{0};
};

SectionCapacities section_capacities(std::size_t width, std::size_t height);

// Upper bound of the `.jxl` file size of a width x height frame: with an output
// buffer of this capacity, an encode never reports BUFFER_TOO_SMALL.
std::size_t max_encoded_size(std::size_t width, std::size_t height);

// submit() queues device work and receive() returns the files in submission
// order. A host thread assembles the headers while the device processes AC
// groups. The caller serializes submit() and receive().
class FrameEncoder {
public:
    // worker_threads is retained for source compatibility.
    FrameEncoder(std::int32_t device_ordinal, GpuArch arch, std::size_t worker_threads,
                 std::size_t pipeline_depth);
    FrameEncoder(const FrameEncoder&) = delete;
    FrameEncoder& operator=(const FrameEncoder&) = delete;
    // Waits for the frames in flight to complete.
    ~FrameEncoder();

    std::size_t pipeline_depth() const;
    std::size_t in_flight() const;

    // Starts encoding a device-resident NV12 image into the `.jxl` file written
    // to the host buffer `out_file` of `out_capacity` bytes. Requires
    // in_flight() < pipeline_depth(). The image must stay unchanged and
    // `out_file` and `stats` valid until the frame is received. False, with
    // the frame not in flight, on a CUDA error or for a single-AC-group frame.
    //
    // `luma`/`chroma` are device addresses (see nv12_to_xyb). The planes are read
    // through textures, which require 32-byte-aligned row pitches; a plane whose
    // pitch is not aligned is first copied into an aligned buffer. The image is read after the
    // work enqueued on `input_stream`, a stream of the encoder's device, before
    // the call. `distance` drives quantization. The chroma plane holds
    // (width+1)/2 pairs per row over (height+1)/2 rows. Single-AC-group frames
    // use the combined-section layout and are not handled here.
    //
    // Each file embeds its own AC codebook. The encoder learns the codebook
    // from the scan and periodically updates it using samples from later images.
    // An identical input sequence produces identical bytes at any pipeline depth.
    //
    // When `stats` is non-null it receives the "frontend", "entropy" and
    // "assembly" stage timings; the front end is then waited for in submit().
    [[nodiscard]] bool submit(const std::uint8_t* luma, std::size_t luma_pitch,
                              const std::uint8_t* chroma, std::size_t chroma_pitch,
                              cudaStream_t input_stream, std::size_t width, std::size_t height,
                              float distance, std::uint8_t* out_file, std::size_t out_capacity,
                              std::vector<StageTiming>* stats = nullptr);

    // Waits for the oldest frame in flight and returns its result. Requires
    // in_flight() > 0. The file size is known before the first write to
    // `out_file`, so BUFFER_TOO_SMALL leaves the buffer untouched. FAILED on a
    // CUDA error or if a coded DC section exceeds its capacity.
    [[nodiscard]] EncodeResult receive();

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_FRAME_ENCODER_H_
