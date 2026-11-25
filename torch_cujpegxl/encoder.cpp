// SPDX-License-Identifier: BSD-3-Clause
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "cujpegxl/cujpegxl.h"

namespace py = pybind11;

namespace {

void check(cujpegxl_status status) {
    TORCH_CHECK(status == CUJPEGXL_OK, cujpegxl_status_string(status));
}

struct Slot {
    at::Tensor storage;
    at::Tensor input;
    cujpegxl_output output{};
};

class Encoder {
public:
    Encoder(uint32_t width, uint32_t height, float distance, c10::DeviceIndex device_index,
            uint32_t depth)
        : config_{width, height, distance}, device_{c10::kCUDA, device_index} {
        TORCH_CHECK(device_index >= 0, "device_index must identify a CUDA device");
        c10::cuda::CUDAGuard guard(device_);
        cujpegxl_encoder* native = nullptr;
        check(cujpegxl_encoder_create(&config_, device_index, depth, &native));
        handle_.reset(native);
        size_t capacity = 0;
        check(cujpegxl_max_output_size(&config_, &capacity));
        slots_.resize(depth);
        for (Slot& slot : slots_) {
            slot.storage = at::empty({static_cast<int64_t>(capacity)},
                                     at::TensorOptions().dtype(at::kByte).device(at::kCPU).pinned_memory(true));
            slot.output = {slot.storage.data_ptr<uint8_t>(), 0, capacity};
        }
    }

    ~Encoder() {
        c10::cuda::CUDAGuard guard(device_);
        handle_.reset();  // Drain before destroying retained input and output tensors.
    }

    py::bytes encode(const at::Tensor& image) {
        auto lock = acquire();
        c10::cuda::CUDAGuard guard(device_);
        require_empty();
        validate(image);
        submit(image);
        return receive();
    }

    py::list encode_batch(const std::vector<at::Tensor>& images) {
        auto lock = acquire();
        c10::cuda::CUDAGuard guard(device_);
        require_empty();
        auto files = enqueue(images);
        drain(files);
        return files;
    }

    py::list push(const std::vector<at::Tensor>& images) {
        auto lock = acquire();
        c10::cuda::CUDAGuard guard(device_);
        return enqueue(images);
    }

    py::bytes receive_one() {
        auto lock = acquire();
        c10::cuda::CUDAGuard guard(device_);
        TORCH_CHECK(handle_ && pending_, "no pending encode");
        return receive();
    }

    py::list flush() {
        auto lock = acquire();
        c10::cuda::CUDAGuard guard(device_);
        TORCH_CHECK(handle_, "encoder failed and must be recreated");
        py::list files;
        drain(files);
        return files;
    }

private:
    std::unique_lock<std::mutex> acquire() {
        py::gil_scoped_release release;
        return std::unique_lock<std::mutex>(mutex_);
    }

    void require_empty() const {
        TORCH_CHECK(pending_ == 0, "flush pending images before starting a synchronous encode");
    }

    void validate(const at::Tensor& image) const {
        TORCH_CHECK(image.device() == device_ && image.scalar_type() == at::kByte,
                    "expected a uint8 tensor on ", device_);
        TORCH_CHECK(image.dim() == 3 && image.size(0) == config_.height &&
                    image.size(1) == config_.width && image.size(2) == 3,
                    "expected HWC RGB at the configured height and width");
        TORCH_CHECK(image.stride(2) == 1 && image.stride(1) == 3 &&
                    image.stride(0) >= static_cast<int64_t>(config_.width) * 3,
                    "expected packed RGB pixels with a positive row pitch");
    }

    void finish_call(cujpegxl_status status) {
        if (status != CUJPEGXL_OK) {
            handle_.reset();
            check(status);
        }
    }

    void submit(const at::Tensor& image) {
        TORCH_CHECK(handle_, "encoder failed and must be recreated");
        Slot& slot = slots_[(oldest_ + pending_) % slots_.size()];
        slot.input = image;
        cujpegxl_rgb8_input pixels{image.data_ptr<uint8_t>(), static_cast<size_t>(image.stride(0)),
                                   c10::cuda::getCurrentCUDAStream(device_.index()).stream()};
        py::gil_scoped_release release;
        finish_call(cujpegxl_encoder_submit_rgb8(handle_.get(), &pixels, &slot.output));
        ++pending_;
    }

    py::bytes receive() {
        cujpegxl_output* output = nullptr;
        {
            py::gil_scoped_release release;
            finish_call(cujpegxl_encoder_receive(handle_.get(), &output));
        }
        Slot& slot = slots_[oldest_];
        slot.input.reset();
        oldest_ = (oldest_ + 1) % slots_.size();
        --pending_;
        return py::bytes(reinterpret_cast<const char*>(output->data), output->size);
    }

    py::list enqueue(const std::vector<at::Tensor>& images) {
        TORCH_CHECK(handle_, "encoder failed and must be recreated");
        for (const auto& image : images) validate(image);
        py::list files;
        for (const auto& image : images) {
            if (pending_ == slots_.size()) files.append(receive());
            submit(image);
        }
        return files;
    }

    void drain(py::list& files) {
        while (pending_ > 0) files.append(receive());
    }

    cujpegxl_config config_;
    c10::Device device_;
    std::mutex mutex_;
    std::vector<Slot> slots_;
    std::unique_ptr<cujpegxl_encoder, decltype(&cujpegxl_encoder_destroy)> handle_{nullptr, cujpegxl_encoder_destroy};
    size_t oldest_{0};
    size_t pending_{0};
};
}  // namespace

void register_decoder(py::module_&);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module, py::mod_gil_not_used()) {
    register_decoder(module);
    py::class_<Encoder>(module, "Encoder")
        .def(py::init<uint32_t, uint32_t, float, c10::DeviceIndex, uint32_t>(),
             py::arg("width"), py::arg("height"), py::arg("distance"),
             py::arg("device_index") = 0, py::arg("pipeline_depth") = 3)
        .def("encode", &Encoder::encode, py::arg("image"))
        .def("encode_batch", &Encoder::encode_batch, py::arg("images"))
        .def("encode_batch", [](Encoder& self, const at::Tensor& batch) {
            TORCH_CHECK(batch.dim() == 4, "expected NHWC RGB batch");
            return self.encode_batch(batch.unbind(0));
        }, py::arg("images"))
        .def("push", &Encoder::push, py::arg("images"))
        .def("_receive", &Encoder::receive_one)
        .def("flush", &Encoder::flush);
}
