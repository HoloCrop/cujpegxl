#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>
#include <span>
#include <string_view>

#include "src/camera_decode.h"

namespace py = pybind11;
extern "C" int camera_coefficients(const unsigned char*, size_t, unsigned, unsigned, camera_block*,
                                   camera_tables*, camera_plan*, camera_allocate, void*);
void render_camera(float*, const camera_block*, const camera_tables*, const int32_t*, bool,
                   const float*, unsigned char*, int, int, int, int, int, cudaStream_t);

void decode_camera_groups(float*, const int32_t*, camera_plan, const camera_block*, int*, int, int,
                          cudaStream_t);

namespace {
struct HostPayload {
    at::Tensor values;
    std::exception_ptr error;
    static int32_t* allocate(void* context, size_t bytes) noexcept {
        auto& self = *static_cast<HostPayload*>(context);
        try {
            self.values = at::empty({int64_t((bytes + 3) / 4)},
                                    at::TensorOptions().dtype(at::kInt).pinned_memory(true));
            return self.values.data_ptr<int32_t>();
        } catch (...) {
            self.error = std::current_exception();
            return nullptr;
        }
    }
};
struct PendingDecode {
    at::Tensor pixels;
    std::vector<at::Tensor> status;
    std::vector<at::cuda::CUDAEvent> checked;
    at::cuda::CUDAEvent ready;

    void validate() const {
        for (size_t i = 0; i < status.size(); ++i)
            if (status[i].defined())
                TORCH_CHECK(*status[i].data_ptr<int>() == 0, "invalid camera AC group at batch index ", i);
    }

    at::Tensor result() const {
        ready.synchronize();
        validate();
        return pixels;
    }

    at::Tensor on_current_stream() {
        for (const auto& event : checked)
            if (event.isCreated()) event.synchronize();
        validate();
        auto consumer = c10::cuda::getCurrentCUDAStream(pixels.get_device());
        ready.block(consumer);
        c10::cuda::CUDACachingAllocator::recordStream(pixels.storage().data_ptr(), consumer);
        return pixels;
    }
};

struct Decoder {
    int width, height;
    c10::Device device;
    Decoder(int w, int h, c10::DeviceIndex index) : width(w), height(h), device(at::kCUDA, index) {
        TORCH_CHECK(w > 0 && h > 0 && int64_t(w + 15LL) * (h + 7LL) < INT_MAX / 4,
                    "invalid image dimensions");
    }

    at::Tensor batch(std::span<const std::string_view> files,
                     std::optional<at::Tensor> maps, PendingDecode* pending = nullptr) const {
        c10::cuda::CUDAGuard guard(device);
        TORCH_CHECK(files.size() > 0 && files.size() <= 65535, "expected 1 to 65535 files");
        int ow = width, oh = height;
        size_t stride = 0;
        if (maps) {
            TORCH_CHECK(maps->device() == device && maps->scalar_type() == at::kFloat &&
                            maps->is_contiguous() && (maps->dim() == 3 || maps->dim() == 4) &&
                            maps->size(-1) == 2 &&
                            (maps->dim() == 3 || maps->size(0) == int64_t(files.size())),
                        "expected contiguous CUDA float32 maps [H,W,2] or [B,H,W,2]");
            TORCH_CHECK(maps->size(-2) > 0 && maps->size(-3) > 0 &&
                            maps->size(-2) * maps->size(-3) < INT_MAX - 255,
                        "invalid remap dimensions");
            ow = maps->size(-2);
            oh = maps->size(-3);
            if (maps->dim() == 4)
                stride = size_t(ow) * oh;
        }
        int count = 4 * ((width + 15) / 16) * ((height + 7) / 8);
        int64_t batch_size = files.size();
        auto pinned = at::TensorOptions().dtype(at::kByte).pinned_memory(true);
        auto out =
            at::empty({batch_size, oh, ow, 3}, at::TensorOptions().device(device).dtype(at::kByte));
        if (pending) {
            pending->status.resize(batch_size);
            pending->checked.resize(batch_size);
        }
        auto caller = c10::cuda::getCurrentCUDAStream();
        at::cuda::CUDAEvent ready;
        std::vector<at::cuda::CUDAEvent> finished(batch_size);
        if (batch_size > 1)
            ready.record(caller);

        auto decode = [&](int64_t i) {
            auto stream =
                batch_size == 1 ? caller : c10::cuda::getStreamFromPool(false, device.index());
            c10::cuda::CUDAStreamGuard guard(stream);
            if (batch_size > 1)
                ready.block(stream);
            auto metadata =
                at::empty({int64_t(sizeof(camera_tables) + count * sizeof(camera_block))}, pinned);
            auto* tables = reinterpret_cast<camera_tables*>(metadata.data_ptr());
            auto* headers = reinterpret_cast<camera_block*>(tables + 1);
            HostPayload payload;
            const auto& file = files[i];
            camera_plan plan{};
            int size = camera_coefficients(reinterpret_cast<const unsigned char*>(file.data()),
                                           file.size(), width, height, headers, tables, &plan,
                                           HostPayload::allocate, &payload);
            if (payload.error)
                std::rethrow_exception(payload.error);
            TORCH_CHECK(size >= 0, "invalid or unsupported camera JPEG XL at batch index ", i);
            auto gpu_metadata = metadata.to(device, true);
            auto gpu_values = payload.values.narrow(0, 0, size).to(device, true);
            auto gpu_tables = reinterpret_cast<const camera_tables*>(gpu_metadata.data_ptr());
            auto gpu_headers = reinterpret_cast<const camera_block*>(gpu_tables + 1);
            auto blocks = at::empty({int64_t(count) * 64}, out.options().dtype(at::kFloat));
            if (batch_size > 1) {
                c10::cuda::CUDACachingAllocator::recordStream(out.storage().data_ptr(), stream);
                if (maps)
                    c10::cuda::CUDACachingAllocator::recordStream(maps->storage().data_ptr(),
                                                                  stream);
            }
            at::cuda::CUDAEvent checked;
            at::Tensor host_status;
            if (plan.groups) {
                auto status = at::empty({1}, gpu_values.options());
                host_status = at::empty({1}, pinned.dtype(at::kInt));
                decode_camera_groups(blocks.data_ptr<float>(), gpu_values.data_ptr<int32_t>(), plan,
                                     gpu_headers, status.data_ptr<int>(), count, width, stream);
                host_status.copy_(status, true);
                (pending ? pending->checked[i] : checked).record(stream);
            }
            render_camera(
                blocks.data_ptr<float>(), gpu_headers, gpu_tables, gpu_values.data_ptr<int32_t>(),
                plan.groups != 0, maps ? maps->data_ptr<float>() + i * stride * 2 : nullptr,
                out.data_ptr<uint8_t>() + i * ow * oh * 3, count, width, height, ow, oh, stream);
            if (plan.groups && pending) {
                pending->status[i] = std::move(host_status);
            } else if (plan.groups) {
                checked.synchronize();
                TORCH_CHECK(*host_status.data_ptr<int>() == 0,
                            "invalid camera AC group at batch index ", i);
            }
            if (batch_size > 1)
                finished[i].record(stream);
        };
        if (batch_size == 1) {
            decode(0);
        } else {
            // Workers enqueue uploads as soon as their own entropy stream is ready.
            // Only the caller's CUDA stream waits for the batch to finish.
            std::vector<std::exception_ptr> errors(batch_size);
            int workers = std::min<int64_t>(at::get_num_threads(), batch_size);
#pragma omp parallel for schedule(dynamic) num_threads(workers)
            for (int64_t i = 0; i < batch_size; ++i) {
                try {
                    decode(i);
                } catch (...) {
                    errors[i] = std::current_exception();
                }
            }
            for (auto& event : finished)
                if (event.isCreated())
                    event.block(caller);
            for (auto& error : errors)
                if (error)
                    std::rethrow_exception(error);
        }
        if (pending) pending->ready.record(caller);
        return out;
    }
};
}  // namespace
static std::string_view view_bytes(const py::bytes& bytes) {
    char* data;
    Py_ssize_t size;
    if (PyBytes_AsStringAndSize(bytes.ptr(), &data, &size) < 0)
        throw py::error_already_set();
    return {data, static_cast<size_t>(size)};
}

static std::vector<std::string_view> byte_views(const std::vector<py::bytes>& images) {
    std::vector<std::string_view> files;
    files.reserve(images.size());
    for (const auto& image : images) files.emplace_back(view_bytes(image));
    return files;
}

void register_decoder(py::module_& module) {
    py::class_<PendingDecode>(module, "_PendingDecode")
        .def("result", &PendingDecode::result, py::call_guard<py::gil_scoped_release>())
        .def("on_current_stream", &PendingDecode::on_current_stream, py::call_guard<py::gil_scoped_release>());
    py::class_<Decoder>(module, "Decoder")
        .def(py::init<int, int, c10::DeviceIndex>(), py::arg("width"), py::arg("height"),
             py::arg("device_index") = 0)
        .def(
            "decode",
            [](const Decoder& decoder, const py::bytes& bytes) {
                std::string_view file = view_bytes(bytes);
                py::gil_scoped_release release;
                return decoder.batch(std::span(&file, 1), {}).select(0, 0);
            },
            py::arg("image"))
        .def(
            "decode_batch",
            [](const Decoder& decoder, const std::vector<py::bytes>& images,
               std::optional<at::Tensor> maps) {
                auto files = byte_views(images);
                py::gil_scoped_release release;
                return decoder.batch(files, maps);
            },
            py::arg("images"), py::arg("maps") = py::none())
        .def("_enqueue", [](const Decoder& decoder, const std::vector<py::bytes>& images,
                            std::optional<at::Tensor> maps) {
            auto files = byte_views(images);
            py::gil_scoped_release release;
            auto pending = std::make_unique<PendingDecode>();
            pending->pixels = decoder.batch(files, maps, pending.get());
            return pending;
        }, py::arg("images"), py::arg("maps") = py::none());
}
