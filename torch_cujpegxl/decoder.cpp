#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>

namespace py = pybind11;
extern "C" int camera_coefficients(const unsigned char*, size_t, unsigned, unsigned, float*);
void render_camera(float*, const float*, size_t, unsigned char*, int,int,int,int,int,int,cudaStream_t);

namespace {
struct Decoder {
    int width, height;
    c10::Device device;
    Decoder(int w, int h, c10::DeviceIndex index) : width(w), height(h), device(at::kCUDA,index) {
        TORCH_CHECK(w>0 && h>0 && int64_t(w+15LL)*(h+7LL) < INT_MAX/4, "invalid image dimensions");
    }

    at::Tensor batch(const std::vector<std::string>& files, std::optional<at::Tensor> maps) const {
        c10::cuda::CUDAGuard guard(device);
        TORCH_CHECK(files.size()>0 && files.size()<=65535,"expected 1 to 65535 files");
        int ow=width, oh=height;
        size_t stride=0;
        if (maps) {
            TORCH_CHECK(maps->device()==device && maps->scalar_type()==at::kFloat &&
                        maps->is_contiguous() && (maps->dim()==3 || maps->dim()==4) &&
                        maps->size(-1)==2 && (maps->dim()==3 || maps->size(0)==int64_t(files.size())),
                        "expected contiguous CUDA float32 maps [H,W,2] or [B,H,W,2]");
            TORCH_CHECK(maps->size(-2)>0 && maps->size(-3)>0 && maps->numel()/maps->size(0)<INT_MAX &&
                        maps->size(-2)*maps->size(-3)<INT_MAX-255,"invalid remap dimensions");
            ow=maps->size(-2); oh=maps->size(-3);
            if (maps->dim()==4) stride=size_t(ow)*oh;
        }
        int count=4*((width+15)/16)*((height+7)/8);
        int64_t batch_size=files.size();
        auto host=at::empty({batch_size, int64_t(count)*64},at::TensorOptions().dtype(at::kFloat).pinned_memory(true));
        auto *data=host.data_ptr<float>();
        at::parallel_for(0,batch_size,1,[&](int64_t start,int64_t end) {
            for (auto i=start;i<end;++i) {
                auto& file=files[i];
                TORCH_CHECK(camera_coefficients(reinterpret_cast<const unsigned char*>(file.data()),
                    file.size(),width,height,data+i*count*64)==0,
                    "invalid or unsupported camera JPEG XL at batch index ",i);
            }
        });
        auto blocks=host.to(device,true);
        auto out=at::empty({batch_size,oh,ow,3},blocks.options().dtype(at::kByte));
        render_camera(blocks.data_ptr<float>(),maps ? maps->data_ptr<float>() : nullptr,stride,
            out.data_ptr<uint8_t>(),batch_size,count,width,height,ow,oh,c10::cuda::getCurrentCUDAStream());
        return out;
    }
};
}
void register_decoder(py::module_& module) {
    py::class_<Decoder>(module,"Decoder")
        .def(py::init<int,int,c10::DeviceIndex>(),py::arg("width"),py::arg("height"),py::arg("device_index")=0)
        .def("decode",[](Decoder& self,const std::string& file) { return self.batch({file},{}).select(0,0); },
             py::arg("image"),py::call_guard<py::gil_scoped_release>())
        .def("decode_batch",&Decoder::batch,py::arg("images"),py::arg("maps")=py::none(),
             py::call_guard<py::gil_scoped_release>());
}
