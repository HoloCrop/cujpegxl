from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent.resolve()
kernels = ['device_support', 'entropy', 'frame_encoder', 'frontend_fused', 'frontend_quantize', 'prefix_build']
bitstream = ['ans', 'codestream', 'container', 'context_map', 'frame_assembly', 'histogram_writer', 'prefix_code', 'token_stream']
sources = [
    'torch_cujpegxl/encoder.cpp', 'torch_cujpegxl/decoder.cpp', 'torch_cujpegxl/decode.cu',
    'src/cujpegxl.cc', 'src/coeff_order.cc', 'src/worker_pool.cc',
    *[f'src/{name}.cu' for name in kernels],
    *[f'src/bitstream/{name}.cc' for name in bitstream],
]
extension = CUDAExtension(
    'torch_cujpegxl._native',
    sources,
    include_dirs=[str(root), str(root / 'include')],
    extra_compile_args={'cxx': ['-O3', '-g0'], 'nvcc': ['-O3', '--extended-lambda', '--expt-relaxed-constexpr']},
)
class CameraBuild(BuildExtension):
    def build_extensions(self):
        objects = self.compiler.compile(
            ['src/camera_coefficients.c'], output_dir=self.build_temp,
            extra_postargs=['-O3', '-g0', '-std=gnu11', '-fvisibility=hidden'],
        )
        self.extensions[0].extra_objects.extend(objects)
        super().build_extensions()

setup(ext_modules=[extension], cmdclass={'build_ext': CameraBuild})
