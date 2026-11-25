from pathlib import Path
import subprocess

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
    extra_link_args=['-fopenmp'],
    include_dirs=[str(root), str(root / 'include')],
    extra_compile_args={'cxx': ['-O3', '-g0', '-fopenmp'], 'nvcc': ['-O3', '--extended-lambda', '--expt-relaxed-constexpr']},
)
class CameraBuild(BuildExtension):
    def build_extensions(self):
        directory = Path(self.build_temp).resolve() / 'camera_coefficients'
        subprocess.run([
            'cmake', '-S', str(root / 'third_party/jxldec'), '-B', str(directory),
            '-DCMAKE_BUILD_TYPE=Release',
        ], check=True)
        subprocess.run(['cmake', '--build', str(directory), '--target', 'camera_coefficients'], check=True)
        self.extensions[0].extra_objects.append(str(directory / 'libcamera_coefficients.a'))
        super().build_extensions()

setup(ext_modules=[extension], cmdclass={'build_ext': CameraBuild})
