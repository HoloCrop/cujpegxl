# HoloCrop cujpegxl

This fork of [cujpegxl](https://github.com/m-pilia/cujpegxl) adds packed RGB8 input, a camera-profile CUDA decoder and synchronous and asynchronous PyTorch bindings. Each encoded image is a standalone JPEG XL file readable by standard decoders.

## Camera encoding

The HoloCrop profile was evaluated on an RTX 4070 at distance 0.36 with RGB8 renders from RAW captures. It uses horizontal 4:2:2 chroma sampling and fixed 8×8 transforms. One set of fitted coefficient thresholds is shared across the evaluated crops. The fitting objective includes coding cost, reconstruction error and RF-DETR sensitivity. The measurements apply to this profile and hardware.

Pass interleaved sRGB pixels on the GPU to `cujpegxl_encoder_submit_rgb8`. Keep an encoder per image geometry to reuse its allocations and coefficient-order history across submissions. All information required to decode an image is included in its file.

### PyTorch CUDA tensors

The `torch-cujpegxl` wheel builds the encoder and tensor binding together against PyTorch and the CUDA toolkit. `Encoder` accepts interleaved, uint8 HWC RGB on its CUDA device, including tensors with row padding. It waits on the calling thread's current PyTorch stream and returns independently owned JPEG XL files.

```python
from torch_cujpegxl import Encoder

height, width, _ = image.shape
encoder = Encoder(width, height, distance=0.36, device_index=image.device.index)
file = encoder.encode(image)
files = encoder.encode_batch(images)  # NHWC tensor or sequence of HWC tensors
```

For continuous capture, keep the encoder alive and call `push(camera_images)` for each frame. Consume the returned files in submission order and call `flush()` once at the end to receive the tail. Keep submitted pixels unchanged until their file is returned. A flush retains the encoder's scan history.

### Torch camera decoder

`Decoder(width, height, device_index=0)` accepts independent RGB8 sRGB files
using the supported single-pass 4:2:2 DCT8 camera profile. When accepting arbitrary
JPEG XL uploads, catch `CameraStreamError` and pass the original bytes to a
general JPEG XL decoder such as libjxl. CUDA execution errors propagate separately.

CPU workers parse the
headers and low-frequency data. CUDA decodes independent high-frequency ANS
groups using shared entropy tables, dequantizes the coefficients and reconstructs
the training viewport. Camera entropy models outside the shared-memory path
use a sparse CPU reader and the same GPU reconstruction.

```python
from torch_cujpegxl import Decoder

decoder = Decoder(width, height)
rgb = decoder.decode(file)                     # CUDA uint8 [height, width, 3]
training = decoder.decode_batch(files, maps)   # CUDA uint8 [batch, out_h, out_w, 3]
```

The optional contiguous CUDA float32 map stores source pixel coordinates
`(x, y)` at each output pixel, with integer coordinates at pixel centres.
Pass one `[out_h, out_w, 2]` map for a shared camera calibration or
`[batch, out_h, out_w, 2]` for individual calibrations. Remapping uses
OpenCV-style bilinear interpolation and a constant black border. Batch workers
use independent CUDA streams to overlap parsing, pinned uploads and reconstruction.
The result is ready in submission order on the caller's Torch stream. CPU parallelism
follows Torch's intra-op thread setting. Independent loader threads can also decode
one file each on their own streams. A nonblocking copy into pinned CPU memory lets a
staging writer overlap readback with the next image.

### Asynchronous Torch camera batches

Use one context-managed codec per image geometry and device. Submit on the CUDA
stream that produced the inputs. Keep input pixels and remap coordinates
unmodified until completion. Each submission returns a concurrent future.
Encoding returns camera files in input order. Decoding returns validated CUDA
pixels and registers them on the stream calling `result()`.

```python
from torch_cujpegxl import AsyncDecoder, AsyncEncoder

with AsyncEncoder(width, height, distance=0.36) as encoder:
    first = encoder.encode_batch_async(first_camera_set)
    second = encoder.encode_batch_async(second_camera_set)
    write_files(first.result())
    write_files(second.result())

with AsyncDecoder(width, height) as decoder:
    pixels = decoder.decode_batch_async(camera_files, maps)
    load_next_frame()
    consume(pixels.on_current_stream())
```

For GPU consumers, `on_current_stream()` validates the entropy and queues a CUDA
event dependency for the render. Use `result()` for host completion.

Import both classes from `torch_cujpegxl`. Two outstanding batches are allowed
by default. At capacity, submission waits for a slot. Closing drains accepted
work. Retrieve each future to observe codec errors. A failed encoder must be
recreated. The decoder can continue after a corrupt batch.

## Usage

See examples for usage in Bazel ([examples/bazel_consumer](https://github.com/m-pilia/cujpegxl/tree/master/examples/bazel_consumer)) and CMake ([examples/cmake_consumer](https://github.com/m-pilia/cujpegxl/tree/master/examples/cmake_consumer)) projects.

Python bindings are provided in [//python:pycujpegxl](https://github.com/m-pilia/cujpegxl/blob/master/python/BUILD.bazel). Currently they take host buffers as input and are meant primarily for evaluation, not for high-performance applications.

Target CUDA architecture(s) can be specified with the `--@rules_cuda//cuda:archs` flag, e.g.:

```sh
bazelisk build //... '--@rules_cuda//cuda:archs=sm_86;sm_89'
```

Example toolchain integrations for hermetic cross-compilation targeting Tegra T234 on JetPack 6.2 (CUDA 12.6) and 7.2 (CUDA 13.2) are provided for illustrative purpose:

```sh
bazelisk build --config=jetson_jp62 //packaging:install_tree
bazelisk build --config=jetson_jp72 //packaging:install_tree
```

## Evaluation tools

A small development set of freely licensed images is bundled in [//data:frames](https://github.com/m-pilia/cujpegxl/tree/master/data). See [data/LICENSE.txt](https://github.com/m-pilia/cujpegxl/blob/master/data/LICENSE.txt) for credits and respective licenses.

The `//eval_dataset` package implements tooling to generate a reproducible evaluation dataset using high-quality and free-license images, while `//tools/quality_benchmark` implements comparative compression quality benchmarking, see [eval_dataset/README.md](https://github.com/m-pilia/cujpegxl/blob/master/eval_dataset/README.md) for information.

Upstream encoder results using a 100 image subset of the evaluation dataset at 2160p resolution (`bazelisk run //eval_dataset:prepare_dataset -- --resolution 2160p --limit 100 ...`):

![ssimulacra2](doc/2026-10-29_ssimulacra2.png)

## Code checks

To run tests and various code checks:

```sh
# Including GPU tests
bazelisk test //...

# Excluding GPU tests
bazelisk test //... --test_tag_filters=-gpu

# Sanitizers
bazelisk test --config=asan //... --test_tag_filters=-gpu,-no-sanitizers
bazelisk test --config=ubsan //... --test_tag_filters=-gpu,-no-sanitizers
bazelisk test --config=tsan //... --test_tag_filters=-gpu,-no-sanitizers

# Code style
bazelisk build --config=clang-tidy //...
bazelisk run //tools/format:format.check

# Fix formatting
bazelisk run //tools/format
```

## Credits

The implementation is a derivative work of libjxl, see the License section below for details.

The source code of this project is almost entirely LLM-generated, using a mix of GLM-5.3, GLM-5.2, Qwen 3.8, GPT-5.6 Sol, and Claude Opus 4.8.

## Disclaimer

JPEG and JPEG XL are trademarks of their respective owners. cujpegxl is an independent project and is not affiliated with or endorsed by the JPEG committee.

## License

The cujpegxl software is made available under a [BSD-3-Clause license](LICENSE). The camera decoder incorporates [jxldec](third_party/jxldec/README.md), distributed under its [own license](third_party/jxldec/LICENSE.md).

This software is a [derivative work](https://en.wikipedia.org/wiki/Derivative_work) of [libjxl](https://github.com/libjxl/libjxl), the reference JPEG XL implementation, which is made available by its authors [under the same license type](https://github.com/libjxl/libjxl/blob/bdde644b94c125a15e532b2572b96306371a7d4e/LICENSE).

For additional patent rights, see the [PATENTS file in the libjxl source codebase](https://raw.githubusercontent.com/libjxl/libjxl/bdde644b94c125a15e532b2572b96306371a7d4e/PATENTS).
