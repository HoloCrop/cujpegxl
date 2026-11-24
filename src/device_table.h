// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// NVCC needs device storage for a constexpr array indexed at runtime. Its
// host pass still uses the same initializer as an ordinary constexpr table.
#if defined(__NVCC__) && defined(__CUDA_ARCH__)
#define CUJPEGXL_DEVICE_TABLE static __device__
#else
#define CUJPEGXL_DEVICE_TABLE
#endif
