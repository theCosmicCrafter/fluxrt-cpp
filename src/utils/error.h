// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Error-checking macros for CUDA and TensorRT calls.
// Per CONSTITUTION.md Article V: every CUDA / TRT call must be wrapped.

#pragma once

#include <cstdio>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace fluxrt {

// ----- Exception type used for unrecoverable runtime errors --------------
class RuntimeError : public std::runtime_error {
  public:
    explicit RuntimeError(const std::string& msg) : std::runtime_error(msg) {}
};

// ----- CUDA error check ---------------------------------------------------
// Use as: CUDA_CHECK(cudaMalloc(&ptr, n));
// On failure throws RuntimeError with file:line and CUDA error string.

inline void cuda_check_impl(cudaError_t err, const char* expr,
                            const char* file, int line) {
    if (err != cudaSuccess) {
        char buf[1024];
        std::snprintf(buf, sizeof(buf),
                      "[CUDA] %s failed at %s:%d -> %s (%d): %s",
                      expr, file, line,
                      cudaGetErrorName(err), static_cast<int>(err),
                      cudaGetErrorString(err));
        throw RuntimeError(buf);
    }
}

#define CUDA_CHECK(expr) ::fluxrt::cuda_check_impl((expr), #expr, __FILE__, __LINE__)

// ----- CUDA last-error check (for kernel launches) -----------------------
// Use as: CUDA_CHECK_LAUNCH("my_kernel"); after a <<<>>> launch.
// First peeks at last error (sync-free), then synchronizes to catch async errors.

inline void cuda_check_launch_impl(const char* kernel_name,
                                   const char* file, int line) {
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        char buf[1024];
        std::snprintf(buf, sizeof(buf),
                      "[CUDA-LAUNCH] kernel '%s' launch failed at %s:%d -> %s: %s",
                      kernel_name, file, line,
                      cudaGetErrorName(err), cudaGetErrorString(err));
        throw RuntimeError(buf);
    }
}

#define CUDA_CHECK_LAUNCH(name) \
    ::fluxrt::cuda_check_launch_impl((name), __FILE__, __LINE__)

// ----- TensorRT error check ----------------------------------------------
// Wraps a bool/non-null TensorRT call. Use as:
//   TRT_CHECK(builder->buildSerializedNetwork(...));
// Falsy values (false / null pointer) throw with file:line.
//
// We don't include NvInfer.h here to keep this header lightweight; the
// macro just requires the expression to be contextually convertible to bool.

#define TRT_CHECK(expr)                                                        \
    do {                                                                       \
        if (!(expr)) {                                                         \
            char trt_buf[1024];                                                \
            std::snprintf(trt_buf, sizeof(trt_buf),                            \
                          "[TRT] %s returned false/null at %s:%d",             \
                          #expr, __FILE__, __LINE__);                          \
            throw ::fluxrt::RuntimeError(trt_buf);                             \
        }                                                                      \
    } while (0)

// ----- Generic assertion (use sparingly) ---------------------------------
#define FLUXRT_ASSERT(cond, msg)                                               \
    do {                                                                       \
        if (!(cond)) {                                                         \
            char a_buf[1024];                                                  \
            std::snprintf(a_buf, sizeof(a_buf),                                \
                          "[ASSERT] (%s) failed at %s:%d: %s",                 \
                          #cond, __FILE__, __LINE__, (msg));                   \
            throw ::fluxrt::RuntimeError(a_buf);                               \
        }                                                                      \
    } while (0)

}  // namespace fluxrt
