// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Host-side declarations for shared block-plugin CUDA kernels.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace fluxrt {
namespace plugins {

// GELU in-place
void launch_gelu(float* data, int32_t n, cudaStream_t stream);

// Bias add (broadcast bias over (seq_len, dim))
void launch_bias_add(float* data, const float* bias,
                      int32_t seq_len, int32_t dim, cudaStream_t stream);

// RMSNorm  (y = x / sqrt(mean(x^2)+eps) * weight)
void launch_rms_norm(const float* x, float* y,
                        const float* weight, float eps,
                        int32_t seq_len, int32_t dim, cudaStream_t stream);

// Element-wise add: z = x + y
void launch_elem_add(const float* x, const float* y, float* z,
                      int32_t n, cudaStream_t stream);

// Scale-and-add: out = x + gate * y
void launch_scale_add(const float* x, const float* y, float* out,
                        float gate, int32_t n, cudaStream_t stream);

}  // namespace plugins
}  // namespace fluxrt
