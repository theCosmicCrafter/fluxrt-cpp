// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Host-side wrappers for attention CUDA kernels.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace fluxrt {
namespace plugins {

// Scaled dot-product attention.
//
//   Q : (batch, num_heads, q_len, head_dim)
//   K : (batch, num_heads, kv_len, head_dim)
//   V : (batch, num_heads, kv_len, head_dim)
//   out : (batch, num_heads, q_len, head_dim)
void attention_fwd(const float* Q, const float* K, const float* V,
                   float* out,
                   int32_t batch, int32_t num_heads,
                   int32_t q_len, int32_t kv_len, int32_t head_dim,
                   cudaStream_t stream);

}  // namespace plugins
}  // namespace fluxrt
