// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0
//
// CUDA kernels for gather / scatter of active tokens during sparse execution.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace fluxrt {
namespace plugins {

// ---------------------------------------------------------------------------
// Gather active tokens from a full-sequence tensor.
//
//   src   : (seq_len, dim)   — full sequence
//   dst   : (active_count, dim) — output
//   indices : (active_count,)  — list of active token positions in [0, seq_len)
// ---------------------------------------------------------------------------
void gather_tokens(const float* src, float* dst,
                   const int32_t* indices,
                   int32_t active_count, int32_t dim,
                   cudaStream_t stream);

// ---------------------------------------------------------------------------
// Scatter active tokens back into a full-sequence tensor.
//
//   src   : (active_count, dim) — gathered results
//   dst   : (seq_len, dim)   — full sequence (written to active positions only)
//   indices : (active_count,)  — list of active token positions
// ---------------------------------------------------------------------------
void scatter_tokens(const float* src, float* dst,
                    const int32_t* indices,
                    int32_t active_count, int32_t dim,
                    cudaStream_t stream);

// ---------------------------------------------------------------------------
// Apply AdaLN-style modulation:  out = (1 + scale) * x + shift
//
//   x     : (seq_len, dim)
//   scale : (dim,)  broadcast
//   shift : (dim,)  broadcast
//   out   : (seq_len, dim)
// ---------------------------------------------------------------------------
void apply_modulation(const float* x, const float* scale, const float* shift,
                      float* out, int32_t seq_len, int32_t dim,
                      cudaStream_t stream);

// ---------------------------------------------------------------------------
// LayerNorm (elementwise_affine = false) + optional modulation.
// Computes: y = (x - mean) / sqrt(var + eps) * scale + shift
// If scale/shift are nullptr, applies standard LayerNorm.
//
//   x     : (seq_len, dim)
//   y     : (seq_len, dim)
//   gamma : (dim,) or nullptr
//   beta  : (dim,) or nullptr
// ---------------------------------------------------------------------------
void layer_norm_modulated(const float* x, float* y,
                            const float* gamma, const float* beta,
                            float eps, int32_t seq_len, int32_t dim,
                            cudaStream_t stream);

}  // namespace plugins
}  // namespace fluxrt
