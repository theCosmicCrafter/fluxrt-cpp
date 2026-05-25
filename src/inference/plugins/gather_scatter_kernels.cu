// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0

#include "gather_scatter_kernels.h"

#include <cuda_runtime.h>
#include <cmath>

namespace fluxrt {
namespace plugins {

// ---------------------------------------------------------------------------
// Gather tokens
// ---------------------------------------------------------------------------
__global__ void gather_tokens_kernel(const float* __restrict__ src,
                                      float* __restrict__ dst,
                                      const int32_t* __restrict__ indices,
                                      int32_t active_count, int32_t dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = active_count * dim;
    if (idx >= total) return;

    int token_idx = idx / dim;
    int feat_idx  = idx % dim;

    int32_t src_token = indices[token_idx];
    dst[idx] = src[src_token * dim + feat_idx];
}

void gather_tokens(const float* src, float* dst,
                   const int32_t* indices,
                   int32_t active_count, int32_t dim,
                   cudaStream_t stream)
{
    int total = active_count * dim;
    if (total == 0) return;
    int block = 256;
    int grid  = (total + block - 1) / block;
    gather_tokens_kernel<<<grid, block, 0, stream>>>(src, dst, indices, active_count, dim);
}

// ---------------------------------------------------------------------------
// Scatter tokens
// ---------------------------------------------------------------------------
__global__ void scatter_tokens_kernel(const float* __restrict__ src,
                                      float* __restrict__ dst,
                                      const int32_t* __restrict__ indices,
                                      int32_t active_count, int32_t dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = active_count * dim;
    if (idx >= total) return;

    int token_idx = idx / dim;
    int feat_idx  = idx % dim;

    int32_t dst_token = indices[token_idx];
    dst[dst_token * dim + feat_idx] = src[idx];
}

void scatter_tokens(const float* src, float* dst,
                    const int32_t* indices,
                    int32_t active_count, int32_t dim,
                    cudaStream_t stream)
{
    int total = active_count * dim;
    if (total == 0) return;
    int block = 256;
    int grid  = (total + block - 1) / block;
    scatter_tokens_kernel<<<grid, block, 0, stream>>>(src, dst, indices, active_count, dim);
}

// ---------------------------------------------------------------------------
// Modulation: out = (1 + scale) * x + shift
// ---------------------------------------------------------------------------
__global__ void apply_modulation_kernel(const float* __restrict__ x,
                                         const float* __restrict__ scale,
                                         const float* __restrict__ shift,
                                         float* __restrict__ out,
                                         int32_t seq_len, int32_t dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * dim;
    if (idx >= total) return;

    int feat = idx % dim;
    out[idx] = (1.0f + scale[feat]) * x[idx] + shift[feat];
}

void apply_modulation(const float* x, const float* scale, const float* shift,
                      float* out, int32_t seq_len, int32_t dim,
                      cudaStream_t stream)
{
    int total = seq_len * dim;
    if (total == 0) return;
    int block = 256;
    int grid  = (total + block - 1) / block;
    apply_modulation_kernel<<<grid, block, 0, stream>>>(x, scale, shift, out, seq_len, dim);
}

// ---------------------------------------------------------------------------
// LayerNorm (elementwise_affine = false) + optional gamma/beta
// Manual block reduction via shared memory — no CUB dependency.
// ---------------------------------------------------------------------------
__global__ void layer_norm_modulated_kernel(const float* __restrict__ x,
                                              float* __restrict__ y,
                                              const float* __restrict__ gamma,
                                              const float* __restrict__ beta,
                                              float eps, int32_t dim)
{
    int token = blockIdx.x;
    const float* x_row = x + token * dim;
    float* y_row = y + token * dim;

    // --- warp reduce helper ------------------------------------------------
    auto warp_reduce_sum = [](float val) {
        for (int offset = 16; offset > 0; offset /= 2)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        return val;
    };

    // --- compute mean ------------------------------------------------------
    float mean = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        mean += x_row[i];

    mean = warp_reduce_sum(mean);
    __shared__ float shared_mean;
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&shared_mean, mean);
    }
    __syncthreads();
    mean = shared_mean / dim;
    __syncthreads();

    // --- compute variance --------------------------------------------------
    float var = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float d = x_row[i] - mean;
        var += d * d;
    }

    var = warp_reduce_sum(var);
    __shared__ float shared_var;
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&shared_var, var);
    }
    __syncthreads();
    var = rsqrtf(shared_var / dim + eps);
    __syncthreads();

    // --- normalize + affine ------------------------------------------------
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = (x_row[i] - mean) * var;
        if (gamma) val *= gamma[i];
        if (beta)  val += beta[i];
        y_row[i] = val;
    }
}

void layer_norm_modulated(const float* x, float* y,
                            const float* gamma, const float* beta,
                            float eps, int32_t seq_len, int32_t dim,
                            cudaStream_t stream)
{
    if (seq_len == 0 || dim == 0) return;
    layer_norm_modulated_kernel<<<seq_len, 256, 0, stream>>>(x, y, gamma, beta, eps, dim);
}

}  // namespace plugins
}  // namespace fluxrt
