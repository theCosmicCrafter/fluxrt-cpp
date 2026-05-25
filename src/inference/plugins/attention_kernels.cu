// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0
//
// CUDA kernels for scaled dot-product attention used inside the block plugins.
//
// Layout: (batch, seq_len, num_heads, head_dim)  →  contiguous in head_dim,
// then num_heads, then seq_len (diffusers / PyTorch convention).

#include "attention_kernels.h"

#include <cuda_runtime.h>
#include <cmath>

namespace fluxrt {
namespace plugins {

// ---------------------------------------------------------------------------
// softmax along the last dimension (attn_scores: B x H x M x N)
// ---------------------------------------------------------------------------
__global__ void softmax_kernel(const float* __restrict__ in,
                                float* __restrict__ out,
                                int32_t M, int32_t N)
{
    // One block per (batch, head, q_token)
    int b = blockIdx.z;
    int h = blockIdx.y;
    int m = blockIdx.x;

    int bh = (b * gridDim.y + h);
    int base = bh * M * N + m * N;

    // Find max for numerical stability
    float max_val = -1e30f;
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        max_val = fmaxf(max_val, in[base + n]);
    }

    // Warp reduce max
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    }
    __shared__ float shared_max;
    if ((threadIdx.x & 31) == 0) {
        atomicMax(reinterpret_cast<int*>(&shared_max), __float_as_int(max_val));
    }
    __syncthreads();
    max_val = shared_max;

    // Compute exp and sum
    float sum = 0.0f;
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        float e = expf(in[base + n] - max_val);
        out[base + n] = e;
        sum += e;
    }

    // Warp reduce sum
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    __shared__ float shared_sum;
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&shared_sum, sum);
    }
    __syncthreads();
    sum = shared_sum;

    // Normalize
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        out[base + n] /= sum;
    }
}

// ---------------------------------------------------------------------------
// Attention: out = softmax(Q @ K^T / sqrt(d_k)) @ V
//
//   Q : (B, H, M, D)
//   K : (B, H, N, D)
//   V : (B, H, N, D)
//   out : (B, H, M, D)
//
// We assume Q, K, V are already split into heads and transposed to
// (B, H, seq, D) layout.
// ---------------------------------------------------------------------------
__global__ void attention_fwd_kernel(const float* __restrict__ Q,
                                      const float* __restrict__ K,
                                      const float* __restrict__ V,
                                      float* __restrict__ out,
                                      int32_t B, int32_t H,
                                      int32_t M, int32_t N, int32_t D)
{
    // One block per (batch, head, q_token)
    int b = blockIdx.z;
    int h = blockIdx.y;
    int m = blockIdx.x;

    int bh = b * H + h;
    int q_offset = bh * M * D + m * D;
    int kv_offset = bh * N * D;

    float scale = 1.0f / sqrtf(static_cast<float>(D));

    // Compute Q @ K^T for this query token
    // scores[n] = dot(Q[m], K[n]) * scale
    extern __shared__ float s_scores[];

    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot += Q[q_offset + d] * K[kv_offset + n * D + d];
        }
        s_scores[n] = dot * scale;
    }
    __syncthreads();

    // Softmax over s_scores[0..N-1]
    float max_val = -1e30f;
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        max_val = fmaxf(max_val, s_scores[n]);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    }
    __shared__ float shared_max;
    if ((threadIdx.x & 31) == 0) shared_max = max_val;
    __syncthreads();
    max_val = shared_max;

    float sum = 0.0f;
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        float e = expf(s_scores[n] - max_val);
        s_scores[n] = e;
        sum += e;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    if ((threadIdx.x & 31) == 0) shared_max = sum;  // reuse shared_max
    __syncthreads();
    sum = shared_max;

    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        s_scores[n] /= sum;
    }
    __syncthreads();

    // Compute weighted sum: out[m,d] = sum_n score[n] * V[n,d]
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float val = 0.0f;
        for (int n = 0; n < N; ++n) {
            val += s_scores[n] * V[kv_offset + n * D + d];
        }
        out[q_offset + d] = val;
    }
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------
void attention_fwd(const float* Q, const float* K, const float* V,
                   float* out,
                   int32_t batch, int32_t num_heads,
                   int32_t q_len, int32_t kv_len, int32_t head_dim,
                   cudaStream_t stream)
{
    if (q_len == 0 || kv_len == 0 || head_dim == 0) return;

    dim3 grid(q_len, num_heads, batch);
    int threads = 256;
    int smem = kv_len * sizeof(float);
    attention_fwd_kernel<<<grid, threads, smem, stream>>>(
        Q, K, V, out, batch, num_heads, q_len, kv_len, head_dim);
}

}  // namespace plugins
}  // namespace fluxrt
