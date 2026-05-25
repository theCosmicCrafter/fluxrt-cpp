// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0
//
// CUDA kernels shared by the block plugins (GELU, bias-add, RMSNorm).

#include "flux2_kernels.h"

namespace fluxrt {
namespace plugins {

// ---------------------------------------------------------------------------
// GELU
// ---------------------------------------------------------------------------
__global__ void gelu_kernel(float* __restrict__ data, int32_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = data[idx];
    float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    data[idx] = x * cdf;
}

void launch_gelu(float* data, int32_t n, cudaStream_t stream)
{
    if (n <= 0) return;
    int block = 256;
    int grid  = (n + block - 1) / block;
    gelu_kernel<<<grid, block, 0, stream>>>(data, n);
}

// ---------------------------------------------------------------------------
// Bias add  (broadcast bias of size dim over (seq_len, dim))
// ---------------------------------------------------------------------------
__global__ void bias_add_kernel(float* __restrict__ data,
                                 const float* __restrict__ bias,
                                 int32_t seq_len, int32_t dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * dim;
    if (idx >= total) return;
    int d = idx % dim;
    data[idx] += bias[d];
}

void launch_bias_add(float* data, const float* bias,
                      int32_t seq_len, int32_t dim, cudaStream_t stream)
{
    if (!bias || seq_len == 0 || dim == 0) return;
    int total = seq_len * dim;
    int block = 256;
    int grid  = (total + block - 1) / block;
    bias_add_kernel<<<grid, block, 0, stream>>>(data, bias, seq_len, dim);
}

// ---------------------------------------------------------------------------
// RMSNorm  (y = x / sqrt(mean(x^2)+eps) * weight)
// ---------------------------------------------------------------------------
__global__ void rms_norm_kernel(const float* __restrict__ x,
                                 float* __restrict__ y,
                                 const float* __restrict__ weight,
                                 float eps, int32_t dim)
{
    int token = blockIdx.x;
    const float* x_row = x + token * dim;
    float* y_row = y + token * dim;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        ss += x_row[i] * x_row[i];

    for (int offset = 16; offset > 0; offset /= 2)
        ss += __shfl_down_sync(0xFFFFFFFF, ss, offset);

    __shared__ float shared_ss;
    if ((threadIdx.x & 31) == 0) atomicAdd(&shared_ss, ss);
    __syncthreads();
    ss = rsqrtf(shared_ss / dim + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        y_row[i] = x_row[i] * ss * weight[i];
}

void launch_rms_norm(const float* x, float* y,
                        const float* weight, float eps,
                        int32_t seq_len, int32_t dim, cudaStream_t stream)
{
    if (seq_len == 0 || dim == 0) return;
    rms_norm_kernel<<<seq_len, 256, 0, stream>>>(x, y, weight, eps, dim);
}

// ---------------------------------------------------------------------------
// Element-wise add  (z = x + y)
// ---------------------------------------------------------------------------
__global__ void elem_add_kernel(const float* __restrict__ x,
                                 const float* __restrict__ y,
                                 float* __restrict__ z, int32_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    z[idx] = x[idx] + y[idx];
}

void launch_elem_add(const float* x, const float* y, float* z,
                      int32_t n, cudaStream_t stream)
{
    if (n <= 0) return;
    int block = 256;
    int grid  = (n + block - 1) / block;
    elem_add_kernel<<<grid, block, 0, stream>>>(x, y, z, n);
}

// ---------------------------------------------------------------------------
// Scale-and-add residual  (out = x + gate * y)
// ---------------------------------------------------------------------------
__global__ void scale_add_kernel(const float* __restrict__ x,
                                   const float* __restrict__ y,
                                   float* __restrict__ out,
                                   float gate, int32_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = x[idx] + gate * y[idx];
}

void launch_scale_add(const float* x, const float* y, float* out,
                        float gate, int32_t n, cudaStream_t stream)
{
    if (n <= 0) return;
    int block = 256;
    int grid  = (n + block - 1) / block;
    scale_add_kernel<<<grid, block, 0, stream>>>(x, y, out, gate, n);
}

}  // namespace plugins
}  // namespace fluxrt
