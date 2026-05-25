// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0

#include "spatial_kv_cache.h"
#include "../../utils/error.h"

#include <iostream>

namespace fluxrt {
namespace kernels {

__global__ void preprocess_mask_kernel(int32_t* mask, const bool* valid, int full_seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < full_seq_len) {
        if (!valid[idx]) {
            mask[idx] = 2; // execute and update if currently invalid
        }
    }
}

__global__ void sync_output_cache_kernel(
    float* prediction,
    float* output_cache,
    const int32_t* mask,
    bool* valid,
    int text_seq_len,
    int image_seq_len,
    int output_channels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // token index
    int ch = blockIdx.y * blockDim.y + threadIdx.y;  // channel index

    if (idx < image_seq_len && ch < output_channels) {
        int mask_idx = text_seq_len + idx;
        int32_t m = mask[mask_idx];
        bool execute = (m != 0);
        bool update = (m == 2);

        int flat_idx = idx * output_channels + ch;
        float pred_val = prediction[flat_idx];
        float cache_val = output_cache[flat_idx];

        prediction[flat_idx] = execute ? pred_val : cache_val;

        if (update) {
            output_cache[flat_idx] = pred_val;
        }

        if (ch == 0 && update) {
            valid[mask_idx] = true;
        }
    }
}

__global__ void sync_kv_cache_kernel(
    float* keys,
    float* values,
    float* cached_keys,
    float* cached_values,
    const int32_t* mask,
    int full_seq_len,
    int head_dim_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int elem = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < full_seq_len && elem < head_dim_size) {
        int32_t m = mask[idx];
        bool execute = (m != 0);
        bool update = (m == 2);

        int flat_idx = idx * head_dim_size + elem;

        float k_val = keys[flat_idx];
        float v_val = values[flat_idx];
        float c_k_val = cached_keys[flat_idx];
        float c_v_val = cached_values[flat_idx];

        keys[flat_idx] = execute ? k_val : c_k_val;
        values[flat_idx] = execute ? v_val : c_v_val;

        if (update) {
            cached_keys[flat_idx] = k_val;
            cached_values[flat_idx] = v_val;
        }
    }
}

SpatialCache::SpatialCache(int image_seq_len, int text_seq_len, int output_channels,
                           int num_attention_heads, int attention_head_dim,
                           int num_double_layers, int num_single_layers)
    : image_seq_len_(image_seq_len), text_seq_len_(text_seq_len),
      full_seq_len_(text_seq_len + image_seq_len), output_channels_(output_channels),
      num_heads_(num_attention_heads), head_dim_(attention_head_dim),
      num_double_layers_(num_double_layers), num_single_layers_(num_single_layers) {
    allocate_buffers();
}

SpatialCache::~SpatialCache() {
    free_buffers();
}

void SpatialCache::allocate_buffers() {
    CUDA_CHECK(cudaMalloc(&d_valid_, full_seq_len_ * sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_valid_, 0, full_seq_len_ * sizeof(bool)));

    size_t out_cache_size = image_seq_len_ * output_channels_ * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_output_cache_, out_cache_size));
    CUDA_CHECK(cudaMemset(d_output_cache_, 0, out_cache_size));

    size_t kv_cache_size = full_seq_len_ * num_heads_ * head_dim_ * sizeof(float);
    d_double_keys_.resize(num_double_layers_);
    d_double_values_.resize(num_double_layers_);
    for (int i = 0; i < num_double_layers_; ++i) {
        CUDA_CHECK(cudaMalloc(&d_double_keys_[i], kv_cache_size));
        CUDA_CHECK(cudaMalloc(&d_double_values_[i], kv_cache_size));
        CUDA_CHECK(cudaMemset(d_double_keys_[i], 0, kv_cache_size));
        CUDA_CHECK(cudaMemset(d_double_values_[i], 0, kv_cache_size));
    }
    d_single_keys_.resize(num_single_layers_);
    d_single_values_.resize(num_single_layers_);
    for (int i = 0; i < num_single_layers_; ++i) {
        CUDA_CHECK(cudaMalloc(&d_single_keys_[i], kv_cache_size));
        CUDA_CHECK(cudaMalloc(&d_single_values_[i], kv_cache_size));
        CUDA_CHECK(cudaMemset(d_single_keys_[i], 0, kv_cache_size));
        CUDA_CHECK(cudaMemset(d_single_values_[i], 0, kv_cache_size));
    }
}

void SpatialCache::free_buffers() {
    // Report errors but don't throw from destructor path.
    auto check_free = [](void* ptr, const char* name) {
        if (ptr) {
            cudaError_t err = cudaFree(ptr);
            if (err != cudaSuccess) {
                std::cerr << "[SpatialCache] cudaFree(" << name << ") failed: "
                          << cudaGetErrorString(err) << "\n";
            }
        }
    };
    check_free(d_valid_, "d_valid_");
    check_free(d_output_cache_, "d_output_cache_");
    for (auto ptr : d_double_keys_) check_free(ptr, "d_double_keys_");
    for (auto ptr : d_double_values_) check_free(ptr, "d_double_values_");
    for (auto ptr : d_single_keys_) check_free(ptr, "d_single_keys_");
    for (auto ptr : d_single_values_) check_free(ptr, "d_single_values_");
}

void SpatialCache::reset(cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(d_valid_, 0, full_seq_len_ * sizeof(bool), stream));
}

void SpatialCache::preprocess_mask(int32_t* d_mask, cudaStream_t stream) {
    int threads = 256;
    int blocks = (full_seq_len_ + threads - 1) / threads;
    preprocess_mask_kernel<<<blocks, threads, 0, stream>>>(d_mask, d_valid_, full_seq_len_);
    CUDA_CHECK(cudaGetLastError());
}

void SpatialCache::sync_output_cache(float* d_prediction, const int32_t* d_mask, cudaStream_t stream) {
    dim3 threads(32, 32);
    dim3 blocks((image_seq_len_ + threads.x - 1) / threads.x,
                (output_channels_ + threads.y - 1) / threads.y);
    sync_output_cache_kernel<<<blocks, threads, 0, stream>>>(
        d_prediction, d_output_cache_, d_mask, d_valid_,
        text_seq_len_, image_seq_len_, output_channels_
    );
    CUDA_CHECK(cudaGetLastError());
}

void SpatialCache::sync_kv_cache(float* d_keys, float* d_values,
                                 const int32_t* d_mask, int block_id, bool is_double_block,
                                 cudaStream_t stream) {
    int head_dim_size = num_heads_ * head_dim_;
    dim3 threads(16, 64);
    dim3 blocks((full_seq_len_ + threads.x - 1) / threads.x,
                (head_dim_size + threads.y - 1) / threads.y);

    float* cached_keys = is_double_block ? d_double_keys_[block_id] : d_single_keys_[block_id];
    float* cached_values = is_double_block ? d_double_values_[block_id] : d_single_values_[block_id];

    sync_kv_cache_kernel<<<blocks, threads, 0, stream>>>(
        d_keys, d_values, cached_keys, cached_values, d_mask,
        full_seq_len_, head_dim_size
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels
} // namespace fluxrt
