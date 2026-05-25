// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

namespace fluxrt {
namespace kernels {

class SpatialCache {
public:
    SpatialCache(int image_seq_len, int text_seq_len, int output_channels,
                 int num_attention_heads, int attention_head_dim,
                 int num_double_layers, int num_single_layers);
    ~SpatialCache();

    // Prevent copying
    SpatialCache(const SpatialCache&) = delete;
    SpatialCache& operator=(const SpatialCache&) = delete;

    // Reset all validity flags to false (e.g. for a new timestep).
    // Does NOT clear the cached values themselves.
    void reset(cudaStream_t stream = nullptr);

    void preprocess_mask(int32_t* d_mask, cudaStream_t stream = nullptr);

    void sync_output_cache(float* d_prediction, const int32_t* d_mask, cudaStream_t stream = nullptr);

    void sync_kv_cache(float* d_keys, float* d_values,
                       const int32_t* d_mask, int block_id, bool is_double_block,
                       cudaStream_t stream = nullptr);

private:
    int image_seq_len_;
    int text_seq_len_;
    int full_seq_len_;
    int output_channels_;
    int num_heads_;
    int head_dim_;
    int num_double_layers_;
    int num_single_layers_;

    bool* d_valid_ = nullptr;
    float* d_output_cache_ = nullptr;

    std::vector<float*> d_double_keys_;
    std::vector<float*> d_double_values_;
    std::vector<float*> d_single_keys_;
    std::vector<float*> d_single_values_;

    void allocate_buffers();
    void free_buffers();
};

} // namespace kernels
} // namespace fluxrt
