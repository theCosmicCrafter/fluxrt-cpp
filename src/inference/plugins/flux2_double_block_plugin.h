// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0
//
// IPluginV3 for a Flux2 double-stream transformer block.
//
// Inputs (dense mode):
//   [0] img_hidden_states  : (B, img_seq_len, inner_dim)
//   [1] txt_hidden_states  : (B, txt_seq_len, inner_dim)
//   [2] img_temb           : (B, 3*inner_dim)   AdaLN modulation params for img
//   [3] txt_temb           : (B, 3*inner_dim)   AdaLN modulation params for txt
//
// Outputs:
//   [0] img_out : (B, img_seq_len, inner_dim)
//   [1] txt_out : (B, txt_seq_len, inner_dim)

#pragma once

#include "flux2_plugin_base.h"
#include "weight_loader.h"

#include <NvInferRuntime.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fluxrt {
namespace plugins {

class Flux2DoubleBlockPlugin : public Flux2PluginBase {
public:
    // Constructor used by the creator.
    // weight_dir: directory containing per-tensor .bin files for this block.
    Flux2DoubleBlockPlugin(int32_t layer_idx,
                             int32_t inner_dim,
                             int32_t num_heads,
                             int32_t head_dim,
                             int32_t mlp_hidden_dim,
                             const std::string& weight_dir);

    ~Flux2DoubleBlockPlugin();

    // IPluginV3 (clone)
    IPluginV3* clone() noexcept override;

    // Core
    AsciiChar const* getPluginName() const noexcept override;
    AsciiChar const* getPluginVersion() const noexcept override;
    AsciiChar const* getPluginNamespace() const noexcept override;

    // Build
    int32_t getOutputDataTypesImpl(DataType* outputTypes, int32_t nbOutputs,
                                   DataType const* inputTypes,
                                   int32_t nbInputs) const noexcept override;

    int32_t getOutputShapesImpl(DimsExprs const* inputs, int32_t nbInputs,
                                DimsExprs const* shapeInputs,
                                int32_t nbShapeInputs,
                                DimsExprs* outputs, int32_t nbOutputs,
                                IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombinationImpl(
        int32_t pos, DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override;

    int32_t configurePluginImpl(
        DynamicPluginTensorDesc const* in, int32_t nbInputs,
        DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    int32_t getNbOutputs() const noexcept override { return 2; }

    // Serialization
    PluginFieldCollection const* getFieldsToSerializeImpl() noexcept override;

    // Runtime
    int32_t onShapeChangeImpl(
        PluginTensorDesc const* in, int32_t nbInputs,
        PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    int32_t enqueueImpl(
        PluginTensorDesc const* inputDesc,
        PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override;

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override;

    PluginFieldCollection const* getFieldsToSerialize() noexcept override;

private:
    // Block hyperparameters
    int32_t layer_idx_;
    int32_t inner_dim_;
    int32_t num_heads_;
    int32_t head_dim_;
    int32_t mlp_hidden_dim_;

    // Weight directory (used during creation; not needed at runtime after load)
    std::string weight_dir_;

    // Loaded device weights
    struct Weights {
        // img attention
        float* img_attn_to_q = nullptr;      size_t img_attn_to_q_sz = 0;
        float* img_attn_to_q_bias = nullptr; size_t img_attn_to_q_bias_sz = 0;
        float* img_attn_to_k = nullptr;      size_t img_attn_to_k_sz = 0;
        float* img_attn_to_k_bias = nullptr; size_t img_attn_to_k_bias_sz = 0;
        float* img_attn_to_v = nullptr;      size_t img_attn_to_v_sz = 0;
        float* img_attn_to_v_bias = nullptr; size_t img_attn_to_v_bias_sz = 0;
        float* img_attn_to_out = nullptr;    size_t img_attn_to_out_sz = 0;
        float* img_attn_to_out_bias = nullptr; size_t img_attn_to_out_bias_sz = 0;
        float* img_attn_norm_q = nullptr;    size_t img_attn_norm_q_sz = 0;
        float* img_attn_norm_k = nullptr;    size_t img_attn_norm_k_sz = 0;

        // txt attention
        float* txt_attn_to_q = nullptr;      size_t txt_attn_to_q_sz = 0;
        float* txt_attn_to_q_bias = nullptr; size_t txt_attn_to_q_bias_sz = 0;
        float* txt_attn_to_k = nullptr;      size_t txt_attn_to_k_sz = 0;
        float* txt_attn_to_k_bias = nullptr; size_t txt_attn_to_k_bias_sz = 0;
        float* txt_attn_to_v = nullptr;      size_t txt_attn_to_v_sz = 0;
        float* txt_attn_to_v_bias = nullptr; size_t txt_attn_to_v_bias_sz = 0;
        float* txt_attn_to_out = nullptr;    size_t txt_attn_to_out_sz = 0;
        float* txt_attn_to_out_bias = nullptr; size_t txt_attn_to_out_bias_sz = 0;
        float* txt_attn_norm_q = nullptr;    size_t txt_attn_norm_q_sz = 0;
        float* txt_attn_norm_k = nullptr;    size_t txt_attn_norm_k_sz = 0;

        // img MLP
        float* img_mlp_0 = nullptr; size_t img_mlp_0_sz = 0;
        float* img_mlp_0_bias = nullptr; size_t img_mlp_0_bias_sz = 0;
        float* img_mlp_2 = nullptr; size_t img_mlp_2_sz = 0;
        float* img_mlp_2_bias = nullptr; size_t img_mlp_2_bias_sz = 0;

        // txt MLP
        float* txt_mlp_0 = nullptr; size_t txt_mlp_0_sz = 0;
        float* txt_mlp_0_bias = nullptr; size_t txt_mlp_0_bias_sz = 0;
        float* txt_mlp_2 = nullptr; size_t txt_mlp_2_sz = 0;
        float* txt_mlp_2_bias = nullptr; size_t txt_mlp_2_bias_sz = 0;
    } w_;

    bool load_weights();

    // cuBLAS handle (created lazily on first enqueue)
    cublasHandle_t cublas_handle_ = nullptr;
    bool init_cublas();
    void destroy_cublas();

    // Forward pass helpers
    bool dense_forward(const float* img_in, const float* txt_in,
                       const float* img_temb, const float* txt_temb,
                       float* img_out, float* txt_out,
                       int32_t B, int32_t img_len, int32_t txt_len,
                       void* workspace, size_t workspace_bytes,
                       cudaStream_t stream);

    // Workspace sizing
    size_t compute_dense_workspace(int32_t B, int32_t img_len, int32_t txt_len) const;
};

// ---------------------------------------------------------------------------
// Plugin creator
// ---------------------------------------------------------------------------
class Flux2DoubleBlockPluginCreator : public Flux2PluginCreatorBase {
public:
    AsciiChar const* getPluginName() const noexcept override;
    AsciiChar const* getPluginVersion() const noexcept override;
    IPluginV3* createPlugin(AsciiChar const* name,
                            PluginFieldCollection const* fc,
                            TensorRTPhase phase) noexcept override;
};

}  // namespace plugins
}  // namespace fluxrt
