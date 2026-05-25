// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0

#include "flux2_double_block_plugin.h"
#include "gather_scatter_kernels.h"
#include "attention_kernels.h"
#include "flux2_kernels.h"

#include <NvInferRuntime.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>

namespace fluxrt {
namespace plugins {

using namespace nvinfer1;

// ---------------------------------------------------------------------------
// cuBLAS GEMM wrapper  (row-major PyTorch style)
//  Y = X @ W^T + b
//  X : (M, K)  W : (N, K)  Y : (M, N)
// ---------------------------------------------------------------------------
static inline cublasStatus_t gemm_row_major(cublasHandle_t handle,
                                             const float* W, const float* X,
                                             float* Y,
                                             int M, int N, int K,
                                             float alpha, float beta,
                                             cudaStream_t stream)
{
    cublasSetStream(handle, stream);
    return cublasSgemm(handle,
                       CUBLAS_OP_T, CUBLAS_OP_N,
                       N, M, K,
                       &alpha,
                       W, K,
                       X, K,
                       &beta,
                       Y, N);
}

// ---------------------------------------------------------------------------
// Constructor / clone / destructor
// ---------------------------------------------------------------------------
Flux2DoubleBlockPlugin::Flux2DoubleBlockPlugin(
    int32_t layer_idx, int32_t inner_dim,
    int32_t num_heads, int32_t head_dim,
    int32_t mlp_hidden_dim,
    const std::string& weight_dir)
    : Flux2PluginBase("Flux2DoubleBlockPlugin", "1", "fluxrt")
    , layer_idx_(layer_idx)
    , inner_dim_(inner_dim)
    , num_heads_(num_heads)
    , head_dim_(head_dim)
    , mlp_hidden_dim_(mlp_hidden_dim)
    , weight_dir_(weight_dir)
{
    load_weights();
}

Flux2DoubleBlockPlugin::~Flux2DoubleBlockPlugin()
{
    destroy_cublas();
}

IPluginV3* Flux2DoubleBlockPlugin::clone() noexcept
{
    auto* p = new Flux2DoubleBlockPlugin(
        layer_idx_, inner_dim_, num_heads_, head_dim_, mlp_hidden_dim_, weight_dir_);
    return p;
}

bool Flux2DoubleBlockPlugin::init_cublas()
{
    if (cublas_handle_) return true;
    cublasStatus_t st = cublasCreate(&cublas_handle_);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[Flux2DoubleBlockPlugin] cublasCreate failed\n";
        return false;
    }
    return true;
}

void Flux2DoubleBlockPlugin::destroy_cublas()
{
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
        cublas_handle_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Core
// ---------------------------------------------------------------------------
AsciiChar const* Flux2DoubleBlockPlugin::getPluginName() const noexcept
{
    return "Flux2DoubleBlockPlugin";
}

AsciiChar const* Flux2DoubleBlockPlugin::getPluginVersion() const noexcept
{
    return "1";
}

AsciiChar const* Flux2DoubleBlockPlugin::getPluginNamespace() const noexcept
{
    return "fluxrt";
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------
int32_t Flux2DoubleBlockPlugin::getOutputDataTypesImpl(
    DataType* outputTypes, int32_t nbOutputs,
    DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (nbOutputs != 2) return -1;
    outputTypes[0] = inputTypes[0];  // img float
    outputTypes[1] = inputTypes[1];  // txt float
    return 0;
}

int32_t Flux2DoubleBlockPlugin::getOutputShapesImpl(
    DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* /*shapeInputs*/, int32_t /*nbShapeInputs*/,
    DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& /*exprBuilder*/) noexcept
{
    if (nbInputs < 4 || nbOutputs != 2) return -1;
    outputs[0] = inputs[0];  // img shape
    outputs[1] = inputs[1];  // txt shape
    return 0;
}

bool Flux2DoubleBlockPlugin::supportsFormatCombinationImpl(
    int32_t /*pos*/, DynamicPluginTensorDesc const* inOut,
    int32_t /*nbInputs*/, int32_t /*nbOutputs*/) noexcept
{
    return inOut[0].desc.type == DataType::kFLOAT &&
           inOut[0].desc.format == TensorFormat::kLINEAR;
}

int32_t Flux2DoubleBlockPlugin::configurePluginImpl(
    DynamicPluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept
{
    return 0;
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------
static PluginFieldCollection s_empty_fc{0, nullptr};

PluginFieldCollection const* Flux2DoubleBlockPlugin::getFieldsToSerializeImpl() noexcept
{
    return &s_empty_fc;
}

// ---------------------------------------------------------------------------
// Workspace sizing
// ---------------------------------------------------------------------------
size_t Flux2DoubleBlockPlugin::compute_dense_workspace(
    int32_t B, int32_t img_len, int32_t txt_len) const
{
    size_t img_elems = static_cast<size_t>(B) * img_len * inner_dim_;
    size_t txt_elems = static_cast<size_t>(B) * txt_len * inner_dim_;
    size_t img_mlp   = static_cast<size_t>(B) * img_len * mlp_hidden_dim_;
    size_t txt_mlp   = static_cast<size_t>(B) * txt_len * mlp_hidden_dim_;

    // We need: img_norm, txt_norm, img_Q, img_K, img_V, txt_Q, txt_K, txt_V,
    //          img_attn_out, txt_attn_out, img_mlp_mid, txt_mlp_mid,
    //          img_mlp_out, txt_mlp_out
    size_t total = 0;
    total += img_elems + txt_elems;           // norm buffers
    total += 3 * img_elems + 3 * txt_elems;   // QKV
    total += img_elems + txt_elems;           // attn_out
    total += img_mlp + txt_mlp;               // mlp_mid
    total += img_elems + txt_elems;           // mlp_out
    return total * sizeof(float);
}

// ---------------------------------------------------------------------------
// Runtime: onShapeChange
// ---------------------------------------------------------------------------
int32_t Flux2DoubleBlockPlugin::onShapeChangeImpl(
    PluginTensorDesc const* in, int32_t /*nbInputs*/,
    PluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept
{
    // Cache shapes if needed
    (void)in;
    return 0;
}

// ---------------------------------------------------------------------------
// Runtime: enqueue
// ---------------------------------------------------------------------------
int32_t Flux2DoubleBlockPlugin::enqueueImpl(
    PluginTensorDesc const* inputDesc,
    PluginTensorDesc const* /*outputDesc*/,
    void const* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept
{
    if (!init_cublas()) return -1;

    int32_t B = inputDesc[0].dims.d[0];
    int32_t img_len = inputDesc[0].dims.d[1];
    int32_t txt_len = inputDesc[1].dims.d[1];

    const float* img_in  = static_cast<const float*>(inputs[0]);
    const float* txt_in  = static_cast<const float*>(inputs[1]);
    const float* img_temb = static_cast<const float*>(inputs[2]);
    const float* txt_temb = static_cast<const float*>(inputs[3]);

    float* img_out = static_cast<float*>(outputs[0]);
    float* txt_out = static_cast<float*>(outputs[1]);

    size_t ws_needed = compute_dense_workspace(B, img_len, txt_len);
    // TensorRT guarantees workspace >= getWorkspaceSize(), so we trust it.
    (void)ws_needed;

    bool ok = dense_forward(img_in, txt_in, img_temb, txt_temb,
                            img_out, txt_out,
                            B, img_len, txt_len,
                            workspace, ws_needed, stream);
    return ok ? 0 : -1;
}

// ---------------------------------------------------------------------------
// attachToContext
// ---------------------------------------------------------------------------
IPluginV3* Flux2DoubleBlockPlugin::attachToContext(IPluginResourceContext* /*context*/) noexcept
{
    return clone();
}

// ---------------------------------------------------------------------------
// getFieldsToSerialize
// ---------------------------------------------------------------------------
PluginFieldCollection const* Flux2DoubleBlockPlugin::getFieldsToSerialize() noexcept
{
    static PluginField fields[6];
    static PluginFieldCollection fc{6, fields};
    // Serialize hyperparameters so engine can recreate the plugin
    // In a full implementation we'd allocate and fill these properly.
    return &fc;
}

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------
static bool load_weight_file(const std::string& path, float** d_ptr, size_t* sz)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return false;
    std::streamsize bytes = file.tellg();
    file.seekg(0, std::ios::beg);
    *sz = bytes / sizeof(float);
    cudaMalloc(d_ptr, bytes);
    std::vector<float> host(*sz);
    file.read(reinterpret_cast<char*>(host.data()), bytes);
    cudaMemcpy(*d_ptr, host.data(), bytes, cudaMemcpyHostToDevice);
    return true;
}

bool Flux2DoubleBlockPlugin::load_weights()
{
    auto try_load = [&](const char* name, float** d_ptr, size_t* sz) {
        std::string path = weight_dir_ + "/" + name + ".bin";
        if (!load_weight_file(path, d_ptr, sz)) {
            std::cerr << "[Flux2DoubleBlockPlugin] Missing weight: " << path << "\n";
            return false;
        }
        return true;
    };

    // img attention
    if (!try_load("img_attn_to_q",     &w_.img_attn_to_q,     &w_.img_attn_to_q_sz))     return false;
    if (!try_load("img_attn_to_q_bias", &w_.img_attn_to_q_bias, &w_.img_attn_to_q_bias_sz)) { w_.img_attn_to_q_bias = nullptr; w_.img_attn_to_q_bias_sz = 0; }
    if (!try_load("img_attn_to_k",     &w_.img_attn_to_k,     &w_.img_attn_to_k_sz))     return false;
    if (!try_load("img_attn_to_k_bias", &w_.img_attn_to_k_bias, &w_.img_attn_to_k_bias_sz)) { w_.img_attn_to_k_bias = nullptr; }
    if (!try_load("img_attn_to_v",     &w_.img_attn_to_v,     &w_.img_attn_to_v_sz))     return false;
    if (!try_load("img_attn_to_v_bias", &w_.img_attn_to_v_bias, &w_.img_attn_to_v_bias_sz)) { w_.img_attn_to_v_bias = nullptr; }
    if (!try_load("img_attn_to_out",   &w_.img_attn_to_out,   &w_.img_attn_to_out_sz))   return false;
    if (!try_load("img_attn_to_out_bias", &w_.img_attn_to_out_bias, &w_.img_attn_to_out_bias_sz)) { w_.img_attn_to_out_bias = nullptr; }
    try_load("img_attn_norm_q", &w_.img_attn_norm_q, &w_.img_attn_norm_q_sz);
    try_load("img_attn_norm_k", &w_.img_attn_norm_k, &w_.img_attn_norm_k_sz);

    // txt attention
    if (!try_load("txt_attn_to_q",     &w_.txt_attn_to_q,     &w_.txt_attn_to_q_sz))     return false;
    if (!try_load("txt_attn_to_q_bias", &w_.txt_attn_to_q_bias, &w_.txt_attn_to_q_bias_sz)) { w_.txt_attn_to_q_bias = nullptr; }
    if (!try_load("txt_attn_to_k",     &w_.txt_attn_to_k,     &w_.txt_attn_to_k_sz))     return false;
    if (!try_load("txt_attn_to_k_bias", &w_.txt_attn_to_k_bias, &w_.txt_attn_to_k_bias_sz)) { w_.txt_attn_to_k_bias = nullptr; }
    if (!try_load("txt_attn_to_v",     &w_.txt_attn_to_v,     &w_.txt_attn_to_v_sz))     return false;
    if (!try_load("txt_attn_to_v_bias", &w_.txt_attn_to_v_bias, &w_.txt_attn_to_v_bias_sz)) { w_.txt_attn_to_v_bias = nullptr; }
    if (!try_load("txt_attn_to_out",   &w_.txt_attn_to_out,   &w_.txt_attn_to_out_sz))   return false;
    if (!try_load("txt_attn_to_out_bias", &w_.txt_attn_to_out_bias, &w_.txt_attn_to_out_bias_sz)) { w_.txt_attn_to_out_bias = nullptr; }
    try_load("txt_attn_norm_q", &w_.txt_attn_norm_q, &w_.txt_attn_norm_q_sz);
    try_load("txt_attn_norm_k", &w_.txt_attn_norm_k, &w_.txt_attn_norm_k_sz);

    // img MLP
    if (!try_load("img_mlp_0",     &w_.img_mlp_0,     &w_.img_mlp_0_sz))     return false;
    if (!try_load("img_mlp_0_bias", &w_.img_mlp_0_bias, &w_.img_mlp_0_bias_sz)) { w_.img_mlp_0_bias = nullptr; }
    if (!try_load("img_mlp_2",     &w_.img_mlp_2,     &w_.img_mlp_2_sz))     return false;
    if (!try_load("img_mlp_2_bias", &w_.img_mlp_2_bias, &w_.img_mlp_2_bias_sz)) { w_.img_mlp_2_bias = nullptr; }

    // txt MLP
    if (!try_load("txt_mlp_0",     &w_.txt_mlp_0,     &w_.txt_mlp_0_sz))     return false;
    if (!try_load("txt_mlp_0_bias", &w_.txt_mlp_0_bias, &w_.txt_mlp_0_bias_sz)) { w_.txt_mlp_0_bias = nullptr; }
    if (!try_load("txt_mlp_2",     &w_.txt_mlp_2,     &w_.txt_mlp_2_sz))     return false;
    if (!try_load("txt_mlp_2_bias", &w_.txt_mlp_2_bias, &w_.txt_mlp_2_bias_sz)) { w_.txt_mlp_2_bias = nullptr; }

    return true;
}

// ---------------------------------------------------------------------------
// Dense forward pass
// ---------------------------------------------------------------------------
bool Flux2DoubleBlockPlugin::dense_forward(
    const float* img_in, const float* txt_in,
    const float* img_temb, const float* txt_temb,
    float* img_out, float* txt_out,
    int32_t B, int32_t img_len, int32_t txt_len,
    void* workspace, size_t /*workspace_bytes*/,
    cudaStream_t stream)
{
    // Workspace layout (all float*):
    // [0] img_norm  : B*img_len*D
    // [1] txt_norm  : B*txt_len*D
    // [2] img_Q     : B*img_len*D
    // [3] img_K     : B*img_len*D
    // [4] img_V     : B*img_len*D
    // [5] txt_Q     : B*txt_len*D
    // [6] txt_K     : B*txt_len*D
    // [7] txt_V     : B*txt_len*D
    // [8] img_attn  : B*img_len*D
    // [9] txt_attn  : B*txt_len*D
    // [10] img_mlp_mid : B*img_len*mlp_hidden
    // [11] txt_mlp_mid : B*txt_len*mlp_hidden
    // [12] img_mlp_out : B*img_len*D
    // [13] txt_mlp_out : B*txt_len*D

    float* ws = static_cast<float*>(workspace);
    size_t img_elems = static_cast<size_t>(B) * img_len * inner_dim_;
    size_t txt_elems = static_cast<size_t>(B) * txt_len * inner_dim_;
    size_t img_mlp   = static_cast<size_t>(B) * img_len * mlp_hidden_dim_;
    size_t txt_mlp   = static_cast<size_t>(B) * txt_len * mlp_hidden_dim_;

    float* img_norm = ws;
    float* txt_norm = img_norm + img_elems;
    float* img_Q    = txt_norm + txt_elems;
    float* img_K    = img_Q + img_elems;
    float* img_V    = img_K + img_elems;
    float* txt_Q    = img_V + img_elems;
    float* txt_K    = txt_Q + txt_elems;
    float* txt_V    = txt_K + txt_elems;
    float* img_attn = txt_V + txt_elems;
    float* txt_attn = img_attn + img_elems;
    float* img_mlp_mid = txt_attn + txt_elems;
    float* txt_mlp_mid = img_mlp_mid + img_mlp;
    float* img_mlp_out = txt_mlp_mid + txt_mlp;
    float* txt_mlp_out = img_mlp_out + img_elems;

    // Copy inputs to outputs (for residual addition later)
    cudaMemcpyAsync(img_out, img_in, img_elems * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(txt_out, txt_in, txt_elems * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // --- LayerNorm (elementwise_affine = false) ------------------------
    layer_norm_modulated(img_in,  img_norm, nullptr, nullptr, 1e-6f, B * img_len, inner_dim_, stream);
    layer_norm_modulated(txt_in,  txt_norm, nullptr, nullptr, 1e-6f, B * txt_len, inner_dim_, stream);

    // --- Modulation: out = (1 + scale) * x + shift ----------------------
    // temb layout: [shift, scale, gate] each of size inner_dim
    // For simplicity we assume batch=1 and temb is pre-computed per token
    // In practice temb is (B, 3*inner_dim) and we broadcast
    apply_modulation(img_norm, img_temb + inner_dim_, img_temb,
                     img_norm, B * img_len, inner_dim_, stream);
    apply_modulation(txt_norm, txt_temb + inner_dim_, txt_temb,
                     txt_norm, B * txt_len, inner_dim_, stream);

    // --- QKV projections (cuBLAS GEMM) ---------------------------------
    float alpha = 1.0f, beta = 0.0f;
    auto gemm = [&](const float* W, const float* X, float* Y, int M, int N, int K) {
        return gemm_row_major(cublas_handle_, W, X, Y, M, N, K, alpha, beta, stream);
    };

    // img QKV
    gemm(w_.img_attn_to_q, img_norm, img_Q, B * img_len, inner_dim_, inner_dim_);
    gemm(w_.img_attn_to_k, img_norm, img_K, B * img_len, inner_dim_, inner_dim_);
    gemm(w_.img_attn_to_v, img_norm, img_V, B * img_len, inner_dim_, inner_dim_);
    // txt QKV
    gemm(w_.txt_attn_to_q, txt_norm, txt_Q, B * txt_len, inner_dim_, inner_dim_);
    gemm(w_.txt_attn_to_k, txt_norm, txt_K, B * txt_len, inner_dim_, inner_dim_);
    gemm(w_.txt_attn_to_v, txt_norm, txt_V, B * txt_len, inner_dim_, inner_dim_);

    // Add biases if present
    auto add_bias = [&](float* data, const float* bias, int32_t seq_len) {
        launch_bias_add(data, bias, B * seq_len, inner_dim_, stream);
    };
    add_bias(img_Q, w_.img_attn_to_q_bias, img_len);
    add_bias(img_K, w_.img_attn_to_k_bias, img_len);
    add_bias(img_V, w_.img_attn_to_v_bias, img_len);
    add_bias(txt_Q, w_.txt_attn_to_q_bias, txt_len);
    add_bias(txt_K, w_.txt_attn_to_k_bias, txt_len);
    add_bias(txt_V, w_.txt_attn_to_v_bias, txt_len);

    // --- QK RMSNorm -----------------------------------------------------
    if (w_.img_attn_norm_q) {
        launch_rms_norm(img_Q, img_Q, w_.img_attn_norm_q, 1e-6f, B * img_len, inner_dim_, stream);
        launch_rms_norm(img_K, img_K, w_.img_attn_norm_k, 1e-6f, B * img_len, inner_dim_, stream);
    }
    if (w_.txt_attn_norm_q) {
        launch_rms_norm(txt_Q, txt_Q, w_.txt_attn_norm_q, 1e-6f, B * txt_len, inner_dim_, stream);
        launch_rms_norm(txt_K, txt_K, w_.txt_attn_norm_k, 1e-6f, B * txt_len, inner_dim_, stream);
    }

    // --- Attention ------------------------------------------------------
    // Reshape to (B, num_heads, seq_len, head_dim)
    attention_fwd(img_Q, img_K, img_V, img_attn,
                  B, num_heads_, img_len, img_len, head_dim_, stream);
    attention_fwd(txt_Q, txt_K, txt_V, txt_attn,
                  B, num_heads_, txt_len, txt_len, head_dim_, stream);

    // --- Output projection ----------------------------------------------
    gemm(w_.img_attn_to_out, img_attn, img_attn, B * img_len, inner_dim_, inner_dim_);
    gemm(w_.txt_attn_to_out, txt_attn, txt_attn, B * txt_len, inner_dim_, inner_dim_);
    add_bias(img_attn, w_.img_attn_to_out_bias, img_len);
    add_bias(txt_attn, w_.txt_attn_to_out_bias, txt_len);

    // --- MLP ------------------------------------------------------------
    // img MLP: linear1 -> GELU -> linear2
    gemm(w_.img_mlp_0, img_norm, img_mlp_mid, B * img_len, mlp_hidden_dim_, inner_dim_);
    add_bias(img_mlp_mid, w_.img_mlp_0_bias, img_len);
    launch_gelu(img_mlp_mid, B * img_len * mlp_hidden_dim_, stream);
    gemm(w_.img_mlp_2, img_mlp_mid, img_mlp_out, B * img_len, inner_dim_, mlp_hidden_dim_);
    add_bias(img_mlp_out, w_.img_mlp_2_bias, img_len);

    // txt MLP
    gemm(w_.txt_mlp_0, txt_norm, txt_mlp_mid, B * txt_len, mlp_hidden_dim_, inner_dim_);
    add_bias(txt_mlp_mid, w_.txt_mlp_0_bias, txt_len);
    launch_gelu(txt_mlp_mid, B * txt_len * mlp_hidden_dim_, stream);
    gemm(w_.txt_mlp_2, txt_mlp_mid, txt_mlp_out, B * txt_len, inner_dim_, mlp_hidden_dim_);
    add_bias(txt_mlp_out, w_.txt_mlp_2_bias, txt_len);

    // --- Add residuals (img_out = img_in + gate*(img_attn + img_mlp_out))
    // For now gate=1.0; TODO: read gate from temb
    int total_img = B * img_len * inner_dim_;
    int total_txt = B * txt_len * inner_dim_;
    // img_attn += img_mlp_out  (accumulate into img_attn workspace)
    launch_elem_add(img_attn, img_mlp_out, img_attn, total_img, stream);
    launch_elem_add(txt_attn, txt_mlp_out, txt_attn, total_txt, stream);
    // img_out = img_in + 1.0 * img_attn
    launch_scale_add(img_in, img_attn, img_out, 1.0f, total_img, stream);
    launch_scale_add(txt_in, txt_attn, txt_out, 1.0f, total_txt, stream);

    return true;
}

// ---------------------------------------------------------------------------
// Creator
// ---------------------------------------------------------------------------
AsciiChar const* Flux2DoubleBlockPluginCreator::getPluginName() const noexcept
{
    return "Flux2DoubleBlockPlugin";
}

AsciiChar const* Flux2DoubleBlockPluginCreator::getPluginVersion() const noexcept
{
    return "1";
}

IPluginV3* Flux2DoubleBlockPluginCreator::createPlugin(
    AsciiChar const* /*name*/,
    PluginFieldCollection const* fc,
    TensorRTPhase /*phase*/) noexcept
{
    // Parse fields: layer_idx, inner_dim, num_heads, head_dim, mlp_hidden_dim, weight_dir
    int32_t layer_idx = 0, inner_dim = 3072, num_heads = 24, head_dim = 128, mlp_hidden_dim = 9216;
    std::string weight_dir;

    for (int i = 0; i < fc->nbFields; ++i) {
        const auto& f = fc->fields[i];
        if (strcmp(f.name, "layer_idx") == 0) layer_idx = *static_cast<const int32_t*>(f.data);
        else if (strcmp(f.name, "inner_dim") == 0) inner_dim = *static_cast<const int32_t*>(f.data);
        else if (strcmp(f.name, "num_heads") == 0) num_heads = *static_cast<const int32_t*>(f.data);
        else if (strcmp(f.name, "head_dim") == 0) head_dim = *static_cast<const int32_t*>(f.data);
        else if (strcmp(f.name, "mlp_hidden_dim") == 0) mlp_hidden_dim = *static_cast<const int32_t*>(f.data);
        else if (strcmp(f.name, "weight_dir") == 0) weight_dir = std::string(static_cast<const char*>(f.data), f.length);
    }

    return new Flux2DoubleBlockPlugin(layer_idx, inner_dim, num_heads, head_dim, mlp_hidden_dim, weight_dir);
}

}  // namespace plugins
}  // namespace fluxrt
