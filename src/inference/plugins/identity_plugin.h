// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Identity test plugin: copies input tensor to output unchanged.
// Used to verify IPluginV3 infrastructure compiles, registers and executes.

#pragma once

#include "flux2_plugin_base.h"

namespace fluxrt {
namespace plugins {

// ---------------------------------------------------------------------------
// IdentityPlugin
// ---------------------------------------------------------------------------
class IdentityPlugin : public Flux2PluginBase {
public:
    IdentityPlugin();

    // Clone
    IPluginV3* clone() noexcept override;

protected:
    // Build
    int32_t configurePluginImpl(
        DynamicPluginTensorDesc const* in, int32_t nbInputs,
        DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    int32_t getOutputDataTypesImpl(
        DataType* outputTypes, int32_t nbOutputs,
        DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    int32_t getOutputShapesImpl(
        DimsExprs const* inputs, int32_t nbInputs,
        DimsExprs const* shapeInputs, int32_t nbShapeInputs,
        DimsExprs* outputs, int32_t nbOutputs,
        IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombinationImpl(
        int32_t pos, DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override;

    int32_t getNbOutputs() const noexcept override { return 1; }

    // Runtime
    int32_t onShapeChangeImpl(
        PluginTensorDesc const* in, int32_t nbInputs,
        PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    int32_t enqueueImpl(
        PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override;

    // Serialization
    PluginFieldCollection const* getFieldsToSerializeImpl() noexcept override;

private:
    PluginFieldCollection serializeFields_{};
};

// ---------------------------------------------------------------------------
// IdentityPluginCreator
// ---------------------------------------------------------------------------
class IdentityPluginCreator : public Flux2PluginCreatorBase {
public:
    IdentityPluginCreator();
    IPluginV3* createPlugin(
        char const* name, PluginFieldCollection const* fc,
        TensorRTPhase phase) noexcept override;
};

}  // namespace plugins
}  // namespace fluxrt
