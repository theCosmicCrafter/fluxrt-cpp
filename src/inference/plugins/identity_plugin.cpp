// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0

#include "identity_plugin.h"
#include "../../utils/error.h"

#include <cstring>

namespace fluxrt {
namespace plugins {

// ---------------------------------------------------------------------------
// IdentityPlugin
// ---------------------------------------------------------------------------

IdentityPlugin::IdentityPlugin()
    : Flux2PluginBase("IdentityPlugin", "1", "fluxrt")
{}

IPluginV3* IdentityPlugin::clone() noexcept {
    return new IdentityPlugin();
}

int32_t IdentityPlugin::configurePluginImpl(
    DynamicPluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    DynamicPluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept
{
    return 0;
}

int32_t IdentityPlugin::getOutputDataTypesImpl(
    DataType* outputTypes, int32_t nbOutputs,
    DataType const* inputTypes, int32_t /*nbInputs*/) const noexcept
{
    for (int32_t i = 0; i < nbOutputs; ++i) {
        outputTypes[i] = inputTypes[i];
    }
    return 0;
}

int32_t IdentityPlugin::getOutputShapesImpl(
    DimsExprs const* inputs, int32_t /*nbInputs*/,
    DimsExprs const* /*shapeInputs*/, int32_t /*nbShapeInputs*/,
    DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& /*exprBuilder*/) noexcept
{
    for (int32_t i = 0; i < nbOutputs; ++i) {
        outputs[i] = inputs[i];
    }
    return 0;
}

bool IdentityPlugin::supportsFormatCombinationImpl(
    int32_t /*pos*/, DynamicPluginTensorDesc const* /*inOut*/,
    int32_t /*nbInputs*/, int32_t /*nbOutputs*/) noexcept
{
    return true;
}

int32_t IdentityPlugin::onShapeChangeImpl(
    PluginTensorDesc const* /*in*/, int32_t /*nbInputs*/,
    PluginTensorDesc const* /*out*/, int32_t /*nbOutputs*/) noexcept
{
    return 0;
}

int32_t IdentityPlugin::enqueueImpl(
    PluginTensorDesc const* inputDesc, PluginTensorDesc const* /*outputDesc*/,
    void const* const* inputs, void* const* outputs,
    void* /*workspace*/, cudaStream_t stream) noexcept
{
    size_t nbytes = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; ++i) {
        nbytes *= inputDesc[0].dims.d[i];
    }
    nbytes *= sizeof(float);  // identity plugin assumes float

    cudaError_t err = cudaMemcpyAsync(outputs[0], inputs[0], nbytes, cudaMemcpyDeviceToDevice, stream);
    return (err == cudaSuccess) ? 0 : -1;
}

PluginFieldCollection const* IdentityPlugin::getFieldsToSerializeImpl() noexcept {
    serializeFields_.nbFields = 0;
    serializeFields_.fields = nullptr;
    return &serializeFields_;
}

// ---------------------------------------------------------------------------
// IdentityPluginCreator
// ---------------------------------------------------------------------------

IdentityPluginCreator::IdentityPluginCreator()
    : Flux2PluginCreatorBase("IdentityPlugin", "1", "fluxrt")
{}

IPluginV3* IdentityPluginCreator::createPlugin(
    char const* /*name*/, PluginFieldCollection const* /*fc*/,
    TensorRTPhase /*phase*/) noexcept
{
    return new IdentityPlugin();
}

}  // namespace plugins
}  // namespace fluxrt
