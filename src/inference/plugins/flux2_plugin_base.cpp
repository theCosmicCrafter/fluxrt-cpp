// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0

#include "flux2_plugin_base.h"

namespace fluxrt {
namespace plugins {

// ---------------------------------------------------------------------------
// Flux2PluginBase
// ---------------------------------------------------------------------------

Flux2PluginBase::Flux2PluginBase(std::string name, std::string version, std::string ns)
    : pluginName_(std::move(name))
    , pluginVersion_(std::move(version))
    , pluginNamespace_(std::move(ns))
{}

IPluginCapability* Flux2PluginBase::getCapabilityInterface(PluginCapabilityType type) noexcept {
    switch (type) {
        case PluginCapabilityType::kCORE:    return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD:   return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
    }
    return nullptr;
}

AsciiChar const* Flux2PluginBase::getPluginName() const noexcept {
    return pluginName_.c_str();
}

AsciiChar const* Flux2PluginBase::getPluginVersion() const noexcept {
    return pluginVersion_.c_str();
}

AsciiChar const* Flux2PluginBase::getPluginNamespace() const noexcept {
    return pluginNamespace_.c_str();
}

int32_t Flux2PluginBase::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs,
    DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    inDescCache_.resize(nbInputs);
    outDescCache_.resize(nbOutputs);
    for (int32_t i = 0; i < nbInputs; ++i) inDescCache_[i] = in[i];
    for (int32_t i = 0; i < nbOutputs; ++i) outDescCache_[i] = out[i];
    return configurePluginImpl(in, nbInputs, out, nbOutputs);
}

int32_t Flux2PluginBase::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs,
    DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    return getOutputDataTypesImpl(outputTypes, nbOutputs, inputTypes, nbInputs);
}

int32_t Flux2PluginBase::getOutputShapes(
    DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* shapeInputs, int32_t nbShapeInputs,
    DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept
{
    return getOutputShapesImpl(inputs, nbInputs, shapeInputs, nbShapeInputs,
                               outputs, nbOutputs, exprBuilder);
}

bool Flux2PluginBase::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut,
    int32_t nbInputs, int32_t nbOutputs) noexcept
{
    return supportsFormatCombinationImpl(pos, inOut, nbInputs, nbOutputs);
}

int32_t Flux2PluginBase::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs,
    PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    return onShapeChangeImpl(in, nbInputs, out, nbOutputs);
}

int32_t Flux2PluginBase::enqueue(
    PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept
{
    return enqueueImpl(inputDesc, outputDesc, inputs, outputs, workspace, stream);
}

PluginFieldCollection const* Flux2PluginBase::getFieldsToSerialize() noexcept {
    return getFieldsToSerializeImpl();
}

// ---------------------------------------------------------------------------
// Flux2PluginCreatorBase
// ---------------------------------------------------------------------------

Flux2PluginCreatorBase::Flux2PluginCreatorBase(std::string name, std::string version, std::string ns)
    : pluginName_(std::move(name))
    , pluginVersion_(std::move(version))
    , pluginNamespace_(std::move(ns))
{
    fields_.nbFields = 0;
    fields_.fields = nullptr;
}

AsciiChar const* Flux2PluginCreatorBase::getPluginName() const noexcept {
    return pluginName_.c_str();
}

AsciiChar const* Flux2PluginCreatorBase::getPluginVersion() const noexcept {
    return pluginVersion_.c_str();
}

AsciiChar const* Flux2PluginCreatorBase::getPluginNamespace() const noexcept {
    return pluginNamespace_.c_str();
}

PluginFieldCollection const* Flux2PluginCreatorBase::getFieldNames() noexcept {
    fields_.nbFields = static_cast<int32_t>(fieldVec_.size());
    fields_.fields = fieldVec_.data();
    return &fields_;
}

}  // namespace plugins
}  // namespace fluxrt
