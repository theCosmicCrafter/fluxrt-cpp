// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Base infrastructure for FluxRT TensorRT IPluginV3 plugins.
//
// Pattern: the plugin class inherits from IPluginV3 + the three capability
// interfaces directly.  getCapabilityInterface() returns `this` cast to the
// requested interface.  This avoids the boilerplate of separate inner-class
// objects at the cost of a single class with many methods.

#pragma once

#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>

namespace fluxrt {
namespace plugins {

using namespace nvinfer1;

// ---------------------------------------------------------------------------
// Base class for all FluxRT IPluginV3 plugins
// ---------------------------------------------------------------------------
//
// Derive from this and implement the "Impl" virtual methods.
//
class Flux2PluginBase : public IPluginV3,
                        public IPluginV3OneCore,
                        public IPluginV3OneBuild,
                        public IPluginV3OneRuntime {
public:
    // ----- IPluginV3 ------------------------------------------------------
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override;
    IPluginV3* clone() noexcept override = 0;

    // ----- IPluginV3OneCore -------------------------------------------------
    AsciiChar const* getPluginName() const noexcept override;
    AsciiChar const* getPluginVersion() const noexcept override;
    AsciiChar const* getPluginNamespace() const noexcept override;

    // ----- IPluginV3OneBuild -----------------------------------------------
    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
                            DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
                               DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
                            DimsExprs const* shapeInputs, int32_t nbShapeInputs,
                            DimsExprs* outputs, int32_t nbOutputs,
                            IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut,
                                   int32_t nbInputs, int32_t nbOutputs) noexcept override;

    int32_t getNbOutputs() const noexcept override = 0;

    int32_t getFormatCombinationLimit() noexcept override { return kDEFAULT_FORMAT_COMBINATION_LIMIT; }

    // ----- IPluginV3OneRuntime ----------------------------------------------
    int32_t setTactic(int32_t /*tactic*/) noexcept override { return 0; }

    int32_t onShapeChange(PluginTensorDesc const* in, int32_t nbInputs,
                          PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
                    void const* const* inputs, void* const* outputs,
                    void* workspace, cudaStream_t stream) noexcept override;

    IPluginV3* attachToContext(IPluginResourceContext* /*context*/) noexcept override { return clone(); }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override;

protected:
    std::string pluginName_;
    std::string pluginVersion_;
    std::string pluginNamespace_;

    // Cached shape / format info (populated by configurePlugin)
    mutable std::vector<DynamicPluginTensorDesc> inDescCache_;
    mutable std::vector<DynamicPluginTensorDesc> outDescCache_;

    Flux2PluginBase(std::string name, std::string version, std::string ns);

    // ---- Methods that derived plugins must override -----------------------

    // Build-time
    virtual int32_t configurePluginImpl(
        DynamicPluginTensorDesc const* in, int32_t nbInputs,
        DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept = 0;

    virtual int32_t getOutputDataTypesImpl(
        DataType* outputTypes, int32_t nbOutputs,
        DataType const* inputTypes, int32_t nbInputs) const noexcept = 0;

    virtual int32_t getOutputShapesImpl(
        DimsExprs const* inputs, int32_t nbInputs,
        DimsExprs const* shapeInputs, int32_t nbShapeInputs,
        DimsExprs* outputs, int32_t nbOutputs,
        IExprBuilder& exprBuilder) noexcept = 0;

    virtual bool supportsFormatCombinationImpl(
        int32_t pos, DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept = 0;

    // Runtime
    virtual int32_t onShapeChangeImpl(
        PluginTensorDesc const* in, int32_t nbInputs,
        PluginTensorDesc const* out, int32_t nbOutputs) noexcept = 0;

    virtual int32_t enqueueImpl(
        PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept = 0;

    // Serialization
    virtual PluginFieldCollection const* getFieldsToSerializeImpl() noexcept = 0;
};

// ---------------------------------------------------------------------------
// Base creator class
// ---------------------------------------------------------------------------
class Flux2PluginCreatorBase : public IPluginCreatorV3One {
public:
    AsciiChar const* getPluginName() const noexcept override;
    AsciiChar const* getPluginVersion() const noexcept override;
    AsciiChar const* getPluginNamespace() const noexcept override;
    PluginFieldCollection const* getFieldNames() noexcept override;

protected:
    std::string pluginName_;
    std::string pluginVersion_;
    std::string pluginNamespace_;
    std::vector<PluginField> fieldVec_;
    PluginFieldCollection fields_{};

    Flux2PluginCreatorBase(std::string name, std::string version, std::string ns);
};

}  // namespace plugins
}  // namespace fluxrt
