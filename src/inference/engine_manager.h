// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Minimal TensorRT engine loader + inference wrapper.

#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

#include "../utils/tensor.h"

namespace fluxrt {


// Wraps a deserialized TRT engine and execution context.
class EngineManager {
  public:
    explicit EngineManager(const std::string& plan_path);
    ~EngineManager();

    // Disable copy; allow move
    EngineManager(const EngineManager&) = delete;
    EngineManager& operator=(const EngineManager&) = delete;
    EngineManager(EngineManager&&) noexcept;
    EngineManager& operator=(EngineManager&&) noexcept;

    // Number of I/O tensors
    int num_io_tensors() const;

    // Tensor name by index
    const char* tensor_name(int idx) const;

    // Shape for a given tensor name (returns dims)
    std::vector<std::size_t> tensor_shape(const char* name) const;

    // Element size in bytes for a tensor's dtype
    std::size_t elem_size(const char* name) const;

    // Allocate device buffers for all I/O tensors
    std::vector<Tensor> allocate_buffers() const;

    // Run inference (assumes buffers are already populated on device)
    void infer(const std::vector<Tensor>& buffers);

    // Helper: set input shapes and allocate buffers for dynamic shapes
    void set_input_shape(const char* name, const std::vector<std::size_t>& shape);

  private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_ = nullptr;
};

}  // namespace fluxrt
