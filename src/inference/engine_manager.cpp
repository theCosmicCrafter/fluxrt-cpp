// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine_manager.h"
#include "../utils/error.h"

#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

namespace fluxrt {

// Placed at file scope to avoid static initialization order issues inside the constructor.
namespace {
struct TrtLogger : nvinfer1::ILogger {
    void log(nvinfer1::ILogger::Severity severity,
             const char* msg) noexcept override {
        // Only log warnings and errors to avoid spam.
        if (severity <= nvinfer1::ILogger::Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << "\n";
        }
    }
};
TrtLogger trt_logger;
}  // namespace

EngineManager::EngineManager(const std::string& plan_path) {
    std::ifstream file(plan_path, std::ios::binary);
    FLUXRT_ASSERT(file.is_open(), ("Cannot open plan: " + plan_path).c_str());

    file.seekg(0, std::ios::end);
    std::size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), static_cast<std::streamsize>(size));
    file.close();

    runtime_.reset(nvinfer1::createInferRuntime(trt_logger));
    FLUXRT_ASSERT(runtime_, "createInferRuntime failed");

    engine_.reset(runtime_->deserializeCudaEngine(buffer.data(), size));
    TRT_CHECK(engine_);

    context_.reset(engine_->createExecutionContext());
    TRT_CHECK(context_);

    CUDA_CHECK(cudaStreamCreate(&stream_));
}

EngineManager::~EngineManager() {
    if (stream_) cudaStreamDestroy(stream_);
}

EngineManager::EngineManager(EngineManager&& other) noexcept
    : runtime_(std::move(other.runtime_)),
      engine_(std::move(other.engine_)),
      context_(std::move(other.context_)),
      stream_(other.stream_) {
    other.stream_ = nullptr;
}

EngineManager& EngineManager::operator=(EngineManager&& other) noexcept {
    if (this != &other) {
        if (stream_) cudaStreamDestroy(stream_);
        runtime_ = std::move(other.runtime_);
        engine_ = std::move(other.engine_);
        context_ = std::move(other.context_);
        stream_ = other.stream_;
        other.stream_ = nullptr;
    }
    return *this;
}

int EngineManager::num_io_tensors() const {
    return engine_->getNbIOTensors();
}

const char* EngineManager::tensor_name(int idx) const {
    return engine_->getIOTensorName(idx);
}

std::vector<std::size_t> EngineManager::tensor_shape(const char* name) const {
    // For fixed-shape engines, engine_->getTensorShape is safe.
    // Use context_->getTensorShape only after setInputShape has been called
    // for all dynamic inputs.
    auto dims = engine_->getTensorShape(name);
    std::vector<std::size_t> shape;
    for (int i = 0; i < dims.nbDims; ++i) {
        FLUXRT_ASSERT(dims.d[i] >= 0, ("Unresolved dynamic shape for tensor: " + std::string(name)).c_str());
        shape.push_back(static_cast<std::size_t>(dims.d[i]));
    }
    return shape;
}

std::size_t EngineManager::elem_size(const char* name) const {
    auto dtype = engine_->getTensorDataType(name);
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT:  return 4;
        case nvinfer1::DataType::kHALF:   return 2;
        case nvinfer1::DataType::kINT8:   return 1;
        case nvinfer1::DataType::kINT32:  return 4;
        case nvinfer1::DataType::kINT64:  return 8;
        case nvinfer1::DataType::kBOOL:   return 1;
        case nvinfer1::DataType::kUINT8:  return 1;
        case nvinfer1::DataType::kFP8:    return 1;
        case nvinfer1::DataType::kBF16:   return 2;
        default:                          return 4;
    }
}

std::vector<Tensor> EngineManager::allocate_buffers() const {
    std::vector<Tensor> buffers;
    int n = num_io_tensors();
    for (int i = 0; i < n; ++i) {
        const char* name = tensor_name(i);
        auto shape = tensor_shape(name);
        std::size_t es = elem_size(name);
        buffers.emplace_back(shape, es);
    }
    return buffers;
}

void EngineManager::infer(const std::vector<Tensor>& buffers) {
    int n = num_io_tensors();
    FLUXRT_ASSERT(static_cast<int>(buffers.size()) == n,
                  "buffer count mismatch");
    for (int i = 0; i < n; ++i) {
        const char* name = tensor_name(i);
        TRT_CHECK(context_->setTensorAddress(name, buffers[i].d_ptr));
    }
    TRT_CHECK(context_->enqueueV3(stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void EngineManager::set_input_shape(const char* name,
                                    const std::vector<std::size_t>& shape) {
    nvinfer1::Dims dims{};
    dims.nbDims = static_cast<int>(shape.size());
    for (std::size_t i = 0; i < shape.size(); ++i) {
        dims.d[i] = static_cast<int32_t>(shape[i]);
    }
    TRT_CHECK(context_->setInputShape(name, dims));
}

}  // namespace fluxrt
