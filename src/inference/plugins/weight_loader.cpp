// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0

#include "weight_loader.h"

#include <fstream>
#include <iostream>

namespace fluxrt {
namespace plugins {

// ---------------------------------------------------------------------------
// DeviceBuffer
// ---------------------------------------------------------------------------
DeviceBuffer::DeviceBuffer(size_t n) : numel(n) {
    if (n > 0) {
        cudaMalloc(&data, n * sizeof(float));
    }
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : data(other.data), numel(other.numel) {
    other.data = nullptr;
    other.numel = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        if (data) cudaFree(data);
        data = other.data;
        numel = other.numel;
        other.data = nullptr;
        other.numel = 0;
    }
    return *this;
}

DeviceBuffer::~DeviceBuffer() {
    if (data) cudaFree(data);
}

bool DeviceBuffer::load_from_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "[WeightLoader] Failed to open: " << path << "\n";
        return false;
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t expected = numel * sizeof(float);
    if (static_cast<size_t>(size) != expected) {
        std::cerr << "[WeightLoader] Size mismatch for " << path
                  << ": expected " << expected << " got " << size << "\n";
        return false;
    }

    std::vector<float> host(numel);
    if (!file.read(reinterpret_cast<char*>(host.data()), size)) {
        std::cerr << "[WeightLoader] Failed to read: " << path << "\n";
        return false;
    }

    cudaMemcpy(data, host.data(), size, cudaMemcpyHostToDevice);
    return true;
}

// ---------------------------------------------------------------------------
// WeightLoader
// ---------------------------------------------------------------------------
WeightLoader::WeightLoader(const std::string& dir) {
    // In a real implementation we would enumerate the directory.
    // For now we leave the map empty; callers will populate it.
    (void)dir;
}

const float* WeightLoader::get(const std::string& name) const {
    auto it = weights_.find(name);
    if (it == weights_.end()) return nullptr;
    return it->second.data;
}

size_t WeightLoader::size(const std::string& name) const {
    auto it = weights_.find(name);
    if (it == weights_.end()) return 0;
    return it->second.numel;
}

bool WeightLoader::has(const std::string& name) const {
    return weights_.find(name) != weights_.end();
}

}  // namespace plugins
}  // namespace fluxrt
