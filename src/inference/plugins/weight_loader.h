// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Simple weight loader: reads raw float32 .bin files into device buffers.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace fluxrt {
namespace plugins {

// ---------------------------------------------------------------------------
// RAII device buffer for a loaded weight tensor.
// ---------------------------------------------------------------------------
struct DeviceBuffer {
    float* data = nullptr;
    size_t numel = 0;

    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n);
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
    ~DeviceBuffer();

    bool load_from_file(const std::string& path);
};

// ---------------------------------------------------------------------------
// Load all .bin files from a directory into a name→buffer map.
// ---------------------------------------------------------------------------
class WeightLoader {
public:
    explicit WeightLoader(const std::string& dir);

    // Access a loaded weight. Returns nullptr if not found.
    const float* get(const std::string& name) const;
    size_t       size(const std::string& name) const;
    bool         has(const std::string& name) const;

private:
    std::unordered_map<std::string, DeviceBuffer> weights_;
};

}  // namespace plugins
}  // namespace fluxrt
