// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Explicit plugin registration helper.
// When plugins are built as a static library the linker may drop the
// REGISTER_TENSORRT_PLUGIN objects; call initFluxRTPlugins() before
// deserializing an engine that contains custom plugins.

#pragma once

namespace fluxrt {
namespace plugins {

// Register all FluxRT plugin creators with the TensorRT plugin registry.
// Safe to call multiple times.
void initFluxRTPlugins();

}  // namespace plugins
}  // namespace fluxrt
