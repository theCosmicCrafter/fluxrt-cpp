// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0

#include "plugin_init.h"
#include "identity_plugin.h"

#include <NvInferRuntime.h>

namespace fluxrt {
namespace plugins {

void initFluxRTPlugins() {
    static bool initialized = false;
    if (initialized) return;

    nvinfer1::IPluginRegistry* registry = getPluginRegistry();
    if (registry) {
        static IdentityPluginCreator identityCreator;
        registry->registerCreator(identityCreator, "fluxrt");
    }

    initialized = true;
}

}  // namespace plugins
}  // namespace fluxrt
