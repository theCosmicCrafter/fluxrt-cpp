// Copyright 2026 FluxRT-CPP Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Plugin registration.  The REGISTER_TENSORRT_PLUGIN macro instantiates
// a static registrar object that registers the creator with the global
// plugin registry before main() starts.

#include "identity_plugin.h"

namespace fluxrt {
namespace plugins {

REGISTER_TENSORRT_PLUGIN(IdentityPluginCreator);

}  // namespace plugins
}  // namespace fluxrt
