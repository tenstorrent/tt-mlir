// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/debug.h"

#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1

namespace tt::runtime::debug {

Env const &Env::get(bool loadKernelsFromDisk, bool enableAsyncTTNN) {
  static Env config(loadKernelsFromDisk, enableAsyncTTNN);
  return config;
}

#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
Hooks const &Hooks::get(
    std::optional<
        std::function<void(RuntimeConfig, Binary, CallbackContext, OpContext)>>
        operatorCallback) {
  static Hooks config(operatorCallback);
  return config;
}
#else
Hooks get() { return Hooks(); }
#endif

RuntimeConfig const &RuntimeConfig::get(double atol, double rtol, double pcc,
                                        std::string artifact_dir) {
  static RuntimeConfig config(atol, rtol, pcc, artifact_dir);
  return config;
}

} // namespace tt::runtime::debug

#endif
