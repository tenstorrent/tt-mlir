// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/debug.h"
#include <set>
#include <unordered_map>

#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1

namespace tt::runtime::debug {

std::unordered_map<std::string, Hooks> Hooks::config;

Env const &Env::get(bool loadKernelsFromDisk) {
  static Env config(loadKernelsFromDisk);
  return config;
}

Hooks const &Hooks::get(
    std::optional<std::string> callbackKey,
    std::optional<std::function<void(Binary, CallbackContext, OpContext)>>
        operatorCallback) {
  if (std::set<std::string> validKeys{"post-op", "pre-op"};
      callbackKey and !validKeys.contains(callbackKey.value())) {
    throw std::runtime_error("callbackKey must be 'post-op' or 'pre-op', got " +
                             callbackKey.value());
  }
  
  if (!callbackKey) {
    static Hooks defaultHooks{};
    return defaultHooks;
  }
  
  if (config.contains(callbackKey.value())) {
    return config.at(callbackKey.value());
  }
  
  if (!operatorCallback) {
    throw std::runtime_error("operatorCallback must be provided");
  }
  
  config.insert({callbackKey.value(), Hooks(callbackKey, operatorCallback.value())});
  return config.at(callbackKey.value());
}

} // namespace tt::runtime::debug

#endif
