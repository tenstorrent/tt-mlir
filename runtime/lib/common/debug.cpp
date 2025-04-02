// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/debug.h"
#include <set>

#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1

namespace tt::runtime::debug {

Env const &Env::get(bool loadKernelsFromDisk) {
  static Env config(loadKernelsFromDisk);
  return config;
}

Hooks const &Hooks::get(
    std::optional<std::string> callbackKey,
    std::optional<std::function<void(Binary, CallbackContext, OpContext)>>
        operatorCallback) {
  if (std::set<std::string> validKeys{"post-op", "pre-op"};
      !validKeys.contains(callbackKey.value())) {
    throw std::runtime_error("callbackKey must be 'post-op' or 'pre-op', got " +
                             callbackKey.value());
  }
  static Hooks config(callbackKey, operatorCallback);
  return config;
}

} // namespace tt::runtime::debug

#endif
