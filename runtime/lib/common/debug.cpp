// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/debug.h"

#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1

namespace tt::runtime::debug {

Env const &Env::get(bool loadKernelsFromDisk) {
  static Env config(loadKernelsFromDisk);
  return config;
}

#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
PreHooks const &PreHooks::get(
    std::optional<std::function<void(Binary, CallbackContext, OpContext)>>
        operatorCallback) {
  static PreHooks config(operatorCallback);
  return config;
}
PostHooks const &PostHooks::get(
    std::optional<std::function<void(Binary, CallbackContext, OpContext)>>
        operatorCallback) {
  static PostHooks config(operatorCallback);
  return config;
}
#else
PreHooks get() { return PreHooks(); }
PostHooks get() { return PostHooks(); }
#endif

} // namespace tt::runtime::debug

#endif
