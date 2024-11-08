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
Hooks const &
Hooks::get(std::optional<std::function<void(std::optional<Binary>,
                                            std::optional<CallbackContext>,
                                            std::optional<OpContext>)>>
               operatorCallback) {
  static Hooks config(operatorCallback);
  return config;
}
#else
Hooks get() { return Hooks(); }
#endif

} // namespace tt::runtime::debug

#endif
