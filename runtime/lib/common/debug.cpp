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

Hooks const &
Hooks::get(std::optional<debug::Hooks::CallbackFn> preOperatorCallback,
           std::optional<debug::Hooks::CallbackFn> postOperatorCallback) {
  static Hooks config(preOperatorCallback, postOperatorCallback);
  return config;
}

} // namespace tt::runtime::debug

#endif
