// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/debug.h"
#include <set>

#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1

namespace tt::runtime::debug {

Env const &Env::get(bool dumpKernelsToDisk, bool loadKernelsFromDisk,
                    bool deviceAddressValidation, bool blockingCQ) {
  static Env config(dumpKernelsToDisk, loadKernelsFromDisk,
                    deviceAddressValidation, blockingCQ);
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
