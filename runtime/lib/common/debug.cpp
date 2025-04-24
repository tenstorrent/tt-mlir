// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/debug.h"
#include <set>

namespace tt::runtime::debug {

#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1

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

#endif // TT_RUNTIME_DEBUG

#if defined(TT_RUNTIME_ENABLE_PERF_TRACE) && TT_RUNTIME_ENABLE_PERF_TRACE == 1

PerfEnv const &PerfEnv::get(std::uint32_t dumpDeviceRate) {
  static PerfEnv config(dumpDeviceRate);
  return config;
}

#endif // TT_RUNTIME_ENABLE_PERF_TRACE

} // namespace tt::runtime::debug
