// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/debug.h"
#include <set>

namespace tt::runtime::debug {

#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1

const Env &Env::get(bool dumpKernelsToDisk, bool loadKernelsFromDisk,
                    bool deviceAddressValidation, bool blockingCQ) {
  static Env config(dumpKernelsToDisk, loadKernelsFromDisk,
                    deviceAddressValidation, blockingCQ);
  return config;
}

const Hooks &
Hooks::get(std::optional<debug::Hooks::CallbackFn> preOperatorCallback,
           std::optional<debug::Hooks::CallbackFn> postOperatorCallback) {
  static Hooks config(preOperatorCallback, postOperatorCallback);
  return config;
}

#endif // TT_RUNTIME_DEBUG

#if defined(TT_RUNTIME_ENABLE_PERF_TRACE) && TT_RUNTIME_ENABLE_PERF_TRACE == 1

const PerfEnv &PerfEnv::get(std::uint32_t dumpDeviceRate,
                            bool enablePerfTrace) {
  static PerfEnv config(dumpDeviceRate, enablePerfTrace);
  return config;
}

#endif // TT_RUNTIME_ENABLE_PERF_TRACE

} // namespace tt::runtime::debug
