// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#if defined(TT_RUNTIME_ENABLE_PERF_TRACE) && TT_RUNTIME_ENABLE_PERF_TRACE == 1

#include "tt/runtime/perf.h"

namespace tt::runtime::perf {

Env &Env::get(std::uint32_t dumpDeviceRate, bool enablePerfTrace,
              const std::string &tracyProgramMetadata) {
  static Env config(dumpDeviceRate, enablePerfTrace, tracyProgramMetadata);
  return config;
}

} // namespace tt::runtime::perf

#endif
