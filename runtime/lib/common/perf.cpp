// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#if defined(TT_RUNTIME_ENABLE_PERF_TRACE) && TT_RUNTIME_ENABLE_PERF_TRACE == 1

#include "tt/runtime/perf.h"

namespace tt::runtime::perf {

const Env &Env::get(std::uint32_t dumpDeviceRate, bool enablePerfTrace) {
  static Env config(dumpDeviceRate, enablePerfTrace);
  return config;
}

} // namespace tt::runtime::perf

#endif
