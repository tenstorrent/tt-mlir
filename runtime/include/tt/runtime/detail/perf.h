// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_PERF_H
#define TT_RUNTIME_DETAIL_PERF_H

#include <ostream>

namespace tt::runtime::perf {
struct Env {
#if defined(TT_RUNTIME_ENABLE_PERF_TRACE) && TT_RUNTIME_ENABLE_PERF_TRACE == 1
  static const Env &
#else
  constexpr static Env
#endif
  get(std::uint32_t dumpDeviceRate = 1000, bool enablePerfTrace = false)
#if defined(TT_RUNTIME_ENABLE_PERF_TRACE) && TT_RUNTIME_ENABLE_PERF_TRACE == 1
      ;
#else
  {
    return Env(1000, false);
  }
#endif

  std::uint32_t dumpDeviceRate;
  bool enablePerfTrace;

private:
  constexpr Env(std::uint32_t dumpDeviceRate, bool enablePerfTrace)
      : dumpDeviceRate(dumpDeviceRate), enablePerfTrace(enablePerfTrace) {}
};

inline std::ostream &operator<<(std::ostream &os, const Env &env) {
  os << "perf::Env{\n"
     << "\t"
     << "dumpDeviceRate: " << env.dumpDeviceRate << "\n"
     << "\t"
     << "enablePerfTrace: " << env.enablePerfTrace << "\n"
     << "}";
  return os;
}
} // namespace tt::runtime::perf

#endif
