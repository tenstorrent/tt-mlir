// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_DEBUG_H
#define TT_RUNTIME_DETAIL_DEBUG_H

#include <functional>
#include <optional>
#include <ostream>

#include "tt/runtime/types.h"

namespace tt::runtime::debug {

struct Env {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  static const Env &
#else
  constexpr static Env
#endif
  get(bool dumpKernelsToDisk = false, bool loadKernelsFromDisk = false,
      bool deviceAddressValidation = false, bool blockingCQ = false)
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
      ;
#else
  {
    return Env(false, false, false, false);
  }
#endif

  bool dumpKernelsToDisk;
  bool loadKernelsFromDisk;
  bool deviceAddressValidation;
  bool blockingCQ;

private:
  constexpr Env(bool dumpKernelsToDisk, bool loadKernelsFromDisk,
                bool deviceAddressValidation, bool blockingCQ)
      : dumpKernelsToDisk(dumpKernelsToDisk),
        loadKernelsFromDisk(loadKernelsFromDisk),
        deviceAddressValidation(deviceAddressValidation),
        blockingCQ(blockingCQ) {}
};

inline std::ostream &operator<<(std::ostream &os, const Env &env) {
  os << "debug::Env{\n"
     << "\t"
     << "dumpKernelsToDisk: " << env.dumpKernelsToDisk << "\n"
     << "\t"
     << "loadKernelsFromDisk: " << env.loadKernelsFromDisk << "\n"
     << "\t"
     << "deviceAddressValidation: " << env.deviceAddressValidation << "\n"
     << "\t"
     << "blockingCQ: " << env.blockingCQ << "\n"
     << "}";
  return os;
}

struct PerfEnv {
#if defined(TT_RUNTIME_ENABLE_PERF_TRACE) && TT_RUNTIME_ENABLE_PERF_TRACE == 1
  static const PerfEnv &
#else
  constexpr static PerfEnv
#endif
  get(std::uint32_t dumpDeviceRate = 1000)
#if defined(TT_RUNTIME_ENABLE_PERF_TRACE) && TT_RUNTIME_ENABLE_PERF_TRACE == 1
      ;
#else
  {
    return PerfEnv(1000);
  }
#endif

  std::uint32_t dumpDeviceRate;

private:
  constexpr PerfEnv(std::uint32_t dumpDeviceRate)
      : dumpDeviceRate(dumpDeviceRate) {}
};

inline std::ostream &operator<<(std::ostream &os, const PerfEnv &perfEnv) {
  os << "debug::PerfEnv{\n"
     << "\t" << "dumpDeviceRate: " << perfEnv.dumpDeviceRate << "\n"
     << "}";
  return os;
}

struct Hooks {
  using CallbackFn = std::function<void(Binary, CallbackContext, OpContext)>;
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  static const Hooks &
  get(std::optional<CallbackFn> preOperatorCallback = std::nullopt,
      std::optional<CallbackFn> postOperatorCallback = std::nullopt);
#else
  constexpr static Hooks get() { return Hooks(); }
#endif

  std::optional<CallbackFn> getPreOperatorCallback() const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    return preOperatorCallback;
#else
    return std::nullopt;
#endif
  }

  std::optional<CallbackFn> getPostOperatorCallback() const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    return postOperatorCallback;
#else
    return std::nullopt;
#endif
  }

  void unregisterHooks() const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    preOperatorCallback = std::nullopt;
    postOperatorCallback = std::nullopt;
#endif
  }

private:
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  Hooks(std::optional<CallbackFn> preOperatorCallback,
        std::optional<CallbackFn> postOperatorCallback)
      : preOperatorCallback(preOperatorCallback),
        postOperatorCallback(postOperatorCallback) {}

  mutable std::optional<CallbackFn> preOperatorCallback;
  mutable std::optional<CallbackFn> postOperatorCallback;

#else
  constexpr Hooks() = default;
#endif
};

inline std::ostream &operator<<(std::ostream &os, const Hooks &hooks) {
  os << "debug::Hooks{\n"
     << "\t"
     << "preOperatorCallback: "
     << static_cast<bool>(hooks.getPreOperatorCallback())
     << "postOperatorCallback: "
     << static_cast<bool>(hooks.getPostOperatorCallback()) << ",\n"
     << "}";
  return os;
}
} // namespace tt::runtime::debug

#endif // TT_RUNTIME_DETAIL_DEBUG_H
