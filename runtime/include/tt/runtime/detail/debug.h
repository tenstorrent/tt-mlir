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
  static Env &
#else
  constexpr static Env
#endif
  get(bool loadKernelsFromDisk = false, std::uint32_t dumpDeviceRate = 1000)
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
      ;
#else
  {
    return Env(false, 1000);
  }
#endif

  std::uint32_t getDumpDeviceRate() {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    return dumpDeviceRate;
#else
    return 1000;
#endif
  }

  void setDumpDeviceRate(std::uint32_t rate) {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    dumpDeviceRate = rate;
#endif
  }

  bool loadKernelsFromDisk;

private:
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  Env(bool loadKernelsFromDisk, std::uint32_t dumpDeviceRate)
      : loadKernelsFromDisk(loadKernelsFromDisk),
        dumpDeviceRate(dumpDeviceRate) {}

  mutable std::uint32_t dumpDeviceRate;
#else
  constexpr Env() = default;
#endif
};

inline std::ostream &operator<<(std::ostream &os, Env &env) {
  os << "debug::Env{\n"
     << "\t" << "loadKernelsFromDisk: " << env.loadKernelsFromDisk << "\n"
     << "dumpDeviceRate: " << std::to_string(env.getDumpDeviceRate()) << "\n"
     << "}";
  return os;
}

struct Hooks {
  using CallbackFn = std::function<void(Binary, CallbackContext, OpContext)>;
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  static Hooks const &
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

inline std::ostream &operator<<(std::ostream &os, Hooks const &hooks) {
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
