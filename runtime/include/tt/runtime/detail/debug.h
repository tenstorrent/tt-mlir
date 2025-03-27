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
  static Env const &
#else
  constexpr static Env
#endif
  get(bool loadKernelsFromDisk = false)
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
      ;
#else
  {
    return Env(false);
  }
#endif

  bool loadKernelsFromDisk;

private:
  constexpr Env(bool loadKernelsFromDisk)
      : loadKernelsFromDisk(loadKernelsFromDisk) {}
};

inline std::ostream &operator<<(std::ostream &os, Env const &env) {
  os << "debug::Env{\n"
     << "\t" << "loadKernelsFromDisk: " << env.loadKernelsFromDisk << "\n"
     << "}";
  return os;
}

struct PreHooks {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  static PreHooks const &
  get(std::optional<std::function<void(Binary, CallbackContext, OpContext)>>
          operatorCallback = std::nullopt);
#else
  constexpr static PreHooks get() { return PreHooks(); }
#endif

  std::optional<std::function<void(Binary, CallbackContext, OpContext)>>
  getOperatorCallback() const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    return operatorCallback;
#else
    return std::nullopt;
#endif
  }

  void unregisterHooks() const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    operatorCallback = std::nullopt;
#endif
  }

private:
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  PreHooks(
      std::optional<std::function<void(Binary, CallbackContext, OpContext)>>
          operatorCallback)
      : operatorCallback(operatorCallback) {}

  mutable std::optional<std::function<void(Binary, CallbackContext, OpContext)>>
      operatorCallback;

#else
  constexpr PreHooks() = default;
#endif
};

struct PostHooks {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  static PostHooks const &
  get(std::optional<std::function<void(Binary, CallbackContext, OpContext)>>
          operatorCallback = std::nullopt);
#else
  constexpr static PostHooks get() { return PostHooks(); }
#endif

  std::optional<std::function<void(Binary, CallbackContext, OpContext)>>
  getOperatorCallback() const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    return operatorCallback;
#else
    return std::nullopt;
#endif
  }

  void unregisterHooks() const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    operatorCallback = std::nullopt;
#endif
  }

private:
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  PostHooks(
      std::optional<std::function<void(Binary, CallbackContext, OpContext)>>
          operatorCallback)
      : operatorCallback(operatorCallback) {}

  mutable std::optional<std::function<void(Binary, CallbackContext, OpContext)>>
      operatorCallback;

#else
  constexpr PostHooks() = default;
#endif
};

inline std::ostream &operator<<(std::ostream &os, PreHooks const &pre_hooks) {
  os << "debug::PreHooks{\n"
     << "\t"
     << "operatorCallback: "
     << static_cast<bool>(pre_hooks.getOperatorCallback()) << ",\n"
     << "}";
  return os;
}

inline std::ostream &operator<<(std::ostream &os, PostHooks const &post_hooks) {
  os << "debug::PostHooks{\n"
     << "\t"
     << "operatorCallback: "
     << static_cast<bool>(post_hooks.getOperatorCallback()) << ",\n"
     << "}";
  return os;
}

} // namespace tt::runtime::debug

#endif // TT_RUNTIME_DETAIL_DEBUG_H
