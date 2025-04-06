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

struct PreOperationHooks {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  static PreOperationHooks const &
  get(std::optional<std::function<void(Binary, CallbackContext, OpContext)>>
          operatorCallback = std::nullopt);
#else
  constexpr static PreOperationHooks get() { return PreOperationHooks(); }
#endif

  std::optional<std::string> getCallbackKey() const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    return callbackKey;
#else
    return std::nullopt;
#endif
  }

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
    callbackKey = std::nullopt;
    operatorCallback = std::nullopt;
#endif
  }

private:
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  PreOperationHooks(
      std::optional<std::function<void(Binary, CallbackContext, OpContext)>>
          operatorCallback)
      : operatorCallback(operatorCallback) {}

  mutable std::optional<std::string> callbackKey;
  mutable std::optional<std::function<void(Binary, CallbackContext, OpContext)>>
      operatorCallback;

#else
  constexpr PreOperationHooks() = default;
#endif
};

struct PostOperationHooks {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  static PostOperationHooks const &
  get(std::optional<std::function<void(Binary, CallbackContext, OpContext)>>
          operatorCallback = std::nullopt);
#else
  constexpr static PostOperationHooks get() { return PostOperationHooks(); }
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
  PostOperationHooks(
      std::optional<std::function<void(Binary, CallbackContext, OpContext)>>
          operatorCallback)
      : operatorCallback(operatorCallback) {}

  mutable std::optional<std::function<void(Binary, CallbackContext, OpContext)>>
      operatorCallback;

#else
  constexpr PostOperationHooks() = default;
#endif
};

inline std::ostream &operator<<(std::ostream &os,
                                PreOperationHooks const &pre_operation_hooks) {
  os << "debug::PreOperationHooks{\n"
     << "\t"
     << "operatorCallback: "
     << static_cast<bool>(pre_operation_hooks.getOperatorCallback()) << ",\n"
     << "}";
  return os;
}

inline std::ostream &
operator<<(std::ostream &os, PostOperationHooks const &post_operation_hooks) {
  os << "debug::PostOperationHooks{\n"
     << "\t"
     << "operatorCallback: "
     << static_cast<bool>(post_operation_hooks.getOperatorCallback()) << ",\n"
     << "}";
  return os;
}

} // namespace tt::runtime::debug

#endif // TT_RUNTIME_DETAIL_DEBUG_H
