// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_DEBUG_H
#define TT_RUNTIME_DETAIL_DEBUG_H

#include <functional>
#include <optional>
#include <ostream>
#include <unordered_set>

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

enum class CallbackKey { PreOp, PostOp, Null };

struct Hooks {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  static Hooks const &
  get(std::optional<CallbackKey> callbackKey = CallbackKey::Null,
      std::optional<std::function<void(Binary, CallbackContext, OpContext)>>
          operatorCallback = std::nullopt);
#else
  constexpr static Hooks get() { return Hooks(); }
#endif

  std::optional<CallbackKey> getCallbackKey() const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    return callbackKey;
#else
    return CallbackKey::Null

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

  const std::unordered_set<CallbackKey> &getValidCallbackKeys() {
    static const std::unordered_set<CallbackKey> validKeys = {
        CallbackKey::PreOp, CallbackKey::PostOp, CallbackKey::Null};
    return validKeys;
  }

  void unregisterHooks() const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    callbackKey = CallbackKey::Null;
    operatorCallback = std::nullopt;
#endif
  }

private:
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  Hooks(std::optional<CallbackKey> callbackKey,
        std::optional<std::function<void(Binary, CallbackContext, OpContext)>>
            operatorCallback)
      : callbackKey(callbackKey), operatorCallback(operatorCallback) {}

  mutable std::optional<CallbackKey> callbackKey;
  mutable std::optional<std::function<void(Binary, CallbackContext, OpContext)>>
      operatorCallback;

#else
  constexpr Hooks() = default;
#endif
};

inline std::ostream &operator<<(std::ostream &os, Hooks const &hooks) {
  os << "debug::Hooks{\n"
     << "\t"
     << "callbackKey: " << static_cast<bool>(hooks.getCallbackKey())
     << "operatorCallback: " << static_cast<bool>(hooks.getOperatorCallback())
     << ",\n"
     << "}";
  return os;
}

} // namespace tt::runtime::debug

#endif // TT_RUNTIME_DETAIL_DEBUG_H
