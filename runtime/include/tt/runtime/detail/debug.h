// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_DEBUG_H
#define TT_RUNTIME_DETAIL_DEBUG_H

#include <functional>
#include <vector>
#include <ostream>

#include "tt/runtime/types.h"

namespace tt::runtime::debug {

struct Env {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  static Env const &
#else
  constexpr static Env
#endif
  get(bool loadKernelsFromDisk = false, bool goldenEval = false)
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
      ;
#else
  {
    return Env(false);
  }
#endif

  bool loadKernelsFromDisk;
  bool goldenEval;

private:
  constexpr Env(bool loadKernelsFromDisk, bool goldenEval)
      : loadKernelsFromDisk(loadKernelsFromDisk),
        goldenEval(goldenEval) {}
};

inline std::ostream &operator<<(std::ostream &os, Env const &env) {
  os << "debug::Env{\n"
     << "\t" << "loadKernelsFromDisk: " << env.loadKernelsFromDisk << "\n"
     << "}";
  return os;
}

struct Hooks {
  using CallbackFn = std::function<void(Binary, CallbackContext, OpContext)>;
  using CallbackHandle = std::size_t;
  using CallbackKV = std::pair<CallbackHandle, CallbackFn>;

  static constexpr std::size_t kInvalidHandle =
      std::numeric_limits<std::size_t>::max();

#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  static Hooks &get();
#else
  constexpr static Hooks get() { return Hooks(); }
#endif

  std::vector<CallbackKV> const& getPreOperatorCallbacks() const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    return preOperatorCallbacks;
#else
    return {};
#endif
  }

  std::vector<CallbackKV> const& getPostOperatorCallbacks() const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    return postOperatorCallbacks;
#else
    return {};
#endif
  }

  CallbackHandle registerPreOperatorCallback(CallbackFn callback) {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    CallbackHandle handle = getNextHandle();
    preOperatorCallbacks.emplace_back(handle, callback);
    return handle;
#else
    LOG_FATAL("TT_RUNTIME_DEBUG not enabled");
    return kInvalidHandle;
#endif
  }

  CallbackHandle registerPostOperatorCallback(CallbackFn callback) {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    CallbackHandle handle = getNextHandle();
    postOperatorCallbacks.emplace_back(handle, callback);
    return handle;
#else
    LOG_FATAL("TT_RUNTIME_DEBUG not enabled");
    return kInvalidHandle;
#endif
  }

  void unregisterPreOperatorCallback(CallbackHandle handle) {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    auto iter =
        std::find_if(preOperatorCallbacks.begin(), preOperatorCallbacks.end(),
                     [handle](CallbackKV kv) { return kv.first == handle; });
    assert(iter != preOperatorCallbacks.end());
    preOperatorCallbacks.erase(iter);
#endif
  }

  void unregisterPostOperatorCallback(CallbackHandle handle) {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    auto iter =
        std::find_if(postOperatorCallbacks.begin(), postOperatorCallbacks.end(),
                     [handle](CallbackKV kv) { return kv.first == handle; });
    assert(iter != postOperatorCallbacks.end());
    postOperatorCallbacks.erase(iter);
#endif
  }

  void unregisterHooks() {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    preOperatorCallbacks.clear();
    postOperatorCallbacks.clear();
#endif
  }

private:
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  Hooks() = default;

  std::size_t getNextHandle() {
    static std::size_t nextHandle = 0;
    return nextHandle++;
  }

  std::vector<CallbackKV> preOperatorCallbacks;
  std::vector<CallbackKV> postOperatorCallbacks;

#else
  constexpr Hooks() = default;
#endif
};

inline std::ostream &operator<<(std::ostream &os, Hooks const &hooks) {
  os << "debug::Hooks{\n"
     << "\t"
     << "preOperatorCallbacks: " << hooks.getPreOperatorCallbacks().size()
     << ",\n"
     << "postOperatorCallbacks: " << hooks.getPostOperatorCallbacks().size()
     << "}";
  return os;
}

struct GoldenEval {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  GoldenEval() = default;
  ~GoldenEval();

  void initialize(const char *mlir,
                  std::vector<::tt::runtime::Tensor> const &inputs);
  void finalize();

  Hooks::CallbackHandle callbackHandle = Hooks::kInvalidHandle;
  bool initialized = false;
  bool ownsInterpreter = false;
#else
  constexpr GoldenEval() {}
#endif
};

} // namespace tt::runtime::debug

#endif // TT_RUNTIME_DETAIL_DEBUG_H
