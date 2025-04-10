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

// Will have to move to another file
struct RuntimeModifications {
  using op = ::tt::target::ttnn::Operation;
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  static RuntimeModifications const &
  get(std::optional<std::uint32_t> dumpDeviceRate = 1000,
      std::unordered_set<op> taggedOps = {});
#else
  constexpr static RuntimeModifications get() { return RuntimeModifications(); }
#endif

  void setDumpDeviceRate(std::uint32_t rate) const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    dumpDeviceRate = rate;
#endif
  }

  std::optional<std::uint32_t> getDumpDeviceRate() const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    return dumpDeviceRate;
#else
    return 1000;
#endif
  }

  void addTaggedOps(std::unordered_set<op> newTaggedOps) const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    taggedOps.merge(newTaggedOps);
#endif
  }

  std::unordered_set<op> getTaggedOps() const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    return taggedOps;
#else
    return {};
#endif
  }

  void unregisterRuntimeModifications() const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    dumpDeviceRate = 1000;
    taggedOps = {};

#endif
  }

private:
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  RuntimeModifications(std::optional<std::uint32_t> dumpDeviceRate,
                       std::unordered_set<op> taggedOps)
      : dumpDeviceRate(dumpDeviceRate), taggedOpIDs(taggedOps) {}

  mutable std::optional<std::uint32_t> dumpDeviceRate;
  mutable std::unordered_set<op> taggedOps;
#else
  constexpr RuntimeModifications() = default;
#endif
};

// Skipping ostream

} // namespace tt::runtime::debug

#endif // TT_RUNTIME_DETAIL_DEBUG_H
