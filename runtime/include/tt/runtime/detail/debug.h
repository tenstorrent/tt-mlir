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
  static Env const &get(bool loadKernelsFromDisk = false,
                        std::vector<std::uint32_t> preTaggedOpIds = {},
                        std::vector<std::uint32_t> postTaggedOpIds = {});
#else
  constexpr static Env get() { return Env(); }
#endif

  std::vector<std::uint32_t> getPreTaggedOpIds() const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    return preTaggedOpIds;
#else
    return {};
#endif
  }

  std::vector<std::uint32_t> getPostTaggedOpIds() const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    return postTaggedOpIds;
#else
    return {};
#endif
  }

  void tagPreOp(uint32_t id) const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    preTaggedOpIds.push_back(id);
#endif
  }

  void tagPostOp(uint32_t id) const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    postTaggedOpIds.push_back(id);
#endif
  }

  void tagPreOps(std::vector<std::uint32_t> ids) const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    preTaggedOpIds.insert(preTaggedOpIds.end(), ids.begin(), ids.end());
    ;
#endif
  }

  void tagPostOps(std::vector<std::uint32_t> ids) const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    postTaggedOpIds.insert(postTaggedOpIds.end(), ids.begin(), ids.end());
#endif
  }

  void untagPreOp(uint32_t id) const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    preTaggedOpIds.erase(
        std::remove(preTaggedOpIds.begin(), preTaggedOpIds.end(), id));
#endif
  }

  void untagPostOp(uint32_t id) const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    postTaggedOpIds.erase(
        std::remove(postTaggedOpIds.begin(), postTaggedOpIds.end(), id));
#endif
  }

  bool isOpPreTagged(uint32_t id) const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    // return preTaggedOpIds.find(id) != preTaggedOpIds.end();
    return std::find(preTaggedOpIds.begin(), preTaggedOpIds.end(), id) !=
           preTaggedOpIds.end();
#else
    return false;
#endif
  }

  bool isOpPostTagged(uint32_t id) const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    // return postTaggedOpIds.find(id) != postTaggedOpIds.end();
    return std::find(postTaggedOpIds.begin(), postTaggedOpIds.end(), id) !=
           postTaggedOpIds.end();
#else
    return false;
#endif
  }

  void unregisterEnv() const {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    preTaggedOpIds = {};
    postTaggedOpIds = {};
#endif
  }

  bool loadKernelsFromDisk;

private:
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  Env(bool loadKernelsFromDisk, std::vector<std::uint32_t> preTaggedOpIds,
      std::vector<std::uint32_t> postTaggedOpIds)
      : loadKernelsFromDisk(loadKernelsFromDisk),
        preTaggedOpIds(preTaggedOpIds), postTaggedOpIds(postTaggedOpIds) {}

  mutable std::vector<std::uint32_t> preTaggedOpIds;
  mutable std::vector<std::uint32_t> postTaggedOpIds;
#else
  constexpr Env() = default;
#endif
};

inline std::ostream &operator<<(std::ostream &os, Env const &env) {
  os << "debug::Env{\n"
     << "\t" << "loadKernelsFromDisk: " << env.loadKernelsFromDisk << "\n"
     << "preTaggedOpIds: " << !env.getPreTaggedOpIds().empty()
     << "postTaggedOpIds: " << !env.getPostTaggedOpIds().empty() << ",\n"
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
