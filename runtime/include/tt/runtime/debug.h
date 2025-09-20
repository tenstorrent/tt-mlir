// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DEBUG_H
#define TT_RUNTIME_DEBUG_H

#include <cassert>
#include <functional>
#include <optional>
#include <ostream>
#include <shared_mutex>

#include "tt/runtime/types.h"

namespace tt::runtime::debug {

struct Env {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  static const Env &
#else
  constexpr static Env
#endif
  get(bool dumpKernelsToDisk = false, bool loadKernelsFromDisk = false,
      bool useLocForKernelName = false, std::string kernelSourceDir = {},
      bool deviceAddressValidation = false, bool blockingCQ = false)
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
      ;
#else
  {
    return Env(false, false, false, {}, false, false);
  }
#endif

  bool dumpKernelsToDisk;
  bool loadKernelsFromDisk;
  bool useLocForKernelName;
  std::string kernelSourceDir;
  bool deviceAddressValidation;
  bool blockingCQ;

private:
  Env(bool dumpKernelsToDisk, bool loadKernelsFromDisk,
      bool useLocForKernelName, std::string kernelSourceDir,
      bool deviceAddressValidation, bool blockingCQ)
      : dumpKernelsToDisk(dumpKernelsToDisk),
        loadKernelsFromDisk(loadKernelsFromDisk),
        useLocForKernelName(useLocForKernelName),
        kernelSourceDir(kernelSourceDir),
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
     << "useLocForKernelName: " << env.useLocForKernelName << "\n"
     << "\t"
     << "kernelSourceDir: " << env.kernelSourceDir << "\n"
     << "\t"
     << "deviceAddressValidation: " << env.deviceAddressValidation << "\n"
     << "\t"
     << "blockingCQ: " << env.blockingCQ << "\n"
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

struct Stats {
public:
  Stats(const Stats &) = delete;
  Stats &operator=(const Stats &) = delete;

  Stats(Stats &&) = delete;
  Stats &operator=(Stats &&) = delete;

#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  static Stats &get();
  void incrementStat(const std::string &stat, std::int64_t value = 1);
  std::int64_t getStat(const std::string &stat) const;
  void removeStat(const std::string &stat);
  void clear();
  std::string toString() const;
#else
  static constexpr Stats get() { return Stats(); }
  inline void incrementStat(const std::string &, std::int64_t = 1) const {}
  inline std::int64_t getStat(const std::string &) const { return 0; }
  inline void removeStat(const std::string &) const {}
  constexpr void clear() const {}
  inline std::string toString() const { return "DebugStats Disabled"; }
#endif

private:
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  Stats() = default;
  mutable std::shared_mutex countersMutex;
  std::unordered_map<std::string, std::int64_t> counters;
#else
  constexpr Stats() = default;
#endif
};

inline std::ostream &operator<<(std::ostream &os, const Stats &stats) {
  os << stats.toString();
  return os;
}

template <typename Func>
void verifyFlatbuffer(const ::flatbuffers::FlatBufferBuilder &fbb,
                      const Func &verifierFn) {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  ::flatbuffers::Verifier verifier(fbb.GetBufferPointer(), fbb.GetSize());
  bool valid = verifierFn(verifier);
  assert(valid && "Failed to verify flatbuffer");
#endif
}

} // namespace tt::runtime::debug

#endif // TT_RUNTIME_DEBUG_H
