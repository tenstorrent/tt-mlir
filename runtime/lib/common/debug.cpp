// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1

#include <mutex>
#include <shared_mutex>
#include <sstream>

#include "tt/runtime/debug.h"

namespace tt::runtime::debug {

const Env &Env::get(bool dumpKernels, bool loadKernels,
                    bool useLocForKernelName, std::string kernelSourceDir,
                    bool deviceAddressValidation, bool blockingCQ) {
  static Env config(dumpKernels, loadKernels, useLocForKernelName,
                    kernelSourceDir, deviceAddressValidation, blockingCQ);
  return config;
}

const Hooks &
Hooks::get(std::optional<debug::Hooks::CallbackFn> preOperatorCallback,
           std::optional<debug::Hooks::CallbackFn> postOperatorCallback) {
  static Hooks config(preOperatorCallback, postOperatorCallback);
  if (preOperatorCallback.has_value()) {
    config.preOperatorCallback = preOperatorCallback;
  }
  if (postOperatorCallback.has_value()) {
    config.postOperatorCallback = postOperatorCallback;
  }
  return config;
}

Stats &Stats::get() {
  static Stats stats;
  return stats;
}

void Stats::incrementStat(const std::string &stat, std::int64_t value) {
  std::unique_lock<std::shared_mutex> lock(countersMutex);
  counters[stat] += value;
}

std::int64_t Stats::getStat(const std::string &stat) const {
  std::shared_lock<std::shared_mutex> lock(countersMutex);
  auto it = counters.find(stat);
  return it == counters.end() ? 0 : it->second;
}

void Stats::removeStat(const std::string &stat) {
  std::unique_lock<std::shared_mutex> lock(countersMutex);
  counters.erase(stat);
}

void Stats::clear() {
  std::unique_lock<std::shared_mutex> lock(countersMutex);
  counters.clear();
}

std::string Stats::toString() const {
  std::shared_lock<std::shared_mutex> lock(countersMutex);

  std::ostringstream oss;
  oss << "DebugStats{\n";
  if (counters.empty()) {
    oss << "\t(no stat counters recorded)\n";
  } else {
    for (const auto &[key, value] : counters) {
      oss << "\t" << key << ": " << value << "\n";
    }
  }
  oss << "\t" << this << "\n";
  oss << "}";
  return oss.str();
}

} // namespace tt::runtime::debug

#endif
