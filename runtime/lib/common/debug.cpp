// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1

#include <mutex>
#include <shared_mutex>

#include "tt/runtime/detail/debug.h"

namespace tt::runtime::debug {

const Env &Env::get(bool dumpKernelsToDisk, bool loadKernelsFromDisk,
                    bool deviceAddressValidation, bool blockingCQ) {
  static Env config(dumpKernelsToDisk, loadKernelsFromDisk,
                    deviceAddressValidation, blockingCQ);
  return config;
}

const Hooks &
Hooks::get(std::optional<debug::Hooks::CallbackFn> preOperatorCallback,
           std::optional<debug::Hooks::CallbackFn> postOperatorCallback) {
  static Hooks config(preOperatorCallback, postOperatorCallback);
  return config;
}

Stats &Stats::instance() {
  static Stats stats;
  return stats;
}

void Stats::incrementStat(const std::string &stat, std::int64_t value) {
  Stats &stats = Stats::instance();
  std::unique_lock<std::shared_mutex> lock(stats.countersMutex);
  stats.counters[stat] += value;
}

std::int64_t Stats::getStat(const std::string &stat) {
  Stats &stats = Stats::instance();
  std::shared_lock<std::shared_mutex> lock(stats.countersMutex);
  auto it = stats.counters.find(stat);
  if (it == stats.counters.end()) {
    return 0;
  }
  return it->second;
}

void Stats::removeStat(const std::string &stat) {
  Stats &stats = Stats::instance();
  std::unique_lock<std::shared_mutex> lock(stats.countersMutex);
  stats.counters.erase(stat);
}

void Stats::clearStats() {
  Stats &stats = Stats::instance();
  std::unique_lock<std::shared_mutex> lock(stats.countersMutex);
  stats.counters.clear();
}

} // namespace tt::runtime::debug

#endif
