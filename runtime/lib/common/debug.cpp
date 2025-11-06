// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1

#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <sstream>

#include <Python.h>

#include "tt/runtime/debug.h"
#include "tt/runtime/detail/python/nanobind_headers.h"

namespace nb = nanobind;

namespace tt::runtime::debug {

const Env &Env::get(bool dumpKernelsToDisk, bool loadKernelsFromDisk,
                    bool useLocForKernelName, std::string kernelSourceDir,
                    bool deviceAddressValidation, bool blockingCQ) {
  static Env config(dumpKernelsToDisk, loadKernelsFromDisk, useLocForKernelName,
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

void RuntimeChiselBridge::initialize(const char *ttir_mlir,
                                     const char *ttnn_mlir) {
  assert(!initialized_);

  ownsInterpreter = (Py_IsInitialized() == 0);
  std::cout << "ownsInterpreter: " << ownsInterpreter << std::endl;
  if (ownsInterpreter) {
    // Py_Initialize();
  }
  nb::object chisel_context;
  nb::callable pre_callable, post_callable;

  auto runtime_module = nb::module_::import_("ttrt.runtime");
  auto util_module = nb::module_::import_("ttrt.common.util");

  auto debug_hooks = runtime_module.attr("DebugHooks");
  auto get_debug_hooks = debug_hooks.attr("get");

  nb::module_ chisel_module = nb::module_::import_("chisel");

  // 3) Construct context (try with args; fallback to no-arg constructor)
  chisel_context = chisel_module.attr("ChiselContext")(std::string(ttir_mlir),
                                                       std::string(ttnn_mlir));

  // 4) Cache bound methods
  pre_callable = nb::cast<nb::callable>(chisel_context.attr("preop"));
  post_callable = nb::cast<nb::callable>(chisel_context.attr("postop"));
  get_debug_hooks(pre_callable, post_callable);

  // 5) Build C++ → Python shims once; only acquire GIL around the calls

  initialized_ = true;
}

void RuntimeChiselBridge::finalize() {
  if (!initialized_) {
    return;
  }
  Hooks::get().unregisterHooks();
  if (ownsInterpreter) {
    Py_Finalize();
    ownsInterpreter = false;
  }
  initialized_ = false;
}

RuntimeChiselBridge::~RuntimeChiselBridge() { finalize(); }

} // namespace tt::runtime::debug

#endif
