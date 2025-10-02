// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/common/runtime_context.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/types.h"

namespace tt::runtime {

RuntimeContext &RuntimeContext::instance() {
  static RuntimeContext instance;
  return instance;
}

RuntimeContext::RuntimeContext() {
#if !defined(DEVICE_RUNTIME_ENABLED)
  LOG_FATAL(
      "Runtime context cannot be initialized when no runtimes are enabled");
#endif

#if defined(TT_RUNTIME_ENABLE_TTNN) && (TT_RUNTIME_ENABLE_TTNN == 1)
  currentDeviceRuntime_ = DeviceRuntime::TTNN;
#elif defined(TT_RUNTIME_ENABLE_TTMETAL) && (TT_RUNTIME_ENABLE_TTMETAL == 1)
  currentDeviceRuntime_ = DeviceRuntime::TTMetal;
#endif
}

std::string RuntimeContext::getMlirHome() const { return mlirHome_.string(); }

void RuntimeContext::setMlirHome(std::string_view mlirHome) {
  constexpr std::string_view metalPath = "third_party/tt-metal/src/tt-metal";
  mlirHome_ = mlirHome;
  metalHome_ = mlirHome_ / metalPath;
}

std::string RuntimeContext::getMetalHome() const { return metalHome_.string(); }

DeviceRuntime RuntimeContext::getCurrentDeviceRuntime() const {
  DeviceRuntime runtime = currentDeviceRuntime_.load(std::memory_order_relaxed);

#if !defined(TT_RUNTIME_ENABLE_TTNN) || (TT_RUNTIME_ENABLE_TTNN == 0)
  LOG_ASSERT(runtime != DeviceRuntime::TTNN);
#endif

#if !defined(TT_RUNTIME_ENABLE_TTMETAL) || (TT_RUNTIME_ENABLE_TTMETAL == 0)
  LOG_ASSERT(runtime != DeviceRuntime::TTMetal);
#endif

  return runtime;
}

void RuntimeContext::setCurrentDeviceRuntime(const DeviceRuntime &runtime) {
#if !defined(TT_RUNTIME_ENABLE_TTNN) || (TT_RUNTIME_ENABLE_TTNN == 0)
  LOG_ASSERT(runtime != DeviceRuntime::TTNN);
#endif
#if !defined(TT_RUNTIME_ENABLE_TTMETAL) || (TT_RUNTIME_ENABLE_TTMETAL == 0)
  LOG_ASSERT(runtime != DeviceRuntime::TTMetal);
#endif
  currentDeviceRuntime_.store(runtime, std::memory_order_relaxed);
}

HostRuntime RuntimeContext::getCurrentHostRuntime() const {
  return currentHostRuntime_.load(std::memory_order_relaxed);
}

void RuntimeContext::setCurrentHostRuntime(const HostRuntime &runtime) {
  currentHostRuntime_.store(runtime, std::memory_order_relaxed);
}

FabricConfig RuntimeContext::getCurrentFabricConfig() const {
  FabricConfig config = currentFabricConfig_.load(std::memory_order_relaxed);
  return config;
}
void RuntimeContext::setCurrentFabricConfig(const FabricConfig &config) {
  currentFabricConfig_.store(config, std::memory_order_relaxed);
}

} // namespace tt::runtime
