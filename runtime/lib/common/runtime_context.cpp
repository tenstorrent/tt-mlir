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
#if (!defined(TT_RUNTIME_ENABLE_TTNN) || (TT_RUNTIME_ENABLE_TTNN == 0)) &&     \
    (!defined(TT_RUNTIME_ENABLE_TTMETAL) ||                                    \
     (TT_RUNTIME_ENABLE_TTMETAL == 0)) &&                                      \
    (!defined(TTMLIR_ENABLE_CUDA) || (TTMLIR_ENABLE_CUDA == 0))
  LOG_FATAL(
      "Runtime context cannot be initialized when no runtimes are enabled");
#endif

#if defined(TT_RUNTIME_ENABLE_TTNN) && (TT_RUNTIME_ENABLE_TTNN == 1)
  currentRuntime = DeviceRuntime::TTNN;
#elif defined(TT_RUNTIME_ENABLE_TTMETAL) && (TT_RUNTIME_ENABLE_TTMETAL == 1)
  currentRuntime = DeviceRuntime::TTMetal;
#elif defined(TTMLIR_ENABLE_CUDA) && (TTMLIR_ENABLE_CUDA == 1)
  currentRuntime = DeviceRuntime::CUDA;
#endif
}

DeviceRuntime RuntimeContext::getCurrentRuntime() const {
  DeviceRuntime runtime = currentRuntime.load(std::memory_order_relaxed);

#if !defined(TT_RUNTIME_ENABLE_TTNN) || (TT_RUNTIME_ENABLE_TTNN == 0)
  LOG_ASSERT(runtime != DeviceRuntime::TTNN);
#endif

#if !defined(TT_RUNTIME_ENABLE_TTMETAL) || (TT_RUNTIME_ENABLE_TTMETAL == 0)
  LOG_ASSERT(runtime != DeviceRuntime::TTMetal);
#endif

#if !defined(TTMLIR_ENABLE_CUDA) || (TTMLIR_ENABLE_CUDA == 0)
  LOG_ASSERT(runtime != DeviceRuntime::CUDA);
#endif

  return runtime;
}

void RuntimeContext::setCurrentRuntime(const DeviceRuntime &runtime) {
#if !defined(TT_RUNTIME_ENABLE_TTNN) || (TT_RUNTIME_ENABLE_TTNN == 0)
  LOG_ASSERT(runtime != DeviceRuntime::TTNN);
#endif
#if !defined(TT_RUNTIME_ENABLE_TTMETAL) || (TT_RUNTIME_ENABLE_TTMETAL == 0)
  LOG_ASSERT(runtime != DeviceRuntime::TTMetal);
#endif
#if !defined(TTMLIR_ENABLE_CUDA) || (TTMLIR_ENABLE_CUDA == 0)
  LOG_ASSERT(runtime != DeviceRuntime::CUDA);
#endif

  currentRuntime.store(runtime, std::memory_order_relaxed);
}

FabricConfig RuntimeContext::getCurrentFabricConfig() const {
  FabricConfig config = currentFabricConfig.load(std::memory_order_relaxed);
  return config;
}
void RuntimeContext::setCurrentFabricConfig(const FabricConfig &config) {
  currentFabricConfig.store(config, std::memory_order_relaxed);
}

} // namespace tt::runtime
