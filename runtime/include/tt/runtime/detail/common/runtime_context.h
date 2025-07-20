// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_COMMON_RUNTIME_CONTEXT_H
#define TT_RUNTIME_DETAIL_COMMON_RUNTIME_CONTEXT_H

#include "tt/runtime/types.h"
#include <atomic>

namespace tt::runtime {

class RuntimeContext {
public:
  RuntimeContext &operator=(const RuntimeContext &) = delete;
  RuntimeContext &operator=(RuntimeContext &&) = delete;
  RuntimeContext(const RuntimeContext &) = delete;
  RuntimeContext(RuntimeContext &&) = delete;

  static RuntimeContext &instance();

  DeviceRuntime getCurrentRuntime() const;
  void setCurrentRuntime(const DeviceRuntime &runtime);

private:
  RuntimeContext();
  ~RuntimeContext() = default;

  std::atomic<DeviceRuntime> currentRuntime = DeviceRuntime::Disabled;
};

} // namespace tt::runtime

#endif // TT_RUNTIME_DETAIL_COMMON_RUNTIME_CONTEXT_H
