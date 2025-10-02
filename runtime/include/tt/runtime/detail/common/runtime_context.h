// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_COMMON_RUNTIME_CONTEXT_H
#define TT_RUNTIME_DETAIL_COMMON_RUNTIME_CONTEXT_H

#include "tt/runtime/types.h"
#include <atomic>
#include <filesystem>

namespace tt::runtime {

class RuntimeContext {
public:
  RuntimeContext &operator=(const RuntimeContext &) = delete;
  RuntimeContext &operator=(RuntimeContext &&) = delete;
  RuntimeContext(const RuntimeContext &) = delete;
  RuntimeContext(RuntimeContext &&) = delete;

  static RuntimeContext &instance();

  std::string getMlirHome() const;
  void setMlirHome(std::string_view mlirHome);

  std::string getMetalHome() const;

  DeviceRuntime getCurrentDeviceRuntime() const;
  void setCurrentDeviceRuntime(const DeviceRuntime &runtime);

  HostRuntime getCurrentHostRuntime() const;
  void setCurrentHostRuntime(const HostRuntime &runtime);

  FabricConfig getCurrentFabricConfig() const;
  void setCurrentFabricConfig(const FabricConfig &config);

private:
  RuntimeContext();
  ~RuntimeContext() = default;

  std::filesystem::path mlirHome_;
  std::filesystem::path metalHome_;

  std::atomic<DeviceRuntime> currentDeviceRuntime_ = DeviceRuntime::Disabled;
  std::atomic<HostRuntime> currentHostRuntime_ = HostRuntime::Local;
  std::atomic<FabricConfig> currentFabricConfig_ = FabricConfig::DISABLED;
};

} // namespace tt::runtime

#endif // TT_RUNTIME_DETAIL_COMMON_RUNTIME_CONTEXT_H
