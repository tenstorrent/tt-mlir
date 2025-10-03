// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/common/runtime_context.h"
#include "tt/runtime/detail/distributed/controller/controller.h"
#include "tt/runtime/detail/distributed/distributed.h"

namespace tt::runtime::distributed {

using Controller = tt::runtime::distributed::controller::Controller;

class ControllerSingleton {
public:
  static Controller &get() {
    if (!controller_) {
      controller_ = std::make_unique<Controller>();
    }
    return *controller_;
  }

  static void shutdown() { controller_.reset(); }

  static bool launched() { return controller_ != nullptr; }

private:
  ControllerSingleton() = default;
  ~ControllerSingleton() = default;

  static inline std::unique_ptr<Controller> controller_ = nullptr;
};

void launchDistributedRuntime(const DistributedOptions &options) {
  LOG_ASSERT(RuntimeContext::instance().getCurrentHostRuntime() ==
                 HostRuntime::Distributed,
             "Distributed controller can only be launched on distributed host "
             "runtime");
  LOG_ASSERT(
      !ControllerSingleton::launched(),
      "Distributed controller already launched, please shutdown the controller "
      "before launching a new one");
  Controller &controller = ControllerSingleton::get();
  switch (options.mode) {
  case DistributedMode::LocalSubprocess:
    controller.launchLocalSubprocess(options.port);
    break;
  }
}

void shutdownDistributedRuntime() {
  LOG_ASSERT(RuntimeContext::instance().getCurrentHostRuntime() ==
                 HostRuntime::Distributed,
             "Distributed controller can only be launched on distributed host "
             "runtime");
  LOG_ASSERT(ControllerSingleton::launched(),
             "Distributed controller not launched, please launch the "
             "controller before "
             "shutting it down");
  ControllerSingleton::shutdown();
}

SystemDesc getCurrentSystemDesc(
    std::optional<::tt::runtime::DispatchCoreType> dispatchCoreType,
    std::optional<::tt::runtime::Device> deviceHandle) {
  DEBUG_ASSERT(
      ControllerSingleton::launched(),
      "Controller must be launched before calling distributed runtime APIs");
  return ControllerSingleton::get().getCurrentSystemDesc(dispatchCoreType,
                                                         deviceHandle);
}

::tt::runtime::Device
openMeshDevice(const ::tt::runtime::MeshDeviceOptions &options) {
  DEBUG_ASSERT(
      ControllerSingleton::launched(),
      "Controller must be launched before calling distributed runtime APIs");
  return ControllerSingleton::get().openMeshDevice(options);
}

void closeMeshDevice(::tt::runtime::Device parentMesh) {
  DEBUG_ASSERT(
      ControllerSingleton::launched(),
      "Controller must be launched before calling distributed runtime APIs");
  ControllerSingleton::get().closeMeshDevice(parentMesh);
}

::tt::runtime::Tensor
createOwnedHostTensor(const void *data, const std::vector<std::uint32_t> &shape,
                      const std::vector<std::uint32_t> &stride,
                      std::uint32_t itemsize, ::tt::target::DataType dataType) {
  DEBUG_ASSERT(
      ControllerSingleton::launched(),
      "Controller must be launched before calling distributed runtime APIs");
  return ControllerSingleton::get().createOwnedHostTensor(data, shape, stride,
                                                          itemsize, dataType);
}

::tt::runtime::Layout getLayout(::tt::runtime::Binary executableHandle,
                                std::uint32_t programIndex,
                                std::uint32_t inputIndex) {
  DEBUG_ASSERT(
      ControllerSingleton::launched(),
      "Controller must be launched before calling distributed runtime APIs");
  return ControllerSingleton::get().getLayout(executableHandle, programIndex,
                                              inputIndex);
}

::tt::runtime::Tensor toLayout(::tt::runtime::Tensor tensor,
                               ::tt::runtime::Device device,
                               ::tt::runtime::Layout layout,
                               std::optional<bool> retain) {
  DEBUG_ASSERT(
      ControllerSingleton::launched(),
      "Controller must be launched before calling distributed runtime APIs");
  return ControllerSingleton::get().toLayout(tensor, device, layout, retain);
}

std::vector<::tt::runtime::Tensor>
submit(::tt::runtime::Device deviceHandle,
       ::tt::runtime::Binary executableHandle, std::uint32_t programIndex,
       std::vector<::tt::runtime::Tensor> &inputs) {
  DEBUG_ASSERT(
      ControllerSingleton::launched(),
      "Controller must be launched before calling distributed runtime APIs");
  return ControllerSingleton::get().submit(deviceHandle, executableHandle,
                                           programIndex, inputs);
}

std::vector<::tt::runtime::Tensor>
toHost(const ::tt::runtime::Tensor &tensorHandle, bool untilize,
       bool blocking) {
  DEBUG_ASSERT(
      ControllerSingleton::launched(),
      "Controller must be launched before calling distributed runtime APIs");
  return ControllerSingleton::get().toHost(tensorHandle, untilize, blocking);
}

void memcpy(void *dst, const ::tt::runtime::Tensor &srcHandle,
            std::optional<tt::target::DataType> targetDataType) {
  DEBUG_ASSERT(
      ControllerSingleton::launched(),
      "Controller must be launched before calling distributed runtime APIs");
  ControllerSingleton::get().memcpy(dst, srcHandle, targetDataType);
}

} // namespace tt::runtime::distributed
