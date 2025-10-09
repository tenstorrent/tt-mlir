// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/distributed/controller/controller.h"
#include "tt/runtime/detail/distributed/distributed.h"

namespace tt::runtime::distributed {

using Controller = tt::runtime::distributed::controller::Controller;
using ShutdownResult = tt::runtime::distributed::controller::ShutdownResult;

class ControllerSingleton {
public:
  static Controller &get() {
    if (!controller_) {
      controller_ = std::make_unique<Controller>();
    }
    return *controller_;
  }

  static void shutdown() {
    LOG_ASSERT(controller_ != nullptr, "Distributed controller not launched");
    ShutdownResult result = controller_->shutdown();
    if (!result.success) {
      LOG_FATAL("Distributed controller shutdown failed with message: ",
                result.errorMessage);
    }
    controller_.reset();
  }

  static bool launched() { return controller_ != nullptr; }

private:
  ControllerSingleton() = default;
  ~ControllerSingleton() = default;

  static inline std::unique_ptr<Controller> controller_ = nullptr;
};

static void assertControllerLaunched() {
  LOG_ASSERT(ControllerSingleton::launched(),
             "Distributed controller not launched, please launch the "
             "controller before calling distributed runtime APIs");
}

void launchDistributedRuntime(const DistributedOptions &options) {
  LOG_ASSERT(
      !ControllerSingleton::launched(),
      "Distributed controller already launched, please shutdown the controller "
      "before launching a new one");
  Controller &controller = ControllerSingleton::get();
  controller.launch(options);
}

void shutdownDistributedRuntime() {
  assertControllerLaunched();
  ControllerSingleton::shutdown();
}

SystemDesc getCurrentSystemDesc(
    std::optional<::tt::runtime::DispatchCoreType> dispatchCoreType,
    std::optional<::tt::runtime::Device> deviceHandle) {
  assertControllerLaunched();
  return ControllerSingleton::get().getCurrentSystemDesc(dispatchCoreType,
                                                         deviceHandle);
}

::tt::runtime::Device
openMeshDevice(const ::tt::runtime::MeshDeviceOptions &options) {
  assertControllerLaunched();
  return ControllerSingleton::get().openMeshDevice(options);
}

void closeMeshDevice(::tt::runtime::Device parentMesh) {
  assertControllerLaunched();
  ControllerSingleton::get().closeMeshDevice(parentMesh);
}

::tt::runtime::Tensor
createOwnedHostTensor(const void *data, const std::vector<std::uint32_t> &shape,
                      const std::vector<std::uint32_t> &stride,
                      std::uint32_t itemsize, ::tt::target::DataType dataType) {
  assertControllerLaunched();
  return ControllerSingleton::get().createOwnedHostTensor(data, shape, stride,
                                                          itemsize, dataType);
}

::tt::runtime::Layout getLayout(::tt::runtime::Binary executableHandle,
                                std::uint32_t programIndex,
                                std::uint32_t inputIndex) {
  assertControllerLaunched();
  return ControllerSingleton::get().getLayout(executableHandle, programIndex,
                                              inputIndex);
}

::tt::runtime::Tensor toLayout(::tt::runtime::Tensor tensor,
                               ::tt::runtime::Device device,
                               ::tt::runtime::Layout layout,
                               std::optional<bool> retain) {
  assertControllerLaunched();
  return ControllerSingleton::get().toLayout(tensor, device, layout, retain);
}

std::vector<::tt::runtime::Tensor>
submit(::tt::runtime::Device deviceHandle,
       ::tt::runtime::Binary executableHandle, std::uint32_t programIndex,
       std::vector<::tt::runtime::Tensor> &inputs) {
  assertControllerLaunched();
  return ControllerSingleton::get().submit(deviceHandle, executableHandle,
                                           programIndex, inputs);
}

std::vector<::tt::runtime::Tensor>
toHost(const ::tt::runtime::Tensor &tensorHandle, bool untilize,
       bool blocking) {
  assertControllerLaunched();
  return ControllerSingleton::get().toHost(tensorHandle, untilize, blocking);
}

void memcpy(void *dst, const ::tt::runtime::Tensor &srcHandle,
            std::optional<tt::target::DataType> targetDataType) {
  assertControllerLaunched();
  ControllerSingleton::get().memcpy(dst, srcHandle, targetDataType);
}

} // namespace tt::runtime::distributed
