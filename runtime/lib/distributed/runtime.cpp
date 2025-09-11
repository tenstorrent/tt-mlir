// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/common/runtime_context.h"
#include "tt/runtime/detail/distributed/distributed.h"
#include "tt/runtime/detail/distributed/server/server.h"

namespace tt::runtime::distributed {

using Server = tt::runtime::distributed::server::Server;

class ServerSingleton {
public:
  static Server &get() {
    if (!server_) {
      server_ = std::make_unique<Server>();
    }
    return *server_;
  }

  static void shutdown() { server_.reset(); }

  static bool launched() { return server_ != nullptr; }

private:
  ServerSingleton() = default;
  ~ServerSingleton() = default;

  static inline std::unique_ptr<Server> server_ = nullptr;
};

void launchDistributedRuntime(const DistributedOptions &options) {
  LOG_ASSERT(
      RuntimeContext::instance().getCurrentHostRuntime() ==
          HostRuntime::Distributed,
      "Distributed server can only be launched on distributed host runtime");
  LOG_ASSERT(!ServerSingleton::launched(),
             "Distributed server already launched, please shutdown the server "
             "before launching a new one");
  Server &server = ServerSingleton::get();
  switch (options.mode) {
  case DistributedMode::LocalSubprocess:
    server.launchLocalSubprocess(options.port);
    break;
  }
}

void shutdownDistributedRuntime() {
  LOG_ASSERT(
      RuntimeContext::instance().getCurrentHostRuntime() ==
          HostRuntime::Distributed,
      "Distributed server can only be launched on distributed host runtime");
  LOG_ASSERT(ServerSingleton::launched(),
             "Distributed server not launched, please launch the server before "
             "shutting it down");
  ServerSingleton::shutdown();
}

SystemDesc getCurrentSystemDesc(
    std::optional<::tt::runtime::DispatchCoreType> dispatchCoreType,
    std::optional<::tt::runtime::Device> deviceHandle) {
  DEBUG_ASSERT(
      ServerSingleton::launched(),
      "Server must be launched before calling distributed runtime APIs");
  return ServerSingleton::get().getCurrentSystemDesc(dispatchCoreType,
                                                     deviceHandle);
}

::tt::runtime::Device
openMeshDevice(const ::tt::runtime::MeshDeviceOptions &options) {
  DEBUG_ASSERT(
      ServerSingleton::launched(),
      "Server must be launched before calling distributed runtime APIs");
  return ServerSingleton::get().openMeshDevice(options);
}

void closeMeshDevice(::tt::runtime::Device parentMesh) {
  DEBUG_ASSERT(
      ServerSingleton::launched(),
      "Server must be launched before calling distributed runtime APIs");
  ServerSingleton::get().closeMeshDevice(parentMesh);
}

::tt::runtime::Tensor
createOwnedHostTensor(const void *data, const std::vector<std::uint32_t> &shape,
                      const std::vector<std::uint32_t> &stride,
                      std::uint32_t itemsize, ::tt::target::DataType dataType) {
  DEBUG_ASSERT(
      ServerSingleton::launched(),
      "Server must be launched before calling distributed runtime APIs");
  return ServerSingleton::get().createOwnedHostTensor(data, shape, stride,
                                                      itemsize, dataType);
}

::tt::runtime::Layout getLayout(::tt::runtime::Binary executableHandle,
                                std::uint32_t programIndex,
                                std::uint32_t inputIndex) {
  DEBUG_ASSERT(
      ServerSingleton::launched(),
      "Server must be launched before calling distributed runtime APIs");
  return ServerSingleton::get().getLayout(executableHandle, programIndex,
                                          inputIndex);
}

::tt::runtime::Tensor toLayout(::tt::runtime::Tensor tensor,
                               ::tt::runtime::Device device,
                               ::tt::runtime::Layout layout,
                               std::optional<bool> retain) {
  DEBUG_ASSERT(
      ServerSingleton::launched(),
      "Server must be launched before calling distributed runtime APIs");
  return ServerSingleton::get().toLayout(tensor, device, layout, retain);
}

std::vector<::tt::runtime::Tensor>
submit(::tt::runtime::Device deviceHandle,
       ::tt::runtime::Binary executableHandle, std::uint32_t programIndex,
       std::vector<::tt::runtime::Tensor> &inputs) {
  DEBUG_ASSERT(
      ServerSingleton::launched(),
      "Server must be launched before calling distributed runtime APIs");
  return ServerSingleton::get().submit(deviceHandle, executableHandle,
                                       programIndex, inputs);
}

std::vector<::tt::runtime::Tensor>
toHost(const ::tt::runtime::Tensor &tensorHandle, bool untilize,
       bool blocking) {
  DEBUG_ASSERT(
      ServerSingleton::launched(),
      "Server must be launched before calling distributed runtime APIs");
  return ServerSingleton::get().toHost(tensorHandle, untilize, blocking);
}

void memcpy(void *dst, const ::tt::runtime::Tensor &srcHandle,
            std::optional<tt::target::DataType> targetDataType) {
  DEBUG_ASSERT(
      ServerSingleton::launched(),
      "Server must be launched before calling distributed runtime APIs");
  ServerSingleton::get().memcpy(dst, srcHandle, targetDataType);
}

} // namespace tt::runtime::distributed
