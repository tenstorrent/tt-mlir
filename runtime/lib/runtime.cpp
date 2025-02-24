// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/runtime.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Version.h"

#if defined(TT_RUNTIME_ENABLE_TTNN)
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/types.h"
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
#include "tt/runtime/detail/ttmetal.h"
#endif

namespace tt::runtime {

namespace detail {
// NOLINTBEGIN
#if defined(TT_RUNTIME_ENABLE_TTNN)
DeviceRuntime globalCurrentRuntime = DeviceRuntime::TTNN;
#elif defined(TT_RUNTIME_ENABLE_TTMETAL)
DeviceRuntime globalCurrentRuntime = DeviceRuntime::TTMetal;
#else
DeviceRuntime globalCurrentRuntime = DeviceRuntime::Disabled;
#endif
// NOLINTEND

void deallocateBuffers(Device device) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::deallocateBuffers(device);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::deallocateBuffers(device);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

void dumpMemoryReport(Device device) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::dumpMemoryReport(device);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::dumpMemoryReport(device);
  }
#endif

  LOG_FATAL("runtime is not enabled");
}

std::unordered_map<tt::runtime::MemoryBufferType, tt::runtime::MemoryView>
getMemoryView(Device device, int deviceID) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::getMemoryView(device, deviceID);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::getMemoryView(device, deviceID);
  }
#endif

  LOG_FATAL("runtime is not enabled");
}
} // namespace detail

DeviceRuntime getCurrentRuntime() {
#if !defined(TT_RUNTIME_ENABLE_TTNN)
  LOG_ASSERT(detail::globalCurrentRuntime != DeviceRuntime::TTNN);
#endif
#if !defined(TT_RUNTIME_ENABLE_TTMETAL)
  LOG_ASSERT(detail::globalCurrentRuntime != DeviceRuntime::TTMetal);
#endif
  return detail::globalCurrentRuntime;
}

std::vector<DeviceRuntime> getAvailableRuntimes() {
  std::vector<DeviceRuntime> runtimes;
#if defined(TT_RUNTIME_ENABLE_TTNN)
  runtimes.push_back(DeviceRuntime::TTNN);
#endif
#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  runtimes.push_back(DeviceRuntime::TTMetal);
#endif
  return runtimes;
}

void setCurrentRuntime(const DeviceRuntime &runtime) {
#if !defined(TT_RUNTIME_ENABLE_TTNN)
  LOG_ASSERT(runtime != DeviceRuntime::TTNN);
#endif
#if !defined(TT_RUNTIME_ENABLE_TTMETAL)
  LOG_ASSERT(runtime != DeviceRuntime::TTMetal);
#endif
  detail::globalCurrentRuntime = runtime;
}

void setCompatibleRuntime(const Binary &binary) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (binary.getFileIdentifier() ==
      ::tt::target::ttnn::TTNNBinaryIdentifier()) {
    return setCurrentRuntime(DeviceRuntime::TTNN);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (binary.getFileIdentifier() ==
      ::tt::target::metal::TTMetalBinaryIdentifier()) {
    return setCurrentRuntime(DeviceRuntime::TTMetal);
  }
#endif
  LOG_FATAL("Unsupported binary file identifier or runtime not enabled");
}

std::pair<SystemDesc, DeviceIds>
getCurrentSystemDesc(std::optional<DispatchCoreType> dispatchCoreType) {
#if defined(TT_RUNTIME_ENABLE_TTNN) || defined(TT_RUNTIME_ENABLE_TTMETAL)
  return system_desc::getCurrentSystemDesc(dispatchCoreType);
#endif
  LOG_FATAL("runtime is not enabled");
}

Tensor createOwnedTensor(std::shared_ptr<void> data,
                         std::vector<std::uint32_t> const &shape,
                         std::vector<std::uint32_t> const &stride,
                         std::uint32_t itemsize,
                         ::tt::target::DataType dataType) {
  LOG_ASSERT(not shape.empty());
  LOG_ASSERT(not stride.empty());
  LOG_ASSERT(itemsize > 0);
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::createOwnedTensor(data, shape, stride, itemsize,
                                                  dataType);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    LOG_FATAL("TT Metal runtime does not support creating owned tensors");
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

Tensor createTensor(std::shared_ptr<void> data,
                    std::vector<std::uint32_t> const &shape,
                    std::vector<std::uint32_t> const &stride,
                    std::uint32_t itemsize, ::tt::target::DataType dataType) {
  LOG_ASSERT(not shape.empty());
  LOG_ASSERT(not stride.empty());
  LOG_ASSERT(itemsize > 0);
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::createTensor(data, shape, stride, itemsize,
                                             dataType);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::createTensor(data, shape, stride, itemsize,
                                                dataType);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

Tensor
createTensor(std::vector<std::shared_ptr<void>> &data,
             std::vector<std::uint32_t> const &shape,
             std::vector<std::uint32_t> const &stride, std::uint32_t itemsize,
             ::tt::target::DataType dataType,
             std::unordered_map<std::string, std::string> const &strategy) {
  LOG_ASSERT(not shape.empty());
  LOG_ASSERT(not stride.empty());
  LOG_ASSERT(itemsize > 0);
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::createTensor(data, shape, stride, itemsize,
                                             dataType, strategy);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    LOG_FATAL("Not implemented");
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

Tensor createTensor(Device device, Layout layout,
                    std::vector<std::uint32_t> const &shape,
                    std::vector<std::uint32_t> const &stride,
                    std::uint32_t itemsize) {
  LOG_ASSERT(not shape.empty());
  LOG_ASSERT(not stride.empty());
  LOG_ASSERT(itemsize > 0);
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::createTensor(device, layout, shape, stride,
                                             itemsize);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    LOG_FATAL("Not implemented");
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

tt::target::DataType getTensorDataType(Tensor tensor) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::getTensorDataType(tensor);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::getTensorDataType(tensor);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

size_t getNumAvailableDevices() {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::getNumAvailableDevices();
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::getNumAvailableDevices();
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

Device openDevice(DeviceIds const &deviceIds, size_t numHWCQs,
                  std::optional<size_t> l1SmallSize,
                  std::optional<DispatchCoreType> dispatchCoreType,
                  std::optional<bool> enableAsyncTTNN) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::openDevice(deviceIds, numHWCQs, l1SmallSize,
                                           dispatchCoreType, enableAsyncTTNN);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::openDevice(
        deviceIds, numHWCQs, l1SmallSize, dispatchCoreType, enableAsyncTTNN);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

void closeDevice(Device device) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::closeDevice(device);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::closeDevice(device);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

void wait(Event event) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    LOG_WARNING("wait API will be deprecated for TTNN runtime.");
    return ::tt::runtime::ttnn::wait(event);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::wait(event);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

void wait(Tensor tensor) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::wait(tensor);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::wait(tensor);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

void wait(std::vector<Tensor> const &tensors) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::wait(tensors);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::wait(tensors);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

std::vector<Tensor> multiDeviceToHost(Tensor tensor, bool untilize)
{
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::multiDeviceToHost(tensor, untilize);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    LOG_FATAL("not implemented");
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

Tensor toHost(Tensor tensor, bool untilize) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::toHost(tensor, untilize);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    LOG_FATAL("not implemented");
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

Tensor toLayout(Tensor tensor, Device device, Layout layout) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::toLayout(tensor, device, layout);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    LOG_FATAL("not implemented");
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

Layout getLayout(Binary executableHandle, std::uint32_t programIndex,
                 std::uint32_t inputIndex) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::getLayout(executableHandle, programIndex,
                                          inputIndex);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    LOG_FATAL("not implemented");
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

void memcpy(void *dst, Tensor src) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::memcpy(dst, src);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    LOG_FATAL("not implemented");
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

void memcpy(Tensor dst, Tensor src) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::memcpy(dst, src);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    LOG_FATAL("not implemented");
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

void deallocateTensor(Tensor &tensor, bool force) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::deallocateTensor(tensor, force);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    LOG_FATAL("not implemented");
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

std::string getOpDebugString(OpContext opContextHandle) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::getOpDebugString(opContextHandle);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::getOpDebugString(opContextHandle);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

std::string getOpLocInfo(OpContext opContextHandle) {
#ifdef TT_RUNTIME_ENABLE_TTNN
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::getOpLocInfo(opContextHandle);
  }
#endif

#ifdef TT_RUNTIME_ENABLE_TTMETAL
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::getOpLocInfo(opContextHandle);
  }
#endif
  throw std::runtime_error("runtime is not enabled");
}

Tensor getOpOutputTensor(OpContext opContextHandle,
                         CallbackContext programContextHandle) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::getOpOutputTensor(opContextHandle,
                                                  programContextHandle);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::getOpOutputTensor(opContextHandle,
                                                     programContextHandle);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

std::vector<float> getTensorData(Tensor tensor) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::getTensorData(tensor);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::getTensorData(tensor);
  }
#endif

  LOG_FATAL("runtime is not enabled");
}

std::vector<Tensor> submit(Device deviceHandle, Binary executableHandle,
                           std::uint32_t programIndex,
                           std::vector<Tensor> const &inputHandles) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::submit(deviceHandle, executableHandle,
                                       programIndex, inputHandles);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    LOG_FATAL("not implemented");
  }
#endif
  LOG_FATAL("runtime is not enabled");
}

Event submit(Device deviceHandle, Binary executableHandle,
             std::uint32_t programIndex,
             std::vector<Tensor> const &inputHandles,
             std::vector<Tensor> const &outputHandles) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    LOG_FATAL("This submit API is deprecated for TTNN. Please switch to the "
              "new API.");
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::submit(deviceHandle, executableHandle,
                                          programIndex, inputHandles,
                                          outputHandles);
  }
#endif
  LOG_FATAL("runtime is not enabled");
}
} // namespace tt::runtime
