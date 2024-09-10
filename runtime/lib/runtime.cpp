// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/runtime.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Version.h"

#if defined(TT_RUNTIME_ENABLE_TTNN)
#include "tt/runtime/detail/ttnn.h"
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
  throw std::runtime_error("runtime is not enabled");
}
} // namespace detail

DeviceRuntime getCurrentRuntime() {
#if !defined(TT_RUNTIME_ENABLE_TTNN)
  assert(detail::globalCurrentRuntime != DeviceRuntime::TTNN);
#endif
#if !defined(TT_RUNTIME_ENABLE_TTMETAL)
  assert(detail::globalCurrentRuntime != DeviceRuntime::TTMetal);
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
  assert(runtime != DeviceRuntime::TTNN);
#endif
#if !defined(TT_RUNTIME_ENABLE_TTMETAL)
  assert(runtime != DeviceRuntime::TTMetal);
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
  throw std::runtime_error(
      "Unsupported binary file identifier or runtime not enabled");
}

std::pair<SystemDesc, DeviceIds> getCurrentSystemDesc() {
#if defined(TT_RUNTIME_ENABLE_TTNN) || defined(TT_RUNTIME_ENABLE_TTMETAL)
  return system_desc::getCurrentSystemDesc();
#endif
  throw std::runtime_error("runtime is not enabled");
}

Tensor createTensor(std::shared_ptr<void> data,
                    std::vector<std::uint32_t> const &shape,
                    std::vector<std::uint32_t> const &stride,
                    std::uint32_t itemsize, ::tt::target::DataType dataType) {
  assert(not shape.empty());
  assert(not stride.empty());
  assert(itemsize > 0);
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
  throw std::runtime_error("runtime is not enabled");
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
  throw std::runtime_error("runtime is not enabled");
}

Device openDevice(std::vector<int> const &deviceIds,
                  std::vector<std::uint8_t> const &numHWCQs) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::openDevice(deviceIds, numHWCQs);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::openDevice(deviceIds, numHWCQs);
  }
#endif
  throw std::runtime_error("runtime is not enabled");
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
  throw std::runtime_error("runtime is not enabled");
}

Event submit(Device deviceHandle, Binary executableHandle,
             std::uint32_t programIndex,
             std::vector<Tensor> const &inputHandles,
             std::vector<Tensor> const &outputHandles) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::submit(deviceHandle, executableHandle,
                                       programIndex, inputHandles,
                                       outputHandles);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::submit(deviceHandle, executableHandle,
                                          programIndex, inputHandles,
                                          outputHandles);
  }
#endif
  throw std::runtime_error("runtime is not enabled");
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
    throw std::runtime_error("Currently not supported after refactor");
  }
#endif

  throw std::runtime_error("runtime is not enabled");
}

void wait(Event event) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::wait(event);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    return ::tt::runtime::ttmetal::wait(event);
  }
#endif
  throw std::runtime_error("runtime is not enabled");
}

Tensor toLayout(Device device, Binary executable, std::uint32_t programIndex,
                std::uint32_t inputIndex, Tensor const &input) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  if (getCurrentRuntime() == DeviceRuntime::TTNN) {
    return ::tt::runtime::ttnn::toLayout(device, executable, programIndex,
                                         inputIndex, input);
  }
#endif

#if defined(TT_RUNTIME_ENABLE_TTMETAL)
  if (getCurrentRuntime() == DeviceRuntime::TTMetal) {
    throw std::runtime_error("Not implemented");
  }

#endif
  throw std::runtime_error("runtime is not enabled");
}
} // namespace tt::runtime
