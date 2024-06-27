// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/runtime.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Version.h"

#if defined(TT_RUNTIME_ENABLE_TTNN)
#include "tt/runtime/detail/ttnn.h"
#elif defined(TT_RUNTIME_ENABLE_TTMETAL)
#include "tt/runtime/detail/ttmetal.h"
#endif

namespace tt::runtime {
std::pair<SystemDesc, DeviceIds> getCurrentSystemDesc() {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  return ::tt::runtime::ttnn::getCurrentSystemDesc();
#else
  throw std::runtime_error("runtime is not enabled");
#endif
}

Tensor createTensor(std::shared_ptr<void> data,
                    std::vector<std::uint32_t> const &shape,
                    std::vector<std::uint32_t> const &stride,
                    std::uint32_t itemsize, ::tt::target::DataType dataType) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  return ::tt::runtime::ttnn::createTensor(data, shape, stride, itemsize,
                                           dataType);
#else
  throw std::runtime_error("runtime is not enabled");
#endif
}

Device openDevice(std::vector<int> deviceIds) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  return ::tt::runtime::ttnn::openDevice(deviceIds);
#else
  throw std::runtime_error("runtime is not enabled");
#endif
}

void closeDevice(Device device) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  return ::tt::runtime::ttnn::closeDevice(device);
#else
  throw std::runtime_error("runtime is not enabled");
#endif
}

Event submit(Device deviceHandle, Binary executableHandle,
             std::uint32_t programIndex,
             std::vector<Tensor> const &inputHandles,
             std::vector<Tensor> const &outputHandles) {
#if defined(TT_RUNTIME_ENABLE_TTNN)
  return ::tt::runtime::ttnn::submit(deviceHandle, executableHandle,
                                     programIndex, inputHandles, outputHandles);
#else
  throw std::runtime_error("runtime is not enabled");
#endif
}

void wait(Event) { throw std::runtime_error("Not implemented"); }

} // namespace tt::runtime
