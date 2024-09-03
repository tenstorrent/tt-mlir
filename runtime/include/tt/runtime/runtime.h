// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_RUNTIME_H
#define TT_RUNTIME_RUNTIME_H

#include <cstdint>
#include <functional>
#include <vector>

#include "tt/runtime/types.h"

namespace tt::runtime {

namespace system_desc {
std::pair<SystemDesc, DeviceIds> getCurrentSystemDesc();
} // namespace system_desc

namespace detail {
void deallocateBuffers(Device device);
}

DeviceRuntime getCurrentRuntime();

std::vector<DeviceRuntime> getAvailableRuntimes();

void setCurrentRuntime(const DeviceRuntime &runtime);

void setCompatibleRuntime(const Binary &binary);

std::pair<SystemDesc, DeviceIds> getCurrentSystemDesc();

Tensor createTensor(std::shared_ptr<void> data,
                    std::vector<std::uint32_t> const &shape,
                    std::vector<std::uint32_t> const &stride,
                    std::uint32_t itemsize, ::tt::target::DataType dataType);

inline Tensor createTensor(std::shared_ptr<void> data, TensorDesc const &desc) {
  return createTensor(data, desc.shape, desc.stride, desc.itemsize,
                      desc.dataType);
}

tt::target::DataType getTensorDataType(Tensor tensor);

Device openDevice(std::vector<int> const &deviceIds = {0},
                  std::vector<std::uint8_t> const &numHWCQs = {});

void closeDevice(Device device);

Event submit(Device device, Binary executable, std::uint32_t programIndex,
             std::vector<Tensor> const &inputs,
             std::vector<Tensor> const &outputs);

void wait(Event event);

} // namespace tt::runtime

#endif
