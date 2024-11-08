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
} // namespace detail

DeviceRuntime getCurrentRuntime();

std::vector<DeviceRuntime> getAvailableRuntimes();

void setCurrentRuntime(const DeviceRuntime &runtime);

void setCompatibleRuntime(const Binary &binary);

std::pair<SystemDesc, DeviceIds> getCurrentSystemDesc();

Tensor createTensor(std::shared_ptr<void> data,
                    std::vector<std::uint32_t> const &shape,
                    std::vector<std::uint32_t> const &stride,
                    std::uint32_t itemsize, ::tt::target::DataType dataType);

Tensor
createTensor(std::vector<std::shared_ptr<void>> &data,
             std::vector<std::uint32_t> const &shape,
             std::vector<std::uint32_t> const &stride, std::uint32_t itemsize,
             ::tt::target::DataType dataType,
             std::unordered_map<std::string, std::string> const &strategy);

inline Tensor createTensor(std::shared_ptr<void> data, TensorDesc const &desc) {
  return createTensor(data, desc.shape, desc.stride, desc.itemsize,
                      desc.dataType);
}

inline Tensor
createTensor(std::vector<std::shared_ptr<void>> &data, TensorDesc const &desc,
             std::unordered_map<std::string, std::string> const &strategy) {
  return createTensor(data, desc.shape, desc.stride, desc.itemsize,
                      desc.dataType, strategy);
}

tt::target::DataType getTensorDataType(Tensor tensor);

size_t getNumAvailableDevices();

Device openDevice(DeviceIds const &deviceIds, size_t numHWCQs = 1);

void closeDevice(Device device);

void wait(Event event);

Tensor toHost(Tensor tensor, bool untilize = false);

Tensor toDevice(Tensor tensor, Device device);

Tensor toDevice(Tensor tensor, Device device, Layout layout);

Tensor toLayout(Tensor tensor, Layout layout);

Layout getLayout(Binary executableHandle, std::uint32_t programIndex,
                 std::uint32_t inputIndex);

std::string getOpDebugString(OpContext opContextHandle);

Tensor getOpOutputTensor(OpContext opContextHandle,
                         CallbackContext programContextHandle);

std::vector<float> getTensorData(Tensor tensor);

void memcpy(void *dst, Tensor src);

void memcpy(Tensor dst, Tensor src);

void deallocateTensor(Tensor &tensor, bool force = false);

std::vector<Tensor> submit(Device deviceHandle, Binary executableHandle,
                           std::uint32_t programIndex,
                           std::vector<Tensor> const &inputs);

Event submit(Device deviceHandle, Binary executableHandle,
             std::uint32_t programIndex, std::vector<Tensor> const &inputs,
             std::vector<Tensor> const &outputs);

} // namespace tt::runtime

#endif
