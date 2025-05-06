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
std::pair<SystemDesc, DeviceIds> getCurrentSystemDesc(
    std::optional<DispatchCoreType> dispatchCoreType = std::nullopt,
    std::optional<Device> meshDevice = std::nullopt);
} // namespace system_desc

namespace detail {
void deallocateBuffers(Device device);
void dumpMemoryReport(Device device);

/*
This function get the memory view per device
  {
    "DRAM": MemoryView,
    "L1": MemoryView,
    "L1Small": MemoryView,
    "Trace": MemoryView
  }
*/
std::unordered_map<tt::runtime::MemoryBufferType, tt::runtime::MemoryView>
getMemoryView(Device device, int deviceID = 0);

} // namespace detail

DeviceRuntime getCurrentRuntime();

std::vector<DeviceRuntime> getAvailableRuntimes();

void setCurrentRuntime(const DeviceRuntime &runtime);

void setCompatibleRuntime(const Binary &binary);

std::pair<SystemDesc, DeviceIds> getCurrentSystemDesc(
    std::optional<DispatchCoreType> dispatchCoreType = std::nullopt,
    std::optional<Device> meshDevice = std::nullopt);

// Creates host tensor with a view of the input data (the buffer of the tensor
// is on the host and it was borrowed from an external buffer which is
// responsible for its allocation/deallocation).
Tensor createBorrowedHostTensor(void *data,
                                const std::vector<std::uint32_t> &shape,
                                const std::vector<std::uint32_t> &stride,
                                std::uint32_t itemsize,
                                ::tt::target::DataType dataType);

// Creates host tensor with a owned copy of the data (the buffer of the tensor
// is on the host and its allocation/deallocation is owned by this tensor
// instance).
Tensor createOwnedHostTensor(const void *data,
                             const std::vector<std::uint32_t> &shape,
                             const std::vector<std::uint32_t> &stride,
                             std::uint32_t itemsize,
                             ::tt::target::DataType dataType);

// TODO(mrakita): Should be deprecated but D2M path is using this, investigate
// if it can also use the new `createBorrowedHostTensor` function.
// https://github.com/tenstorrent/tt-mlir/issues/2757
Tensor createTensor(std::shared_ptr<void> data,
                    const std::vector<std::uint32_t> &shape,
                    const std::vector<std::uint32_t> &stride,
                    std::uint32_t itemsize, ::tt::target::DataType dataType);

// Creates multi-device host tensor with owned storage (buffers of the tensor
// are on the host and their allocation/deallocation is owned by this tensor
// instance).
Tensor createMultiDeviceHostTensor(
    const std::vector<const void *> &data,
    const std::vector<std::uint32_t> &shape,
    const std::vector<std::uint32_t> &stride, std::uint32_t itemsize,
    ::tt::target::DataType dataType,
    const std::unordered_map<std::string, std::string> &strategy);

// Creates multi-device host tensor from already existing host tensor shards.
// Tensor shards can be host tensors with either owned or borrowed storage.
Tensor createMultiDeviceHostTensor(
    const std::vector<Tensor> &tensorShards,
    const std::unordered_map<std::string, std::string> &strategy);

// Creates empty tensor on host/device depending on the passed layout.
Tensor createEmptyTensor(Device device, Layout layout,
                         const std::vector<std::uint32_t> &shape,
                         const std::vector<std::uint32_t> &stride,
                         std::uint32_t itemsize);

inline Tensor createBorrowedHostTensor(void *data, const TensorDesc &desc) {
  return ::tt::runtime::createBorrowedHostTensor(data, desc.shape, desc.stride,
                                                 desc.itemsize, desc.dataType);
}

inline Tensor createOwnedHostTensor(const void *data, const TensorDesc &desc) {
  return ::tt::runtime::createOwnedHostTensor(data, desc.shape, desc.stride,
                                              desc.itemsize, desc.dataType);
}

inline Tensor createMultiDeviceHostTensor(
    const std::vector<const void *> &data, const TensorDesc &desc,
    const std::unordered_map<std::string, std::string> &strategy) {
  return ::tt::runtime::createMultiDeviceHostTensor(
      data, desc.shape, desc.stride, desc.itemsize, desc.dataType, strategy);
}

inline Tensor createEmptyTensor(Device device, Layout layout,
                                const TensorDesc &desc) {
  return ::tt::runtime::createEmptyTensor(device, layout, desc.shape,
                                          desc.stride, desc.itemsize);
}

bool isTensorAllocated(Tensor tensor);
tt::target::DataType getTensorDataType(Tensor tensor);
std::vector<std::byte> getTensorDataBuffer(Tensor tensor);
std::uint32_t getTensorElementSize(Tensor tensor);
std::uint32_t getTensorVolume(Tensor tensor);
std::vector<std::uint32_t> getTensorShape(Tensor tensor);
std::vector<std::uint32_t> getTensorStride(Tensor tensor);
TensorDesc getTensorDesc(Tensor tensor);
bool getTensorRetain(Tensor tensor);
void setTensorRetain(Tensor tensor, bool retain);

size_t getNumAvailableDevices();

Device openMeshDevice(const std::vector<uint32_t> &meshShape,
                      const MeshDeviceOptions &options = MeshDeviceOptions());

void closeMeshDevice(Device parentMesh);

Device createSubMeshDevice(Device parentMesh,
                           const std::vector<uint32_t> &meshShape,
                           const std::optional<const std::vector<uint32_t>>
                               &meshOffset = std::nullopt);

void releaseSubMeshDevice(Device subMesh);

void reshapeMeshDevice(Device meshDevice,
                       const std::vector<uint32_t> &meshShape);

void wait(Event event);

void wait(Tensor tensor);

void wait(const std::vector<Tensor> &tensors);

// Copies device tensor data to host tensor with owned storage, with option to
// untilize data.
std::vector<Tensor> toHost(Tensor tensor, bool untilize = false);

Tensor toLayout(Tensor tensor, Device device, Layout layout,
                std::optional<bool> retain = std::nullopt);

Layout getLayout(Binary executableHandle, std::uint32_t programIndex,
                 std::uint32_t inputIndex);

void memcpy(void *dst, Tensor src);

void memcpy(Tensor dst, Tensor src);

// Deallocates tensor, both device and host. Cannot deallocate host tensors with
// borrowed storage.
void deallocateTensor(Tensor &tensor, bool force = false);

std::string getOpDebugString(OpContext opContextHandle);

std::string getOpLocInfo(OpContext opContextHandle);

Tensor getOpOutputTensor(OpContext opContextHandle,
                         CallbackContext programContextHandle);

std::vector<Tensor> submit(Device deviceHandle, Binary executableHandle,
                           std::uint32_t programIndex,
                           std::vector<Tensor> &inputs);

Event submit(Device deviceHandle, Binary executableHandle,
             std::uint32_t programIndex, const std::vector<Tensor> &inputs,
             const std::vector<Tensor> &outputs);

} // namespace tt::runtime

#endif
