// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_RUNTIME_H
#define TT_RUNTIME_RUNTIME_H

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "tt/runtime/types.h"

namespace tt::runtime {

namespace system_desc {

SystemDesc getCurrentSystemDesc(
    std::optional<DispatchCoreType> dispatchCoreType = std::nullopt,
    std::optional<Device> meshDevice = std::nullopt);
} // namespace system_desc

namespace detail {

uint32_t getNumShards(Tensor tensor);

} // namespace detail

void setMlirHome(std::string_view mlirHome);
void setMetalHome(std::string_view metalHome);

std::vector<DeviceRuntime> getAvailableDeviceRuntimes();
DeviceRuntime getCurrentDeviceRuntime();
void setCurrentDeviceRuntime(const DeviceRuntime &runtime);
void setCompatibleDeviceRuntime(const Binary &binary);

std::vector<HostRuntime> getAvailableHostRuntimes();
HostRuntime getCurrentHostRuntime();
void setCurrentHostRuntime(const HostRuntime &runtime);

SystemDesc getCurrentSystemDesc(
    std::optional<DispatchCoreType> dispatchCoreType = std::nullopt,
    std::optional<Device> meshDevice = std::nullopt);

void launchDistributedRuntime(const DistributedOptions &options = {});
void shutdownDistributedRuntime();

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

// Creates multi-device host tensor with owned storage (buffers of the tensor
// are on the host and their allocation/deallocation is owned by this tensor
// instance).
Tensor createMultiDeviceHostTensor(
    const std::vector<const void *> &data,
    const std::vector<std::uint32_t> &shape,
    const std::vector<std::uint32_t> &stride, std::uint32_t itemsize,
    ::tt::target::DataType dataType,
    const std::unordered_map<std::string, std::string> &strategy,
    const std::vector<uint32_t> &meshShape);

// Creates multi-device host tensor from already existing host tensor shards.
// Tensor shards can be host tensors with either owned or borrowed storage.
Tensor createMultiDeviceHostTensor(
    const std::vector<Tensor> &tensorShards,
    const std::unordered_map<std::string, std::string> &strategy,
    const std::vector<uint32_t> &meshShape);

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
    const std::unordered_map<std::string, std::string> &strategy,
    const std::vector<uint32_t> &meshShape) {
  return ::tt::runtime::createMultiDeviceHostTensor(
      data, desc.shape, desc.stride, desc.itemsize, desc.dataType, strategy,
      meshShape);
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

Arch getArch();

void enablePersistentKernelCache();
void disablePersistentKernelCache();

size_t getNumAvailableDevices();

Device openMeshDevice(const MeshDeviceOptions &options = {});

void closeMeshDevice(Device parentMesh);

Device createSubMeshDevice(Device parentMesh,
                           const std::vector<uint32_t> &meshShape,
                           const std::optional<const std::vector<uint32_t>>
                               &meshOffset = std::nullopt);

void releaseSubMeshDevice(Device subMesh);

void reshapeMeshDevice(Device meshDevice,
                       const std::vector<uint32_t> &meshShape);

std::vector<uint32_t> getMeshShape(Device meshDevice);
std::vector<int> getDeviceIds(Device meshDevice);
size_t getNumHwCqs(Device meshDevice);
bool isProgramCacheEnabled(Device meshDevice);
size_t getL1SmallSize(Device meshDevice);
size_t getTraceRegionSize(Device meshDevice);
size_t getNumDramChannels(Device meshDevice);
size_t getDramSizePerChannel(Device meshDevice);
size_t getL1SizePerCore(Device meshDevice);

void releaseTrace(Device meshDevice, std::uint64_t binaryId,
                  size_t mainProgramId);

void deallocateBuffers(Device device);
void dumpMemoryReport(Device device);
void readDeviceProfilerResults(Device device);

/*
This function gets the memory view per device
  {
    "DRAM": MemoryView,
    "L1": MemoryView,
    "L1Small": MemoryView,
    "Trace": MemoryView
  }
*/
std::unordered_map<tt::runtime::MemoryBufferType, tt::runtime::MemoryView>
getMemoryView(Device device);

void setFabricConfig(FabricConfig config);

void wait(Event event);

void wait(Tensor tensor, std::optional<uint8_t> cqId = std::nullopt);

void wait(const std::vector<Tensor> &tensors,
          std::optional<uint8_t> cqId = std::nullopt);

// Copies device tensor data to host tensor with owned storage, with option to
// untilize data.
std::vector<Tensor> toHost(Tensor tensor, bool untilize = false,
                           bool blocking = true);

Tensor toLayout(Tensor tensor, Device device, Layout layout,
                std::optional<bool> retain = std::nullopt);

bool hasLayout(Tensor tensor, Layout layout);

Layout getLayout(Binary executableHandle, std::uint32_t programIndex,
                 std::uint32_t inputIndex);

void memcpy(void *dst, Tensor src,
            std::optional<tt::target::DataType> targetDataType = std::nullopt);

void memcpy(Tensor dst, Tensor src);

// Deallocates tensor, both device and host. Cannot deallocate host tensors with
// borrowed storage.
void deallocateTensor(Tensor &tensor, bool force = false);

std::string getOpDebugString(OpContext opContextHandle);

std::string getOpLocInfo(OpContext opContextHandle);

std::unordered_map<std::uint32_t, Tensor>
getOpOutputTensor(OpContext opContextHandle,
                  CallbackContext programContextHandle);

// Returns the reference to the output tensor of the current operation.
// In case that operation does not have an output tensor, returns nullopt
// instead.
std::optional<TensorRef> getOpOutputRef(OpContext opContextHandle,
                                        CallbackContext programContextHandle);

// Returns the vector of references to the input tensors of the current
// operation
std::vector<TensorRef> getOpInputRefs(OpContext opContextHandle,
                                      CallbackContext programContextHandle);

// For the given tensor reference, retrieves the tensor from the program's
// tensor pool. Returns the tensor if found, or nullopt if not found or on
// error.
std::optional<Tensor>
retrieveTensorFromPool(CallbackContext programContextHandle,
                       TensorRef tensorRef, bool untilize);

// Updates the tensor in the program's tensor pool that is referenced by the
// given tensor reference. Performs necessary layout and device conversions to
// match the existing tensor.
void updateTensorInPool(CallbackContext programContextHandle,
                        TensorRef tensorRef, Tensor srcTensor);

std::vector<Tensor> submit(Device deviceHandle, Binary executableHandle,
                           std::uint32_t programIndex,
                           std::vector<Tensor> &inputs);

// Dumps tensor data to a file in binary format
void dumpTensor(Tensor tensor, const std::string &filePath);

// Loads tensor data from a binary file
Tensor loadTensor(const std::string &filePath,
                  std::optional<Device> device = std::nullopt);

} // namespace tt::runtime

#endif
