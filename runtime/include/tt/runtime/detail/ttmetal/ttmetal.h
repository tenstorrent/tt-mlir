// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTMETAL_TTMETAL_H
#define TT_RUNTIME_DETAIL_TTMETAL_TTMETAL_H

#define FMT_HEADER_ONLY
#include "tt-metalium/circular_buffer.hpp"
#include "tt-metalium/distributed_host_buffer.hpp"
#include "tt-metalium/event.hpp"
#include "tt-metalium/hal.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/memory_reporter.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/persistent_kernel_cache.hpp"
#include "tt-metalium/program_cache.hpp"
#include "tt-metalium/tt_metal.hpp"

#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/TTMetal/Target.h"

#include <optional>

namespace tt::runtime::ttmetal {

using HostBuffer = std::shared_ptr<::tt::tt_metal::HostBuffer>;
using DistributedHostBuffer =
    std::shared_ptr<::tt::tt_metal::DistributedHostBuffer>;
using MeshBuffer = std::shared_ptr<::tt::tt_metal::distributed::MeshBuffer>;
using MetalTensor =
    std::variant<TensorDesc, HostBuffer, DistributedHostBuffer, MeshBuffer>;

Tensor createBorrowedHostTensor(std::shared_ptr<void> data,
                                const TensorDesc &desc);

inline Tensor createBorrowedHostTensor(void *data, const TensorDesc &desc) {
  return ttmetal::createBorrowedHostTensor(utils::unsafe_borrow_shared(data),
                                           desc);
}

std::shared_ptr<::tt::tt_metal::HostBuffer>
createMetalHostBuffer(const void *data, const std::vector<std::uint32_t> &shape,
                      const size_t sizeBytes,
                      const ::tt::target::DataType dataType);

Tensor createOwnedHostTensor(const void *data,
                             const std::vector<std::uint32_t> &shape,
                             const std::vector<std::uint32_t> &stride,
                             std::uint32_t itemsize,
                             ::tt::target::DataType dataType);

Tensor createMultiDeviceHostTensor(
    const std::vector<Tensor> &tensorShards,
    const std::unordered_map<std::string, std::string> &strategy,
    const std::vector<uint32_t> &meshShape);

Tensor createMultiDeviceHostTensor(
    const std::vector<const void *> &data,
    const std::vector<std::uint32_t> &shape,
    const std::vector<std::uint32_t> &stride, std::uint32_t itemsize,
    ::tt::target::DataType dataType,
    const std::unordered_map<std::string, std::string> &strategy,
    const std::vector<uint32_t> &meshShape);

Layout getLayout(Binary executableHandle, std::uint32_t programIndex,
                 std::uint32_t inputIndex);
Tensor toLayout(Tensor tensor, Device device, Layout layout,
                std::optional<bool> retain);
bool isTensorAllocated(Tensor tensor);
tt::target::DataType getTensorDataType(Tensor tensor);
std::vector<std::byte> getTensorDataBuffer(::tt::runtime::Tensor tensor);
std::vector<std::uint32_t> getTensorShape(::tt::runtime::Tensor tensor);
std::vector<std::uint32_t> getTensorStride(::tt::runtime::Tensor tensor);
std::uint32_t getTensorElementSize(::tt::runtime::Tensor tensor);
std::uint32_t getTensorVolume(::tt::runtime::Tensor tensor);
TensorDesc getTensorDesc(::tt::runtime::Tensor tensor);
HostBuffer getHostBuffer(::tt::runtime::Tensor tensor);
DistributedHostBuffer getDistributedHostBuffer(::tt::runtime::Tensor tensor);
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

void deallocateBuffers(Device device);

void dumpMemoryReport(Device device);

void readDeviceProfilerResults(Device device);

std::unordered_map<tt::runtime::MemoryBufferType, tt::runtime::MemoryView>
getMemoryView(Device device);

void setFabricConfig(FabricConfig config);

void wait(Event event);

void wait(Tensor tensor, std::optional<uint8_t> cqId = std::nullopt);

void wait(const std::vector<Tensor> &tensors,
          std::optional<uint8_t> cqId = std::nullopt);

std::vector<Tensor> toHost(Tensor tensor, bool untilize, bool blocking);

void memcpy(void *dst, Tensor src,
            std::optional<tt::target::DataType> dstDataType = std::nullopt);

void memcpy(Tensor dst, Tensor src);

void memcpy(Tensor dst, TensorDesc dstDesc, Tensor src, TensorDesc srcDesc);

void deallocateTensor(Tensor &tensor, bool force);

std::vector<Tensor> submit(Device deviceHandle, Binary executableHandle,
                           std::uint32_t programIndex,
                           std::vector<Tensor> &inputs);

std::string getOpDebugString(OpContext opContextHandle);

std::string getOpLocInfo(OpContext opContextHandle);

std::unordered_map<std::uint32_t, Tensor>
getOpOutputTensor(OpContext opContextHandle,
                  CallbackContext programContextHandle);

std::optional<tt::runtime::TensorRef>
getOpOutputRef(OpContext opContextHandle, CallbackContext programContextHandle);

std::vector<tt::runtime::TensorRef>
getOpInputRefs(OpContext opContextHandle, CallbackContext programContextHandle);

std::optional<Tensor>
retrieveTensorFromPool(CallbackContext programContextHandle,
                       TensorRef tensorRef, bool untilize);

void updateTensorInPool(CallbackContext programContextHandle,
                        TensorRef tensorRef, Tensor srcTensor);

} // namespace tt::runtime::ttmetal

#endif
