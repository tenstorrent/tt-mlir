// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTMETAL_TTMETAL_H
#define TT_RUNTIME_DETAIL_TTMETAL_TTMETAL_H

#define FMT_HEADER_ONLY
#include "tt-metalium/circular_buffer.hpp"
#include "tt-metalium/event.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/memory_reporter.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/tt_metal.hpp"

#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/TTMetal/Target.h"

namespace tt::runtime::ttmetal {

using DeviceBuffer = std::shared_ptr<::tt::tt_metal::Buffer>;
using MetalTensor = std::variant<TensorDesc, DeviceBuffer>;

Tensor createBorrowedHostTensor(std::shared_ptr<void> data,
                                const TensorDesc &desc);

inline Tensor createOwnedHostTensor(const void *data, const TensorDesc &desc) {
  std::shared_ptr<void> owned = utils::malloc_shared(desc.size());
  std::memcpy(owned.get(), data, desc.size());
  return ttmetal::createBorrowedHostTensor(owned, desc);
}

inline Tensor createOwnedHostTensor(std::shared_ptr<void> data,
                                    const TensorDesc &desc) {
  return ttmetal::createOwnedHostTensor(data.get(), desc);
}

inline Tensor createBorrowedHostTensor(void *data, const TensorDesc &desc) {
  return ttmetal::createBorrowedHostTensor(utils::unsafe_borrow_shared(data),
                                           desc);
}

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

void deallocateBuffers(Device device);

void dumpMemoryReport(Device device);

std::unordered_map<tt::runtime::MemoryBufferType, tt::runtime::MemoryView>
getMemoryView(Device device, int deviceID = 0);

void wait(Event event);

void wait(Tensor tensor);

void wait(const std::vector<Tensor> &tensors);

std::vector<Tensor> toHost(Tensor tensor, bool untilize);

void memcpy(void *dst, Tensor src);

void memcpy(Tensor dst, Tensor src);

void deallocateTensor(Tensor &tensor, bool force);

std::vector<Tensor> submit(Device deviceHandle, Binary executableHandle,
                           std::uint32_t programIndex,
                           std::vector<Tensor> &inputs);

std::string getOpDebugString(OpContext opContextHandle);

std::string getOpLocInfo(OpContext opContextHandle);

Tensor getOpOutputTensor(OpContext opContextHandle,
                         CallbackContext programContextHandle);

} // namespace tt::runtime::ttmetal

#endif
