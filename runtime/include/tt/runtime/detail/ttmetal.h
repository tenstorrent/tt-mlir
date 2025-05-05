// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTMETAL_H
#define TT_RUNTIME_DETAIL_TTMETAL_H

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

Tensor createTensor(std::shared_ptr<void> data,
                    const std::vector<std::uint32_t> &shape,
                    const std::vector<std::uint32_t> &stride,
                    std::uint32_t itemsize, ::tt::target::DataType dataType);

inline Tensor createTensor(std::shared_ptr<void> data, const TensorDesc &desc) {
  return createTensor(data, desc.shape, desc.stride, desc.itemsize,
                      desc.dataType);
}

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

Event submit(Device deviceHandle, Binary executableHandle,
             std::uint32_t programIndex, const std::vector<Tensor> &inputs,
             const std::vector<Tensor> &outputs);

std::string getOpDebugString(OpContext opContextHandle);

std::string getOpLocInfo(OpContext opContextHandle);

Tensor getOpOutputTensor(OpContext opContextHandle,
                         CallbackContext programContextHandle);

using InputBuffer =
    std::tuple<std::uint32_t, std::shared_ptr<::tt::tt_metal::Buffer>,
               std::shared_ptr<::tt::tt_metal::Event>>;

using OutputBuffer =
    std::tuple<std::uint32_t, std::shared_ptr<::tt::tt_metal::Buffer>>;

std::shared_ptr<::tt::tt_metal::Event>
executeCommandQueue(::tt::tt_metal::IDevice *device,
                    const ::tt::target::metal::CommandQueue *cq,
                    std::size_t cq_id, const std::vector<InputBuffer> &inputs,
                    const std::vector<OutputBuffer> &outputs);

// Utils

inline CoreRangeSet toCoreRangeSet(
    const ::flatbuffers::Vector<const tt::target::Dim2dRange *> *coreRangeSet) {
  std::set<CoreRange> coreRanges;
  for (const ::tt::target::Dim2dRange *coreRange : *coreRangeSet) {
    CoreCoord start(coreRange->loc().x(), coreRange->loc().y());
    // End is inclusive
    CoreCoord end(coreRange->loc().x() + coreRange->size().x() - 1,
                  coreRange->loc().y() + coreRange->size().y() - 1);
    coreRanges.emplace(start, end);
  }
  return CoreRangeSet(coreRanges);
}

#pragma clang diagnostic push
// Needed to construct ShardedBufferConfig
#pragma clang diagnostic ignored "-Wc++20-designator"

inline std::shared_ptr<::tt::tt_metal::Buffer>
createBufferFromTensorRef(::tt::tt_metal::IDevice *device,
                          const ::tt::target::metal::TensorRef *tensorRef) {
  const ::tt::target::metal::TensorDesc *tensorDesc = tensorRef->desc();
  const ::tt::target::metal::LayoutDesc *layout = tensorDesc->layout();
  const ::tt::target::metal::MemoryDesc *memoryDesc = layout->memory_desc();
  CoreRangeSet coreRangeSet = toCoreRangeSet(layout->core_range_set());
  auto shardRank = memoryDesc->shape()->size();
  const ::tt::target::Dim2d *tile_shape = memoryDesc->tile_shape();
  std::array<uint32_t, 2> shardShape;
  shardShape[1] = memoryDesc->shape()->Get(shardRank - 1) * tile_shape->x();
  shardShape[0] = tile_shape->y();
  for (unsigned i = 0; i < shardRank - 1; ++i) {
    shardShape[0] *= layout->memory_desc()->shape()->Get(i);
  }
  ::tt::tt_metal::ShardSpec shardSpec(coreRangeSet, shardShape);
  std::array<uint32_t, 2> pageShape = {static_cast<uint32_t>(tile_shape->y()),
                                       shardShape[1]};

  auto tensorRank = tensorDesc->shape()->size();
  auto innerDim = tensorDesc->shape()->Get(tensorRank - 1);
  assert(tensorDesc->shape()->size() >= 2);

  uint32_t outerElements = 1;
  for (size_t i = 0; i < tensorRank - 1; i++) {
    outerElements *= tensorDesc->shape()->Get(i);
  }

  assert(outerElements % pageShape[0] == 0);
  assert(innerDim % pageShape[1] == 0);

  std::array<uint32_t, 2> tensorShape = {
      outerElements / pageShape[0],
      innerDim / pageShape[1],
  };

  ::tt::tt_metal::ShardSpecBuffer shardSpecBuffer(shardSpec, pageShape,
                                                  tensorShape);
  assert(memoryDesc->memory_space() == ::tt::target::MemorySpace::DeviceDRAM ||
         memoryDesc->memory_space() == ::tt::target::MemorySpace::DeviceL1);
  ::tt::tt_metal::BufferType bufferType =
      memoryDesc->memory_space() == ::tt::target::MemorySpace::DeviceDRAM
          ? ::tt::tt_metal::BufferType::DRAM
          : ::tt::tt_metal::BufferType::L1;

  tt::target::DataType dataType = memoryDesc->data_type();
  uint64_t itemSize = ::tt::runtime::utils::dataTypeElementSize(dataType);
  uint64_t pageSize = pageShape[0] * pageShape[1] * itemSize;
  uint64_t size = tensorShape[0] * tensorShape[1] * pageSize;
  auto shardedBufferConfig = ::tt::tt_metal::ShardedBufferConfig{
      .device = device,
      .size = size,
      .page_size = pageSize,
      .buffer_type = bufferType,
      .buffer_layout = ::tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
      .shard_parameters = shardSpecBuffer};

  assert(tensorRef->address());
  std::shared_ptr<::tt::tt_metal::Buffer> buffer =
      ::tt::tt_metal::CreateBuffer(shardedBufferConfig, tensorRef->address());

  return buffer;
}
#pragma clang diagnostic pop

} // namespace tt::runtime::ttmetal

#endif
