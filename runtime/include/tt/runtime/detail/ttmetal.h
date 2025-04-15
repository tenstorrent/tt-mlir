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
                    std::vector<std::uint32_t> const &shape,
                    std::vector<std::uint32_t> const &stride,
                    std::uint32_t itemsize, ::tt::target::DataType dataType);

inline Tensor createTensor(std::shared_ptr<void> data, TensorDesc const &desc) {
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

Device createSubMeshDevice(
    Device parentMesh, const std::pair<uint32_t, uint32_t> &meshShape,
    const std::optional<const std::pair<uint32_t, uint32_t>> &meshOffset =
        std::nullopt);

void releaseSubMeshDevice(Device subMesh);

void deallocateBuffers(Device device);

void dumpMemoryReport(Device device);

std::unordered_map<tt::runtime::MemoryBufferType, tt::runtime::MemoryView>
getMemoryView(Device device, int deviceID = 0);

void wait(Event event);

void wait(Tensor tensor);

void wait(std::vector<Tensor> const &tensors);

Event submit(Device deviceHandle, Binary executableHandle,
             std::uint32_t programIndex, std::vector<Tensor> const &inputs,
             std::vector<Tensor> const &outputs);

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
                    ::tt::target::metal::CommandQueue const *cq,
                    std::size_t cq_id, std::vector<InputBuffer> const &inputs,
                    std::vector<OutputBuffer> const &outputs);

// Utils

inline CoreRangeSet toCoreRangeSet(
    ::flatbuffers::Vector<tt::target::Dim2dRange const *> const *coreRangeSet) {
  std::set<CoreRange> coreRanges;
  for (::tt::target::Dim2dRange const *coreRange : *coreRangeSet) {
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
createBufferFromBufferRef(::tt::tt_metal::IDevice *device,
                          ::tt::target::metal::BufferRef const *bufferRef) {
  ::tt::target::metal::BufferDesc const *bufferDesc = bufferRef->desc();
  auto *grid_shape = bufferDesc->grid_shape();
  auto *shard_shape = bufferDesc->shard_shape();
  ::tt::target::Dim2d const *tile_shape = bufferDesc->tile_shape();
  assert(grid_shape->size() == 2);
  assert(shard_shape->size() == 2);
  CoreRangeSet coreRangeSet = toCoreRangeSet(bufferDesc->core_range_set());
  std::array<uint32_t, 2> shardShape = {
      static_cast<uint32_t>(shard_shape->Get(0) * tile_shape->y()),
      static_cast<uint32_t>(shard_shape->Get(1) * tile_shape->x()),
  };
  ::tt::tt_metal::ShardSpec shardSpec(coreRangeSet, shardShape);

  std::array<uint32_t, 2> pageShape = {
      static_cast<uint32_t>(tile_shape->y()),
      shardShape[1],
  };
  std::array<uint32_t, 2> tensorScalarShape = {
      static_cast<uint32_t>(grid_shape->Get(0) * shard_shape->Get(0) *
                            tile_shape->y()),
      static_cast<uint32_t>(grid_shape->Get(1) * shard_shape->Get(1) *
                            tile_shape->x()),
  };
  assert(tensorScalarShape[0] % pageShape[0] == 0);
  assert(tensorScalarShape[1] % pageShape[1] == 0);
  std::array<uint32_t, 2> tensorShapeInPages = {
      tensorScalarShape[0] / pageShape[0],
      tensorScalarShape[1] / pageShape[1],
  };
  ::tt::tt_metal::ShardSpecBuffer shardSpecBuffer(shardSpec, pageShape,
                                                  tensorShapeInPages);

  assert(bufferDesc->memory_space() == ::tt::target::MemorySpace::DeviceDRAM ||
         bufferDesc->memory_space() == ::tt::target::MemorySpace::DeviceL1);
  ::tt::tt_metal::BufferType bufferType =
      bufferDesc->memory_space() == ::tt::target::MemorySpace::DeviceDRAM
          ? ::tt::tt_metal::BufferType::DRAM
          : ::tt::tt_metal::BufferType::L1;

  tt::target::DataType dataType = bufferDesc->data_type();
  uint64_t itemSize = ::tt::runtime::utils::dataTypeElementSize(dataType);
  assert(bufferDesc->page_size() == (pageShape[0] * pageShape[1] * itemSize));
  auto shardedBufferConfig = ::tt::tt_metal::ShardedBufferConfig{
      .device = device,
      .size = bufferDesc->size(),
      .page_size = bufferDesc->page_size(),
      .buffer_type = bufferType,
      .buffer_layout = ::tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
      .shard_parameters = shardSpecBuffer};

  assert(bufferRef->address());
  std::shared_ptr<::tt::tt_metal::Buffer> buffer =
      ::tt::tt_metal::CreateBuffer(shardedBufferConfig, bufferRef->address());

  return buffer;
}
#pragma clang diagnostic pop

} // namespace tt::runtime::ttmetal

#endif
