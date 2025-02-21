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

tt::target::DataType getTensorDataType(Tensor tensor);

size_t getNumAvailableDevices();

Device
openDevice(DeviceIds const &deviceIds, size_t numHWCQs = 1,
           std::optional<size_t> l1SmallSize = std::nullopt,
           std::optional<DispatchCoreType> dispatchCoreType = std::nullopt,
           [[maybe_unused]] std::optional<bool> enableAsyncTTNN = std::nullopt);

void closeDevice(Device device);

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

std::vector<float> getTensorData(Tensor tensor);

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
createBufferFromTensorRef(::tt::tt_metal::IDevice *device,
                          ::tt::target::metal::TensorRef const *tensorRef) {
  ::tt::target::metal::TensorDesc const *tensorDesc = tensorRef->desc();
  ::tt::target::metal::LayoutDesc const *layout = tensorDesc->layout();
  ::tt::target::metal::MemoryDesc const *memoryDesc = layout->memory_desc();
  CoreRangeSet coreRangeSet = toCoreRangeSet(layout->core_range_set());
  auto shardRank = memoryDesc->shape()->size();
  ::tt::target::Dim2d const *tile_shape = memoryDesc->tile_shape();
  std::array<uint32_t, 2> shardShape;
  shardShape[1] = memoryDesc->shape()->Get(shardRank - 1) * tile_shape->x();
  shardShape[0] = tile_shape->y();
  for (unsigned i = 0; i < shardRank - 1; ++i) {
    shardShape[0] *= layout->memory_desc()->shape()->Get(i);
  }
  ShardSpec shardSpec(coreRangeSet, shardShape);
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

  ShardSpecBuffer shardSpecBuffer(shardSpec, pageShape, tensorShape);
  assert(memoryDesc->memory_space() == ::tt::target::MemorySpace::DeviceDRAM ||
         memoryDesc->memory_space() == ::tt::target::MemorySpace::DeviceL1);
  BufferType bufferType =
      memoryDesc->memory_space() == ::tt::target::MemorySpace::DeviceDRAM
          ? BufferType::DRAM
          : BufferType::L1;

  tt::target::DataType dataType = memoryDesc->data_type();
  uint64_t itemSize = ::tt::runtime::utils::dataTypeElementSize(dataType);
  uint64_t pageSize = pageShape[0] * pageShape[1] * itemSize;
  uint64_t size = tensorShape[0] * tensorShape[1] * pageSize;
  auto shardedBufferConfig =
      ShardedBufferConfig{.device = device,
                          .size = size,
                          .page_size = pageSize,
                          .buffer_type = bufferType,
                          .buffer_layout = TensorMemoryLayout::BLOCK_SHARDED,
                          .shard_parameters = shardSpecBuffer};

  assert(tensorRef->address());
  std::shared_ptr<::tt::tt_metal::Buffer> buffer =
      ::tt::tt_metal::CreateBuffer(shardedBufferConfig, tensorRef->address());

  return buffer;
}
#pragma clang diagnostic pop

} // namespace tt::runtime::ttmetal

#endif
