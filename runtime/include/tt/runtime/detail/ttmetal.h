// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTMETAL_H
#define TT_RUNTIME_DETAIL_TTMETAL_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-qual"
#pragma clang diagnostic ignored "-Wctad-maybe-unsupported"
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wignored-qualifiers"
#pragma clang diagnostic ignored "-Wvla-extension"
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wc++20-extensions"
#pragma clang diagnostic ignored "-Wc++20-designator"
#pragma clang diagnostic ignored "-Wnon-virtual-dtor"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wsuggest-override"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wreorder-ctor"
#pragma clang diagnostic ignored "-Wmismatched-tags"
#pragma clang diagnostic ignored "-Wunused-lambda-capture"
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wimplicit-fallthrough"
#pragma clang diagnostic ignored "-Wstring-conversion"
#pragma clang diagnostic ignored "-Wunneeded-internal-declaration"
#pragma clang diagnostic ignored "-Wunused-local-typedef"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wpessimizing-move"
#pragma clang diagnostic ignored "-Wparentheses"
#pragma clang diagnostic ignored "-Wdeprecated-volatile"
#pragma clang diagnostic ignored "-Wdeprecated-this-capture"
#pragma clang diagnostic ignored "-Wc++23-extensions"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"
#pragma clang diagnostic ignored "-Wlogical-op-parentheses"
#pragma clang diagnostic ignored "-Wundefined-inline"
#pragma clang diagnostic ignored "-Wzero-length-array"
#define FMT_HEADER_ONLY
#include "distributed/mesh_device.hpp"
#include "impl/event/event.hpp"
#include "tt_metal/host_api.hpp"
#pragma clang diagnostic pop

#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/TTMetal/Target.h"

namespace tt::runtime::ttmetal {

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

size_t getNumAvailableDevices();

Device openDevice(DeviceIds const &deviceIds, size_t numHWCQs = 1);

void closeDevice(Device device);

void deallocateBuffers(Device device);

Event submit(Device device, Binary executable, std::uint32_t programIndex,
             std::vector<Tensor> const &inputs,
             std::vector<Tensor> const &outputs);

void wait(Event event);

using InputBuffer =
    std::tuple<std::uint32_t, std::shared_ptr<::tt::tt_metal::Buffer>,
               std::shared_ptr<::tt::tt_metal::Event>>;

using OutputBuffer =
    std::tuple<std::uint32_t, std::shared_ptr<::tt::tt_metal::Buffer>>;

std::shared_ptr<::tt::tt_metal::Event>
executeCommandQueue(::tt::tt_metal::Device *device,
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
createBufferFromTensorRef(::tt::tt_metal::Device *device,
                          ::tt::target::TensorRef const *tensorRef) {
  ::tt::target::TensorDesc const *tensorDesc = tensorRef->desc();
  ::tt::target::LayoutDesc const *layout = tensorDesc->layout();
  ::tt::target::MemoryDesc const *memoryDesc = layout->memory_desc();
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

  auto tensorRank = layout->stride()->size();
  auto innerDim = layout->stride()->Get(tensorRank - 2);
  assert(layout->stride()->size() >= 2);
  assert((layout->stride()->Get(0) * tensorDesc->shape()->Get(0)) %
             (pageShape[0] * innerDim) ==
         0);
  assert(innerDim % pageShape[1] == 0);
  std::array<uint32_t, 2> tensorShape = {
      (layout->stride()->Get(0) * tensorDesc->shape()->Get(0)) /
          (pageShape[0] * innerDim),
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
