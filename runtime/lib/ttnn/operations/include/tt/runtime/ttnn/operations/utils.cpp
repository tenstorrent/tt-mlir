// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::utils {

bool isOnHost(const ::ttnn::Tensor &tensor) {
  // Currently only supports borrowed or owned host storage
  return tensor.storage_type() == ::tt::tt_metal::StorageType::BORROWED or
         tensor.storage_type() == ::tt::tt_metal::StorageType::OWNED;
}

bool isOnDevice(const ::ttnn::Tensor &tensor) {
  // Currently only supports single device storage
  return tensor.storage_type() == ::tt::tt_metal::StorageType::DEVICE;
}

::ttnn::DataType getDataType(const ::tt::target::TensorRef *tensorRef) {
  return ::tt::runtime::ttnn::utils::toTTNNDataType(
      tensorRef->desc()->layout()->memory_desc()->data_type());
}
::ttnn::Device &getDevice(const ::tt::target::DeviceRef *deviceRef,
                          DeviceMap &devicePool) {
  uint32_t deviceId = deviceRef->global_id();
  assert(devicePool.contains(deviceId) && "Device not found in device pool");
  return *devicePool.at(deviceId);
}

CoreRangeSet toCoreRangeSet(
    const ::flatbuffers::Vector<const tt::target::Dim2dRange *> *coreRangeSet) {
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

::tt::tt_metal::MemoryConfig
createMemoryConfig(const ::tt::target::TensorRef *tensorRef) {
  const ::tt::target::LayoutDesc *layout = tensorRef->desc()->layout();
  const ::tt::target::TensorMemoryLayout targetMemoryLayout =
      layout->memory_desc()->memory_layout();
  const ::tt::target::MemorySpace targetMemorySpace =
      layout->memory_desc()->memory_space();
  const ::flatbuffers::Vector<const tt::target::Dim2dRange *>
      *targetCoreRangeSet = layout->core_range_set();
  const ::flatbuffers::Vector<int32_t> *targetShardShape =
      layout->memory_desc()->shape();

  // Print layout
    // std::cout << "Memory layout: " << *layout << std::endl;

  // TODO (jnie): Hardcoding to interleaved and block sharded for now
  // Add support for other types once compiler supports it
  // assert(targetMemoryLayout == ::tt::target::TensorMemoryLayout::Interleaved ||
  //        targetMemoryLayout == ::tt::target::TensorMemoryLayout::BlockSharded);
  // assert(targetMemoryLayout != target::TensorMemoryLayout::BlockSharded ||
  //        targetMemorySpace == target::MemorySpace::DeviceL1 &&
  //            "Only L1 memory space supports sharded memory layout");
  // assert(targetCoreRangeSet->size() == 1 &&
  //        "Currently only single core range/grid is supported");
  // assert(targetShardShape->size() == 2 &&
  //        "Only 2D shard shape is supported in TTNN backend");

  CoreRangeSet ttnnCoreRangeSet = toCoreRangeSet(targetCoreRangeSet);
  std::array<uint32_t, 2> ttnnShardShape;
  std::copy(targetShardShape->begin(), targetShardShape->end(),
            ttnnShardShape.begin());

    

    // aaa
  if (targetMemoryLayout == ::tt::target::TensorMemoryLayout::BlockSharded) {
    // round shape up to 32x32
    // ttnnShardShape = {
    //     ::tt::constants::TILE_HEIGHT *
    //         ((ttnnShardShape[0] + ::tt::constants::TILE_HEIGHT - 1) /
    //          ::tt::constants::TILE_HEIGHT),
    //     ::tt::constants::TILE_WIDTH *
    //         ((ttnnShardShape[1] + ::tt::constants::TILE_WIDTH - 1) /
    //          ::tt::constants::TILE_WIDTH)};
    // assert(ttnnShardShape[0] % ::tt::constants::TILE_HEIGHT == 0 &&
    //        ttnnShardShape[1] % ::tt::constants::TILE_WIDTH == 0 &&
    //        "Shard shape must divide tile shape (32, 32) evenly");
  }

  ::tt::tt_metal::ShardSpec shardSpec(
      ttnnCoreRangeSet, ttnnShardShape,
      ::tt::tt_metal::ShardOrientation::ROW_MAJOR, false);

  ::tt::tt_metal::TensorMemoryLayout ttnnMemLayout =
      ::tt::runtime::ttnn::utils::toTTNNTensorMemoryLayout(targetMemoryLayout);

  ::tt::tt_metal::BufferType ttnnBufferType =
      ::tt::runtime::ttnn::utils::toTTNNBufferType(targetMemorySpace);

  return {ttnnMemLayout, ttnnBufferType, shardSpec};
}

} // namespace tt::runtime::ttnn::operations::utils
