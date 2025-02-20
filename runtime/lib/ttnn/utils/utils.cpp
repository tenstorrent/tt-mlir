// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/ttnn/utils.h"
#include "tt/runtime/detail/logger.h"

namespace tt::runtime::ttnn::utils {

using ::tt::runtime::DeviceRuntime;

// TODO (bug #701)
// Currently the memory layout/location in flatbuffer is incorrect
// These methods are workarounds for operations such that we query the info
// directly from the TTNN tensor. Ideally, we should be able to get all of this
// info directly from the flatbuffer
bool isOnHost(const ::ttnn::StorageType &storageType) {
  return storageType == ::tt::tt_metal::StorageType::BORROWED or
         storageType == ::tt::tt_metal::StorageType::OWNED or
         storageType == ::tt::tt_metal::StorageType::MULTI_DEVICE_HOST;
}

bool isOnDevice(const ::ttnn::StorageType &storageType) {
  return storageType == ::tt::tt_metal::StorageType::DEVICE or
         storageType == ::tt::tt_metal::StorageType::MULTI_DEVICE;
}

bool isValidTileShape(const ::tt::target::Dim2d *shape) {
  return (shape->x() == 1 and shape->y() == 1) or
         (shape->x() == 32 and shape->y() == 32);
}

::ttnn::DataType toTTNNDataType(::tt::target::DataType dataType) {
  switch (dataType) {
  case ::tt::target::DataType::Float32:
    return ::ttnn::DataType::FLOAT32;
  case ::tt::target::DataType::BFloat16:
    return ::ttnn::DataType::BFLOAT16;
  case ::tt::target::DataType::BFP_BFloat8:
    return ::ttnn::DataType::BFLOAT8_B;
  case ::tt::target::DataType::BFP_BFloat4:
    return ::ttnn::DataType::BFLOAT4_B;
  case ::tt::target::DataType::UInt32:
    return ::ttnn::DataType::UINT32;
  case ::tt::target::DataType::UInt16:
    return ::ttnn::DataType::UINT16;
  case ::tt::target::DataType::UInt8:
    return ::ttnn::DataType::UINT8;

  default:
    LOG_FATAL("Unsupported data type");
  }
}

::tt::target::DataType fromTTNNDataType(::ttnn::DataType dataType) {
  switch (dataType) {
  case ::ttnn::DataType::FLOAT32:
    return ::tt::target::DataType::Float32;
  case ::ttnn::DataType::BFLOAT16:
    return ::tt::target::DataType::BFloat16;
  case ::ttnn::DataType::BFLOAT8_B:
    return ::tt::target::DataType::BFP_BFloat8;
  case ::ttnn::DataType::BFLOAT4_B:
    return ::tt::target::DataType::BFP_BFloat4;
  case ::ttnn::DataType::UINT32:
    return ::tt::target::DataType::UInt32;
  case ::ttnn::DataType::UINT16:
    return ::tt::target::DataType::UInt16;
  case ::ttnn::DataType::UINT8:
    return ::tt::target::DataType::UInt8;

  default:
    LOG_FATAL("Unsupported data type");
  }
}

::ttnn::Layout toTTNNLayout(::tt::target::TensorLayout layout) {
  switch (layout) {
  case ::tt::target::TensorLayout::Tile:
    return ::ttnn::Layout::TILE;
  case ::tt::target::TensorLayout::RowMajor:
    return ::ttnn::Layout::ROW_MAJOR;
  default:
    LOG_FATAL("Unsupported layout");
  }
}

::ttnn::TensorMemoryLayout
toTTNNTensorMemoryLayout(::tt::target::TensorMemoryLayout tensorMemoryLayout) {

  switch (tensorMemoryLayout) {
  case ::tt::target::TensorMemoryLayout::Interleaved:
    return ::ttnn::TensorMemoryLayout::INTERLEAVED;
  case ::tt::target::TensorMemoryLayout::SingleBank:
    return ::ttnn::TensorMemoryLayout::SINGLE_BANK;
  case ::tt::target::TensorMemoryLayout::HeightSharded:
    return ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED;
  case ::tt::target::TensorMemoryLayout::WidthSharded:
    return ::ttnn::TensorMemoryLayout::WIDTH_SHARDED;
  case ::tt::target::TensorMemoryLayout::BlockSharded:
    return ::ttnn::TensorMemoryLayout::BLOCK_SHARDED;
  case ::tt::target::TensorMemoryLayout::None:
    LOG_FATAL("Unsupported tensor memory layout None");
  }
}

// This method will be deprecated in favor of method below
//
::tt::tt_metal::BufferType
toTTNNBufferType(::tt::target::MemorySpace memorySpace) {
  switch (memorySpace) {
  case ::tt::target::MemorySpace::System:
  case ::tt::target::MemorySpace::SystemMMIO:
    return ::tt::tt_metal::BufferType::SYSTEM_MEMORY;
  case ::tt::target::MemorySpace::DeviceDRAM:
    return ::tt::tt_metal::BufferType::DRAM;
  case ::tt::target::MemorySpace::DeviceL1:
    return ::tt::tt_metal::BufferType::L1;
  }
}

// Prefer to use this method
//
::ttnn::BufferType toTTNNBufferType(::tt::target::BufferType bufferType) {

  switch (bufferType) {
  case ::tt::target::BufferType::DRAM:
    return ::ttnn::BufferType::DRAM;
  case ::tt::target::BufferType::L1:
    return ::ttnn::BufferType::L1;
  case ::tt::target::BufferType::SystemMemory:
    return ::ttnn::BufferType::SYSTEM_MEMORY;
  case ::tt::target::BufferType::L1Small:
    return ::ttnn::BufferType::L1_SMALL;
  case ::tt::target::BufferType::Trace:
    return ::ttnn::BufferType::TRACE;
  }
};

::ttnn::Layout
inferLayoutFromTileShape(const ::tt::target::TensorRef *tensorRef) {
  const ::tt::target::Dim2d *tileShape =
      tensorRef->desc()->layout()->memory_desc()->tile_shape();
  LOG_ASSERT(isValidTileShape(tileShape));
  if (tileShape->x() == 1 and tileShape->y() == 1) {
    return ::ttnn::Layout::ROW_MAJOR;
  }
  return ::ttnn::Layout::TILE;
}

CoreRangeSet
toCoreRangeSet(const ::flatbuffers::Vector<const ::tt::target::Dim2dRange *>
                   *coreRangeSet) {
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
  const ::tt::target::Dim2d *tileShape = layout->memory_desc()->tile_shape();

  LOG_ASSERT(targetCoreRangeSet->size() == 1,
             "Currently only single core range/grid is supported");

  LOG_ASSERT(targetShardShape->size() == 2,
             "Only 2D shard shape is supported in TTNN backend");

  LOG_ASSERT(::tt::runtime::ttnn::utils::isValidTileShape(tileShape),
             "Invalid tile shape");

  CoreRangeSet ttnnCoreRangeSet = toCoreRangeSet(targetCoreRangeSet);
  std::array<uint32_t, 2> ttnnShardShape;
  std::copy(targetShardShape->begin(), targetShardShape->end(),
            ttnnShardShape.begin());

  ttnnShardShape[0] *= tileShape->y();
  ttnnShardShape[1] *= tileShape->x();

  ::tt::tt_metal::TensorMemoryLayout ttnnMemLayout =
      toTTNNTensorMemoryLayout(targetMemoryLayout);

  ::tt::tt_metal::BufferType ttnnBufferType =
      toTTNNBufferType(targetMemorySpace);

  ::tt::tt_metal::ShardSpec shardSpec(
      ttnnCoreRangeSet, ttnnShardShape,
      ::tt::tt_metal::ShardOrientation::ROW_MAJOR);

  std::optional<::tt::tt_metal::ShardSpec> shardSpecOpt =
      ttnnMemLayout == tt_metal::TensorMemoryLayout::INTERLEAVED
          ? std::nullopt
          : std::make_optional(shardSpec);

  ::tt::tt_metal::MemoryConfig memoryConfig{.memory_layout = ttnnMemLayout,
                                            .buffer_type = ttnnBufferType,
                                            .shard_spec = shardSpecOpt};
  return memoryConfig;
}

Tensor createRuntimeTensorFromTTNN(const ::ttnn::Tensor &tensor) {
  auto tensorPtr = std::make_shared<::ttnn::Tensor>(tensor);
  return Tensor(std::static_pointer_cast<void>(tensorPtr), nullptr,
                DeviceRuntime::TTNN);
}

} // namespace tt::runtime::ttnn::utils
