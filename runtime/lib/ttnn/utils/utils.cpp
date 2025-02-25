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
// info directly from the flatbuffer using the "inSystemMemory" API below
bool isOnHost(const ::ttnn::StorageType &storageType) {
  return storageType == ::ttnn::StorageType::BORROWED ||
         storageType == ::ttnn::StorageType::OWNED ||
         storageType == ::ttnn::StorageType::MULTI_DEVICE_HOST;
}

bool inSystemMemory(const ::tt::target::ttnn::TensorRef *tensorRef) {
  const ::tt::target::ttnn::StorageType storageType =
      tensorRef->desc()->layout()->memory_desc()->storage_type();
  return (storageType == ::tt::target::ttnn::StorageType::Borrowed) ||
         (storageType == ::tt::target::ttnn::StorageType::Owned) ||
         (storageType == ::tt::target::ttnn::StorageType::MultiDeviceHost);
}

bool isOnDevice(const ::ttnn::StorageType &storageType) {
  return storageType == ::ttnn::StorageType::DEVICE ||
         storageType == ::ttnn::StorageType::MULTI_DEVICE;
}

bool isValidTileShape(const ::tt::target::Dim2d *shape) {
  return (shape->x() == 1 && shape->y() == 1) ||
         (shape->x() == 32 && shape->y() == 32);
}

bool isSharded(
    const ::tt::target::ttnn::TensorMemoryLayout &tensorMemoryLayout) {
  return tensorMemoryLayout ==
             ::tt::target::ttnn::TensorMemoryLayout::HeightSharded ||
         tensorMemoryLayout ==
             ::tt::target::ttnn::TensorMemoryLayout::WidthSharded ||
         tensorMemoryLayout ==
             ::tt::target::ttnn::TensorMemoryLayout::BlockSharded;
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

::ttnn::TensorMemoryLayout toTTNNTensorMemoryLayout(
    ::tt::target::ttnn::TensorMemoryLayout tensorMemoryLayout) {

  switch (tensorMemoryLayout) {
  case ::tt::target::ttnn::TensorMemoryLayout::Interleaved:
    return ::ttnn::TensorMemoryLayout::INTERLEAVED;
  case ::tt::target::ttnn::TensorMemoryLayout::SingleBank:
    return ::ttnn::TensorMemoryLayout::SINGLE_BANK;
  case ::tt::target::ttnn::TensorMemoryLayout::HeightSharded:
    return ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED;
  case ::tt::target::ttnn::TensorMemoryLayout::WidthSharded:
    return ::ttnn::TensorMemoryLayout::WIDTH_SHARDED;
  case ::tt::target::ttnn::TensorMemoryLayout::BlockSharded:
    return ::ttnn::TensorMemoryLayout::BLOCK_SHARDED;
  }
}

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

::ttnn::StorageType
toTTNNStorageType(::tt::target::ttnn::StorageType storageType) {
  switch (storageType) {
  case ::tt::target::ttnn::StorageType::Owned:
    return ::ttnn::StorageType::OWNED;
  case ::tt::target::ttnn::StorageType::Device:
    return ::ttnn::StorageType::DEVICE;
  case ::tt::target::ttnn::StorageType::Borrowed:
    return ::ttnn::StorageType::BORROWED;
  case ::tt::target::ttnn::StorageType::MultiDevice:
    return ::ttnn::StorageType::MULTI_DEVICE;
  case ::tt::target::ttnn::StorageType::MultiDeviceHost:
    return ::ttnn::StorageType::MULTI_DEVICE_HOST;
  }
}

::ttnn::Layout
inferLayoutFromTileShape(const ::tt::target::ttnn::TensorRef *tensorRef) {
  const ::tt::target::Dim2d *tileShape =
      tensorRef->desc()->layout()->memory_desc()->tile_shape();
  LOG_ASSERT(isValidTileShape(tileShape));
  if (tileShape->x() == 1 && tileShape->y() == 1) {
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

const ::tt::target::ttnn::MemoryConfig *
getTensorRefMemoryConfig(const ::tt::target::ttnn::TensorRef *tensorRef) {
  return tensorRef->desc()->layout()->memory_desc()->memory_config();
}

std::optional<::ttnn::MemoryConfig>
createMemoryConfigIfNeeded(const ::tt::target::ttnn::MemoryConfig *memcfg) {

  if (!memcfg) {
    return std::nullopt;
  }

  const ::tt::target::ttnn::TensorMemoryLayout targetMemoryLayout =
      memcfg->tensor_memory_layout();
  const ::tt::target::BufferType targetBufferType = memcfg->buffer_type();

  LOG_ASSERT(targetBufferType == ::tt::target::BufferType::DRAM ||
                 targetBufferType == ::tt::target::BufferType::L1,
             "Memory config buffer type should be DRAM or L1");

  ::ttnn::TensorMemoryLayout ttnnMemLayout =
      toTTNNTensorMemoryLayout(targetMemoryLayout);

  ::ttnn::BufferType ttnnBufferType = toTTNNBufferType(targetBufferType);

  std::optional<::tt::tt_metal::ShardSpec> metalShardSpec = std::nullopt;

  if (isSharded(targetMemoryLayout)) {
    LOG_ASSERT(memcfg->shard_spec(), "Sharded tensors must have shard spec");
    const ::flatbuffers::Vector<int32_t> *targetShardShape =
        memcfg->shard_spec()->shard_shape();
    LOG_ASSERT(targetShardShape->size() == 2,
               "Only 2D shard shape is supported in TTNN backend");
    std::array<uint32_t, 2> ttnnShardShape;
    std::copy(targetShardShape->begin(), targetShardShape->end(),
              ttnnShardShape.begin());

    const ::flatbuffers::Vector<const tt::target::Dim2dRange *>
        *targetCoreRangeSet = memcfg->shard_spec()->grid();
    LOG_ASSERT(targetCoreRangeSet->size() == 1,
               "Currently only single core range/grid is supported");
    CoreRangeSet ttnnCoreRangeSet = toCoreRangeSet(targetCoreRangeSet);
    metalShardSpec =
        ::tt::tt_metal::ShardSpec(ttnnCoreRangeSet, ttnnShardShape,
                                  ::tt::tt_metal::ShardOrientation::ROW_MAJOR);
  }

  ::ttnn::MemoryConfig memoryConfig{.memory_layout = ttnnMemLayout,
                                    .buffer_type = ttnnBufferType,
                                    .shard_spec = metalShardSpec};
  return std::make_optional(memoryConfig);
}

Tensor createRuntimeTensorFromTTNN(const ::ttnn::Tensor &tensor) {
  auto tensorPtr = std::make_shared<::ttnn::Tensor>(tensor);
  return Tensor(std::static_pointer_cast<void>(tensorPtr), nullptr,
                DeviceRuntime::TTNN);
}

} // namespace tt::runtime::ttnn::utils
