// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/debug_apis.h"
#include "tt/runtime/detail/ttnn/types/trace_cache.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"
#include "tt/runtime/workarounds.h"

namespace tt::runtime::ttnn::utils {

using ::tt::runtime::DeviceRuntime;

// TODO (bug #701)
// Currently the memory layout/location in flatbuffer is incorrect
// These methods are workarounds for operations such that we query the info
// directly from the TTNN tensor. Ideally, we should be able to get all of this
// info directly from the flatbuffer using the "inSystemMemory" API below
bool isOnHost(const ::ttnn::StorageType &storageType) {
  return storageType == ::ttnn::StorageType::HOST;
}

bool isOnDevice(const ::ttnn::StorageType &storageType) {
  return storageType == ::ttnn::StorageType::DEVICE;
}

bool inSystemMemory(const ::tt::target::ttnn::TensorRef *tensorRef) {
  const ::tt::target::ttnn::StorageType storageType =
      tensorRef->desc()->layout()->memory_desc()->storage_type();
  return storageType == ::tt::target::ttnn::StorageType::Host;
}

bool inDeviceMemory(const ::tt::target::ttnn::TensorRef *tensorRef) {
  const ::tt::target::ttnn::StorageType storageType =
      tensorRef->desc()->layout()->memory_desc()->storage_type();
  return (storageType == ::tt::target::ttnn::StorageType::Device);
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

// TODO (jnie): Add support for fp32, currently there's some precision loss
// which causes some FE tests to fail.
// Tracking here: https://github.com/tenstorrent/tt-metal/issues/21023
bool canTilizeDataTypeOnDevice(const ::ttnn::DataType &dataType) {
  return dataType == ::ttnn::DataType::BFLOAT16;
}

bool canUntilizeDataTypeOnDevice(const ::ttnn::DataType &dataType) {
  return dataType == ::ttnn::DataType::BFLOAT16;
}

const ::tt::target::ttnn::TTNNBinary *
getBinary(const ::tt::runtime::Flatbuffer &binary) {
  bool isTTNN = ::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
      binary.handle.get());
  LOG_ASSERT(isTTNN, "Unsupported binary format");
  return ::tt::target::ttnn::GetSizePrefixedTTNNBinary(binary.handle.get());
}

const ::tt::target::ttnn::Program *
getProgram(const ::tt::runtime::Binary &executableHandle,
           std::uint32_t programIndex) {
  const ::tt::target::ttnn::TTNNBinary &fbb = *getBinary(executableHandle);
  const ::tt::target::ttnn::Program *program =
      fbb.programs()->Get(programIndex);
  return program;
}

::ttnn::operations::reduction::ReduceType getReduceType(uint32_t reduceType) {
  switch (reduceType) {
  case 0:
    return ::ttnn::operations::reduction::ReduceType::Sum;
  case 1:
    return ::ttnn::operations::reduction::ReduceType::Mean;
  case 2:
    return ::ttnn::operations::reduction::ReduceType::Max;
  case 3:
    return ::ttnn::operations::reduction::ReduceType::Min;
  case 4:
    return ::ttnn::operations::reduction::ReduceType::Std;
  case 5:
    return ::ttnn::operations::reduction::ReduceType::Var;
  default:
    LOG_FATAL("Unsupported reduce type");
  }
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
  case ::tt::target::DataType::Int32:
    return ::ttnn::DataType::INT32;

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
  case ::ttnn::DataType::INT32:
    return ::tt::target::DataType::Int32;

  default:
    LOG_FATAL("Unsupported data type");
  }
}

MathFidelity toTTNNMathFidelity(::tt::target::MathFidelity mathFidelity) {
  switch (mathFidelity) {
  case ::tt::target::MathFidelity::LoFi:
    return MathFidelity::LoFi;
  case ::tt::target::MathFidelity::HiFi2:
    return MathFidelity::HiFi2;
  case ::tt::target::MathFidelity::HiFi3:
    return MathFidelity::HiFi3;
  case ::tt::target::MathFidelity::HiFi4:
    return MathFidelity::HiFi4;
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

::tt::target::TensorLayout fromTTNNLayout(::ttnn::Layout layout) {
  switch (layout) {
  case ::ttnn::Layout::TILE:
    return ::tt::target::TensorLayout::Tile;
  case ::ttnn::Layout::ROW_MAJOR:
    return ::tt::target::TensorLayout::RowMajor;
  default:
    LOG_FATAL("Unsupported layout");
  }
}

::ttnn::TensorMemoryLayout toTTNNTensorMemoryLayout(
    ::tt::target::ttnn::TensorMemoryLayout tensorMemoryLayout) {

  switch (tensorMemoryLayout) {
  case ::tt::target::ttnn::TensorMemoryLayout::Interleaved:
    return ::ttnn::TensorMemoryLayout::INTERLEAVED;
  case ::tt::target::ttnn::TensorMemoryLayout::HeightSharded:
    return ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED;
  case ::tt::target::ttnn::TensorMemoryLayout::WidthSharded:
    return ::ttnn::TensorMemoryLayout::WIDTH_SHARDED;
  case ::tt::target::ttnn::TensorMemoryLayout::BlockSharded:
    return ::ttnn::TensorMemoryLayout::BLOCK_SHARDED;
  }
}

::tt::target::ttnn::TensorMemoryLayout
fromTTNNTensorMemoryLayout(::ttnn::TensorMemoryLayout tensorMemoryLayout) {
  switch (tensorMemoryLayout) {
  case ::ttnn::TensorMemoryLayout::INTERLEAVED:
    return ::tt::target::ttnn::TensorMemoryLayout::Interleaved;
  case ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED:
    return ::tt::target::ttnn::TensorMemoryLayout::HeightSharded;
  case ::ttnn::TensorMemoryLayout::WIDTH_SHARDED:
    return ::tt::target::ttnn::TensorMemoryLayout::WidthSharded;
  case ::ttnn::TensorMemoryLayout::BLOCK_SHARDED:
    return ::tt::target::ttnn::TensorMemoryLayout::BlockSharded;
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

::tt::target::BufferType fromTTNNBufferType(::ttnn::BufferType bufferType) {

  switch (bufferType) {
  case ::ttnn::BufferType::DRAM:
    return ::tt::target::BufferType::DRAM;
  case ::ttnn::BufferType::L1:
    return ::tt::target::BufferType::L1;
  case ::ttnn::BufferType::SYSTEM_MEMORY:
    return ::tt::target::BufferType::SystemMemory;
  case ::ttnn::BufferType::L1_SMALL:
    return ::tt::target::BufferType::L1Small;
  case ::ttnn::BufferType::TRACE:
    return ::tt::target::BufferType::Trace;
  }
};

::ttnn::StorageType
toTTNNStorageType(::tt::target::ttnn::StorageType storageType) {
  switch (storageType) {
  case ::tt::target::ttnn::StorageType::Host:
    return ::ttnn::StorageType::HOST;
  case ::tt::target::ttnn::StorageType::Device:
    return ::ttnn::StorageType::DEVICE;
  }
}

::tt::target::ttnn::StorageType
fromTTNNStorageType(::ttnn::StorageType storageType) {
  switch (storageType) {
  case ::ttnn::StorageType::HOST:
    return ::tt::target::ttnn::StorageType::Host;
  case ::ttnn::StorageType::DEVICE:
    return ::tt::target::ttnn::StorageType::Device;
  }
}

::ttnn::Layout inferLayoutFromTileShape(const ::tt::target::Dim2d *tileShape) {
  LOG_ASSERT(isValidTileShape(tileShape));
  if (tileShape->x() == 1 && tileShape->y() == 1) {
    return ::ttnn::Layout::ROW_MAJOR;
  }
  return ::ttnn::Layout::TILE;
}

::ttnn::Layout
inferLayoutFromTileShape(const ::tt::target::ttnn::TensorRef *tensorRef) {
  const ::tt::target::Dim2d *tileShape =
      tensorRef->desc()->layout()->memory_desc()->tile_shape();
  return inferLayoutFromTileShape(tileShape);
}

CoreCoord toTTNNCoreCoord(const ::tt::target::ttnn::CoreCoord &coreCoord) {
  return CoreCoord(coreCoord.x(), coreCoord.y());
}

::tt::target::ttnn::CoreCoord fromTTNNCoreCoord(const CoreCoord &coreCoord) {
  return ::tt::target::ttnn::CoreCoord(coreCoord.x, coreCoord.y);
}

CoreRange toTTNNCoreRange(const tt::target::ttnn::CoreRange &coreRange) {
  CoreCoord start = toTTNNCoreCoord(coreRange.start_coord());
  CoreCoord end = toTTNNCoreCoord(coreRange.end_coord());
  return CoreRange(start, end);
}

::tt::target::ttnn::CoreRange fromTTNNCoreRange(const CoreRange &coreRange) {
  return tt::target::ttnn::CoreRange(fromTTNNCoreCoord(coreRange.start_coord),
                                     fromTTNNCoreCoord(coreRange.end_coord));
}

CoreRangeSet
toTTNNCoreRangeSet(const tt::target::ttnn::CoreRangeSet &coreRangeSet) {
  std::set<CoreRange> coreRanges;
  for (const tt::target::ttnn::CoreRange *coreRange :
       *coreRangeSet.core_ranges()) {
    coreRanges.emplace(toTTNNCoreRange(*coreRange));
  }
  return CoreRangeSet(coreRanges);
}

::flatbuffers::Offset<::tt::target::ttnn::CoreRangeSet>
fromTTNNCoreRangeSet(flatbuffers::FlatBufferBuilder &fbb,
                     const CoreRangeSet &coreRangeSet) {
  std::vector<tt::target::ttnn::CoreRange> coreRanges;
  for (const CoreRange &coreRange : coreRangeSet.ranges()) {
    coreRanges.emplace_back(fromTTNNCoreRange(coreRange));
  }
  return tt::target::ttnn::CreateCoreRangeSetDirect(fbb, &coreRanges);
}

::ttnn::ShardOrientation
toTTNNShardOrientation(tt::target::ttnn::ShardOrientation orientation) {
  switch (orientation) {
  case tt::target::ttnn::ShardOrientation::RowMajor:
    return ::ttnn::ShardOrientation::ROW_MAJOR;
  case tt::target::ttnn::ShardOrientation::ColMajor:
    return ::ttnn::ShardOrientation::COL_MAJOR;
  }
}

::tt::target::ttnn::ShardOrientation
fromTTNNShardOrientation(::ttnn::ShardOrientation orientation) {
  switch (orientation) {
  case ::ttnn::ShardOrientation::ROW_MAJOR:
    return tt::target::ttnn::ShardOrientation::RowMajor;
  case ::ttnn::ShardOrientation::COL_MAJOR:
    return tt::target::ttnn::ShardOrientation::ColMajor;
  }
}

::flatbuffers::Offset<::tt::target::ttnn::ShardSpec>
fromTTNNShardSpec(::flatbuffers::FlatBufferBuilder &fbb,
                  const ::tt::tt_metal::ShardSpec &ttnnShardSpec) {
  auto coreRangeSet =
      ::tt::runtime::ttnn::utils::fromTTNNCoreRangeSet(fbb, ttnnShardSpec.grid);
  std::vector<int32_t> shape(ttnnShardSpec.shape.begin(),
                             ttnnShardSpec.shape.end());
  ::tt::target::ttnn::ShardOrientation orientation =
      ::tt::runtime::ttnn::utils::fromTTNNShardOrientation(
          ttnnShardSpec.orientation);

  return ::tt::target::ttnn::CreateShardSpecDirect(fbb, coreRangeSet, &shape,
                                                   orientation);
}

::tt::CoreType toCoreType(const ::tt::target::ttnn::CoreType &coreType) {
  switch (coreType) {
  case ::tt::target::ttnn::CoreType::WORKER: {
    return ::tt::CoreType::WORKER;
  }
  case ::tt::target::ttnn::CoreType::ETH: {
    return ::tt::CoreType::ETH;
  }
  }
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

  // Verify that shard spec is present only for sharded memory layouts
  LOG_ASSERT((memcfg->shard_spec() != nullptr) ==
             isSharded(targetMemoryLayout));
  std::optional<::tt::tt_metal::ShardSpec> metalShardSpec = std::nullopt;

  if (isSharded(targetMemoryLayout)) {
    const ::flatbuffers::Vector<int32_t> *targetShardShape =
        memcfg->shard_spec()->shape();
    LOG_ASSERT(targetShardShape->size() == 2,
               "Only 2D shard shape is supported in TTNN backend");
    std::array<uint32_t, 2> ttnnShardShape;
    std::copy(targetShardShape->begin(), targetShardShape->end(),
              ttnnShardShape.begin());

    const tt::target::ttnn::CoreRangeSet *targetCoreRangeSet =
        memcfg->shard_spec()->core_range_set();
    CoreRangeSet ttnnCoreRangeSet = toTTNNCoreRangeSet(*targetCoreRangeSet);
    ::ttnn::ShardOrientation ttnnShardOrientation =
        toTTNNShardOrientation(memcfg->shard_spec()->orientation());
    metalShardSpec = ::tt::tt_metal::ShardSpec(ttnnCoreRangeSet, ttnnShardShape,
                                               ttnnShardOrientation);
  }

  ::ttnn::MemoryConfig memoryConfig{ttnnMemLayout, ttnnBufferType,
                                    metalShardSpec};
  return std::make_optional(memoryConfig);
}

::flatbuffers::Offset<::tt::target::ttnn::MemoryConfig>
fromTTNNMemoryConfig(::flatbuffers::FlatBufferBuilder &fbb,
                     const ::ttnn::MemoryConfig &ttnnMemoryConfig) {

  ::tt::target::ttnn::TensorMemoryLayout tensorMemoryLayout =
      ::tt::runtime::ttnn::utils::fromTTNNTensorMemoryLayout(
          ttnnMemoryConfig.memory_layout());

  ::tt::target::BufferType bufferType =
      ::tt::runtime::ttnn::utils::fromTTNNBufferType(
          ttnnMemoryConfig.buffer_type());

  const std::optional<::tt::tt_metal::ShardSpec> &shardSpec =
      ttnnMemoryConfig.shard_spec();
  if (!shardSpec.has_value()) {
    return ::tt::target::ttnn::CreateMemoryConfig(fbb, tensorMemoryLayout,
                                                  bufferType, /*shard_spec=*/0);
  }

  auto fbShardSpec = fromTTNNShardSpec(fbb, shardSpec.value());

  return ::tt::target::ttnn::CreateMemoryConfig(fbb, tensorMemoryLayout,
                                                bufferType, fbShardSpec);
}

::tt::runtime::Tensor
createRuntimeTensorFromTTNN(const ::ttnn::Tensor &tensor,
                            const std::optional<::ttnn::MeshEvent> &meshEvent,
                            bool retain) {
  auto tensorPtr = std::make_shared<::tt::runtime::ttnn::TTNNTensorWrapper>(
      tensor, meshEvent, retain);

  return ::tt::runtime::Tensor(std::static_pointer_cast<void>(tensorPtr),
                               /*data=*/nullptr, DeviceRuntime::TTNN);
}

::tt::runtime::Device
createRuntimeDeviceFromTTNN(::ttnn::MeshDevice *meshDevice) {
  // Create a non-owning shared_ptr to the provided MeshDevice with no-op
  // deleter.
  std::shared_ptr<void> unsafeMeshDeviceSharedPtr =
      ::tt::runtime::utils::unsafeBorrowShared(meshDevice);
  // Wrap the the device in the runtime device.
  auto ttnnTraceCache = std::make_shared<::tt::runtime::ttnn::TraceCache>(
      std::static_pointer_cast<::ttnn::MeshDevice>(unsafeMeshDeviceSharedPtr));
  auto traceCache = std::make_shared<::tt::runtime::TraceCache>(
      std::static_pointer_cast<void>(ttnnTraceCache), DeviceRuntime::TTNN);

  return Device(unsafeMeshDeviceSharedPtr, traceCache, DeviceRuntime::TTNN);
}

::ttnn::Tensor &getTTNNTensorFromRuntimeTensor(::tt::runtime::Tensor tensor) {
  return tensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN)
      .getTensor();
}

::tt::runtime::TensorRef
createRuntimeTensorRefFromTTNN(const ::tt::target::ttnn::TensorRef *tensorRef) {
  std::shared_ptr<const void> tensorRefPtr =
      ::tt::runtime::utils::unsafeBorrowShared(tensorRef);
  return tt::runtime::TensorRef(tensorRefPtr, DeviceRuntime::TTNN);
}

std::vector<const tt::target::ttnn::TensorRef *> convertFbTensorRefsToVector(
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::ttnn::TensorRef>>
        *fbVector) {
  std::vector<const tt::target::ttnn::TensorRef *> stdVector;
  if (!fbVector) {
    return stdVector;
  }

  stdVector.reserve(fbVector->size());
  for (const auto *tensorRef : *fbVector) {
    stdVector.push_back(tensorRef);
  }

  return stdVector;
}

::ttnn::TensorSpec createTensorSpec(const ::ttnn::Shape &shape,
                                    const ::ttnn::DataType &dataType,
                                    const ::ttnn::Layout &layout,
                                    const ::ttnn::MemoryConfig &memoryConfig) {
  ::ttnn::TensorSpec tensorSpec(
      shape, tt::tt_metal::TensorLayout(dataType, layout, memoryConfig));
  return tensorSpec;
}

void *getRawHostDataPtr(const ::ttnn::Tensor &tensor) {
  ::tt::tt_metal::HostBuffer hostBuffer =
      ::tt::tt_metal::host_buffer::get_host_buffer(tensor);
  return static_cast<void *>(hostBuffer.view_bytes().data());
}

} // namespace tt::runtime::ttnn::utils
