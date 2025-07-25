// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/utils.h"
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/debug_apis.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/types.h"
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
getBinary(::tt::runtime::Flatbuffer binary) {
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
  case ::tt::target::ttnn::StorageType::Host:
    return ::ttnn::StorageType::HOST;
  case ::tt::target::ttnn::StorageType::Device:
    return ::ttnn::StorageType::DEVICE;
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

CoreCoord toTTNNCoreCoord(const ::tt::target::ttnn::CoreCoord &coreCoord) {
  return CoreCoord(coreCoord.x(), coreCoord.y());
}

CoreRange toTTNNCoreRange(const tt::target::ttnn::CoreRange &coreRange) {
  CoreCoord start = toTTNNCoreCoord(coreRange.start_coord());
  CoreCoord end = toTTNNCoreCoord(coreRange.end_coord());
  return CoreRange(start, end);
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

::ttnn::ShardOrientation
toTTNNShardOrientation(tt::target::ttnn::ShardOrientation orientation) {
  switch (orientation) {
  case tt::target::ttnn::ShardOrientation::RowMajor:
    return ::ttnn::ShardOrientation::ROW_MAJOR;
  case tt::target::ttnn::ShardOrientation::ColMajor:
    return ::ttnn::ShardOrientation::COL_MAJOR;
  }
}

::ttnn::ShardMode toTTNNShardMode(tt::target::ttnn::ShardMode mode) {
  switch (mode) {
  case tt::target::ttnn::ShardMode::Physical:
    return ::ttnn::ShardMode::PHYSICAL;
  case tt::target::ttnn::ShardMode::Logical:
    return ::ttnn::ShardMode::LOGICAL;
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
    ::ttnn::ShardMode ttnnShardMode =
        toTTNNShardMode(memcfg->shard_spec()->mode());
    LOG_ASSERT(ttnnShardMode == ::ttnn::ShardMode::PHYSICAL &&
                   memcfg->shard_spec()->physical_shard_shape() == 0,
               "Physical shard shape must be empty");
    metalShardSpec = ::tt::tt_metal::ShardSpec(
        ttnnCoreRangeSet, ttnnShardShape, ttnnShardOrientation, ttnnShardMode);
  }

  ::ttnn::MemoryConfig memoryConfig{ttnnMemLayout, ttnnBufferType,
                                    metalShardSpec};
  return std::make_optional(memoryConfig);
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

::ttnn::Tensor &getTTNNTensorFromRuntimeTensor(::tt::runtime::Tensor tensor) {
  return tensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN)
      .getTensor();
}

::tt::runtime::TensorRef
createRuntimeTensorRefFromTTNN(const ::tt::target::ttnn::TensorRef *tensorRef) {
  std::shared_ptr<const void> tensorRefPtr =
      ::tt::runtime::utils::unsafe_borrow_shared(tensorRef);
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
