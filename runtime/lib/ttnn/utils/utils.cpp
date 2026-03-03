// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/common/runtime_context.h"
#include "tt/runtime/detail/ttnn/debug_apis.h"
#include "tt/runtime/detail/ttnn/types/program_desc_cache.h"
#include "tt/runtime/detail/ttnn/types/trace_cache.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"
#include "tt/runtime/workarounds.h"

namespace tt::runtime::ttnn::utils {

using ::tt::runtime::DeviceRuntime;

static tt::runtime::MemoryView
createMemoryView(const tt::tt_metal::detail::MemoryView &memoryView) {
  return tt::runtime::MemoryView{
      .numBanks = memoryView.num_banks,
      .totalBytesPerBank = memoryView.total_bytes_per_bank,
      .totalBytesAllocatedPerBank = memoryView.total_bytes_allocated_per_bank,
      .totalBytesFreePerBank = memoryView.total_bytes_free_per_bank,
      .largestContiguousBytesFreePerBank =
          memoryView.largest_contiguous_bytes_free_per_bank,
      .blockTable = memoryView.block_table,
  };
}

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

// tt-metal tilize supports: bfloat16, float32, uint32, int32, uint16
// See: ttnn/operations/data_movement/tilize/device/tilize_op.cpp
bool canTilizeDataTypeOnDevice(const ::ttnn::DataType &dataType) {
  return dataType == ::ttnn::DataType::BFLOAT16 ||
         dataType == ::ttnn::DataType::FLOAT32 ||
         dataType == ::ttnn::DataType::UINT32 ||
         dataType == ::ttnn::DataType::UINT16 ||
         dataType == ::ttnn::DataType::INT32;
}

// tt-metal tilize on device requires INTERLEAVED or HEIGHT_SHARDED memory.
// See: https://github.com/tenstorrent/tt-mlir/issues/6247
bool canTilizeMemoryLayoutOnDevice(
    const std::optional<::ttnn::MemoryConfig> &memoryConfig) {
  if (!memoryConfig.has_value()) {
    return true; // Default memory config is INTERLEAVED
  }
  const auto &memLayout = memoryConfig->memory_layout();
  return memLayout == ::ttnn::TensorMemoryLayout::INTERLEAVED ||
         memLayout == ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED;
}

bool canTilizeOnDevice(
    const ::ttnn::DataType &dataType,
    const std::optional<::ttnn::MemoryConfig> &memoryConfig) {
  return canTilizeDataTypeOnDevice(dataType) &&
         canTilizeMemoryLayoutOnDevice(memoryConfig);
}

// tt-metal untilize supports: bfloat16, float32, uint32, int32
// (requires use_pack_untilize for uint32/int32)
// See: ttnn/operations/data_movement/untilize/device/untilize_op.cpp
// FP32 untilize fix: https://github.com/tenstorrent/tt-metal/pull/33904
// UINT32 large tensor untilize issue:
// https://github.com/tenstorrent/tt-metal/issues/34072
bool canUntilizeDataTypeOnDevice(const ::ttnn::DataType &dataType) {
  return dataType == ::ttnn::DataType::BFLOAT16 ||
         dataType == ::ttnn::DataType::FLOAT32 ||
         dataType == ::ttnn::DataType::UINT32 ||
         dataType == ::ttnn::DataType::INT32;
}

// tt-metal untilize does not support ND sharding. See:
// https://github.com/tenstorrent/tt-metal/issues/35418
bool canUntilizeOnDevice(
    const ::ttnn::DataType &dataType,
    const std::optional<::ttnn::MemoryConfig> &memoryConfig) {
  bool notSharded = !memoryConfig.has_value() || !memoryConfig->is_sharded();
  bool legacySharded =
      memoryConfig.has_value() && memoryConfig->shard_spec().has_value();
  return canUntilizeDataTypeOnDevice(dataType) && (notSharded || legacySharded);
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

tt::tt_metal::CoreCoord
toTTNNCoreCoord(const ::tt::target::ttnn::CoreCoord &coreCoord) {
  return tt::tt_metal::CoreCoord(coreCoord.x(), coreCoord.y());
}

::tt::target::ttnn::CoreCoord
fromTTNNCoreCoord(const tt::tt_metal::CoreCoord &coreCoord) {
  return ::tt::target::ttnn::CoreCoord(coreCoord.x, coreCoord.y);
}

tt::tt_metal::CoreRange
toTTNNCoreRange(const tt::target::ttnn::CoreRange &coreRange) {
  tt::tt_metal::CoreCoord start = toTTNNCoreCoord(coreRange.start_coord());
  tt::tt_metal::CoreCoord end = toTTNNCoreCoord(coreRange.end_coord());
  return tt::tt_metal::CoreRange(start, end);
}

tt::tt_metal::distributed::MeshCoordinate
toTTNNMeshCoordinate(const tt::target::ttnn::MeshCoord &meshCoord) {
  return tt::tt_metal::distributed::MeshCoordinate(meshCoord.coords()->Get(0),
                                                   meshCoord.coords()->Get(1));
}

tt::tt_metal::distributed::MeshCoordinateRange toTTNNMeshCoordinateRange(
    const tt::target::ttnn::MeshCoordRange &meshCoordRange) {
  tt::tt_metal::distributed::MeshCoordinate start =
      toTTNNMeshCoordinate(*meshCoordRange.start());
  tt::tt_metal::distributed::MeshCoordinate end =
      toTTNNMeshCoordinate(*meshCoordRange.end());
  return tt::tt_metal::distributed::MeshCoordinateRange(start, end);
}

::tt::target::ttnn::CoreRange
fromTTNNCoreRange(const tt::tt_metal::CoreRange &coreRange) {
  return tt::target::ttnn::CoreRange(fromTTNNCoreCoord(coreRange.start_coord),
                                     fromTTNNCoreCoord(coreRange.end_coord));
}

tt::tt_metal::CoreRangeSet
toTTNNCoreRangeSet(const tt::target::ttnn::CoreRangeSet &coreRangeSet) {
  std::set<tt::tt_metal::CoreRange> coreRanges;
  for (const tt::target::ttnn::CoreRange *coreRange :
       *coreRangeSet.core_ranges()) {
    coreRanges.emplace(toTTNNCoreRange(*coreRange));
  }
  return tt::tt_metal::CoreRangeSet(coreRanges);
}

::flatbuffers::Offset<::tt::target::ttnn::CoreRangeSet>
fromTTNNCoreRangeSet(flatbuffers::FlatBufferBuilder &fbb,
                     const tt::tt_metal::CoreRangeSet &coreRangeSet) {
  std::vector<tt::target::ttnn::CoreRange> coreRanges;
  for (const tt::tt_metal::CoreRange &coreRange : coreRangeSet.ranges()) {
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

tt::tt_metal::ShardDistributionStrategy toTTNNShardDistributionStrategy(
    tt::target::ttnn::ShardDistributionStrategy distributionStrategy) {
  switch (distributionStrategy) {
  case tt::target::ttnn::ShardDistributionStrategy::RoundRobin1D:
    return tt::tt_metal::ShardDistributionStrategy::ROUND_ROBIN_1D;
  case tt::target::ttnn::ShardDistributionStrategy::Grid2D:
    return tt::tt_metal::ShardDistributionStrategy::GRID_2D;
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

  const auto targetBufferType = memcfg->buffer_type();
  LOG_ASSERT(targetBufferType == ::tt::target::BufferType::DRAM ||
                 targetBufferType == ::tt::target::BufferType::L1,
             "Memory config buffer type should be DRAM or L1");
  const auto ttnnBufferType = toTTNNBufferType(targetBufferType);

  const auto targetMemLayout = memcfg->tensor_memory_layout();
  const auto memLayout = toTTNNTensorMemoryLayout(targetMemLayout);

  // Verify that shard spec is present only for sharded memory layouts
  const bool hasShardSpec =
      (memcfg->shard_spec() != nullptr) || (memcfg->nd_shard_spec() != nullptr);
  LOG_ASSERT(
      hasShardSpec == isSharded(targetMemLayout),
      "A shard spec must be present if and only if the tensor is sharded");

  // Handle (legacy) shard spec
  if (const auto *shardSpec = memcfg->shard_spec()) {
    const auto *shardShape = shardSpec->shape();
    LOG_ASSERT(shardShape->size() == 2,
               "Only 2D shard shape is supported in TTNN backend");
    std::array<uint32_t, 2> shape;
    std::copy(shardShape->begin(), shardShape->end(), shape.begin());

    const tt::tt_metal::CoreRangeSet coreRangeSet =
        toTTNNCoreRangeSet(*shardSpec->core_range_set());
    const ::ttnn::ShardOrientation orientation =
        toTTNNShardOrientation(shardSpec->orientation());
    auto metalShardSpec =
        ::tt::tt_metal::ShardSpec(coreRangeSet, shape, orientation);

    return ::ttnn::MemoryConfig{memLayout, ttnnBufferType, metalShardSpec};
  }

  // Handle ND shard spec
  if (const auto *ndShardSpec = memcfg->nd_shard_spec()) {
    const auto *shardShape = ndShardSpec->shape();
    std::vector<uint32_t> shape(shardShape->begin(), shardShape->end());

    const tt::tt_metal::CoreRangeSet coreRangeSet =
        toTTNNCoreRangeSet(*ndShardSpec->core_range_set());
    const ::ttnn::ShardOrientation orientation =
        toTTNNShardOrientation(ndShardSpec->orientation());
    const tt::tt_metal::ShardDistributionStrategy strategy =
        toTTNNShardDistributionStrategy(ndShardSpec->distribution_strategy());
    auto metalNdShardSpec = tt::tt_metal::NdShardSpec(
        tt::tt_metal::Shape(ttsl::Span<const uint32_t>(shape)), coreRangeSet,
        orientation, strategy);

    return ::ttnn::MemoryConfig{ttnnBufferType, metalNdShardSpec};
  }

  // Non-sharded memory config
  return ::ttnn::MemoryConfig{memLayout, ttnnBufferType};
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

void *getRawHostDataPtr(const ::ttnn::Tensor &tensor) {
  ::tt::tt_metal::HostBuffer hostBuffer =
      ::tt::tt_metal::host_buffer::get_host_buffer(tensor);
  return static_cast<void *>(hostBuffer.view_bytes().data());
}

::ttnn::TensorSpec createTensorSpec(const ::ttnn::Shape &shape,
                                    const ::ttnn::DataType &dataType,
                                    const ::ttnn::Layout &layout,
                                    const ::ttnn::MemoryConfig &memoryConfig) {
  ::ttnn::TensorSpec tensorSpec(
      shape, tt::tt_metal::TensorLayout(dataType, layout, memoryConfig));
  return tensorSpec;
}

std::unordered_map<tt::runtime::MemoryBufferType, tt::runtime::MemoryView>
getMemoryView(Device deviceHandle) {
  std::unordered_map<tt::runtime::MemoryBufferType, tt::runtime::MemoryView>
      memoryMap;
  ::ttnn::MeshDevice &meshDevice =
      deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);

  auto dramMemoryView = ::tt::tt_metal::detail::GetMemoryView(
      &meshDevice, ::ttnn::BufferType::DRAM);
  auto l1MemoryView = ::tt::tt_metal::detail::GetMemoryView(
      &meshDevice, ::ttnn::BufferType::L1);
  auto l1SmallMemoryView = ::tt::tt_metal::detail::GetMemoryView(
      &meshDevice, ::ttnn::BufferType::L1_SMALL);
  auto traceMemoryView = ::tt::tt_metal::detail::GetMemoryView(
      &meshDevice, ::ttnn::BufferType::TRACE);

  memoryMap[tt::runtime::MemoryBufferType::DRAM] =
      createMemoryView(dramMemoryView);
  memoryMap[tt::runtime::MemoryBufferType::L1] = createMemoryView(l1MemoryView);
  memoryMap[tt::runtime::MemoryBufferType::L1_SMALL] =
      createMemoryView(l1SmallMemoryView);
  memoryMap[tt::runtime::MemoryBufferType::TRACE] =
      createMemoryView(traceMemoryView);

  return memoryMap;
}

} // namespace tt::runtime::ttnn::utils
