// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "utils.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::utils {

// TODO (bug #701)
// Currently the memory layout/location in flatbuffer is incorrect
// These methods are workarounds such that we query the info directly from the
// TTNN tensor Ideally, we should be able to get all of this info directly from
// the flatbuffer
bool isOnHost(const ::ttnn::Tensor &tensor) {
  // Currently only supports borrowed or owned host storage
  return tensor.storage_type() == ::tt::tt_metal::StorageType::BORROWED or
         tensor.storage_type() == ::tt::tt_metal::StorageType::OWNED or
         tensor.storage_type() ==
             ::tt::tt_metal::StorageType::MULTI_DEVICE_HOST;
}

bool isOnDevice(const ::ttnn::Tensor &tensor) {
  // Currently only supports single device storage
  return tensor.storage_type() == ::tt::tt_metal::StorageType::DEVICE or
         tensor.storage_type() == ::tt::tt_metal::StorageType::MULTI_DEVICE;
}

bool isTilized(const ::tt::target::TensorRef *tensorRef) {
  const ::tt::target::Dim2d *tileShape =
      tensorRef->desc()->layout()->memory_desc()->tile_shape();
  return tileShape->x() == 32 and tileShape->y() == 32;
}

::tt::target::MemorySpace
getMemorySpace(const ::tt::target::TensorRef *tensorRef) {
  return tensorRef->desc()->layout()->memory_desc()->memory_space();
}

bool inSystemMemory(const ::tt::target::TensorRef *tensorRef) {
  const ::tt::target::MemorySpace targetMemorySpace = getMemorySpace(tensorRef);
  return targetMemorySpace == ::tt::target::MemorySpace::System or
         targetMemorySpace == ::tt::target::MemorySpace::SystemMMIO;
}

void updateTensorPool(ProgramTensorPool &tensorPool,
                      const ::ttnn::Tensor &tensor, uint32_t outputGlobalId) {

  LOG_INFO("tensor storage: ", static_cast<uint32_t>(tensor.storage_type()));
  LOG_INFO(
      "tensor distributed config is replicate: ",
      std::holds_alternative<::tt::tt_metal::ReplicateTensor>(
          ::ttnn::distributed::api::get_distributed_tensor_config_from_tensor(
              tensor)));
  if (tensorPool.isUserOutput(outputGlobalId)) {
    return;
  } else {
    tensorPool.insert_or_assign(outputGlobalId, tensor);
  }
}

::ttnn::DataType getDataType(const ::tt::target::TensorRef *tensorRef) {
  return ::tt::runtime::ttnn::utils::toTTNNDataType(
      tensorRef->desc()->layout()->memory_desc()->data_type());
}

::ttnn::Layout
inferLayoutFromTileShape(const ::tt::target::TensorRef *tensorRef) {
  const ::tt::target::Dim2d *tileShape =
      tensorRef->desc()->layout()->memory_desc()->tile_shape();
  LOG_ASSERT(::tt::runtime::ttnn::utils::isValidTileShape(tileShape));
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

// This method will soon be deprecated, prefer to use the method below
//
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

  ::tt::tt_metal::ShardSpec shardSpec(
      ttnnCoreRangeSet, ttnnShardShape,
      ::tt::tt_metal::ShardOrientation::ROW_MAJOR, false);

  ::tt::tt_metal::TensorMemoryLayout ttnnMemLayout =
      ::tt::runtime::ttnn::utils::toTTNNTensorMemoryLayout(targetMemoryLayout);

  ::tt::tt_metal::BufferType ttnnBufferType =
      ::tt::runtime::ttnn::utils::toTTNNBufferType(targetMemorySpace);

  return {ttnnMemLayout, ttnnBufferType, shardSpec};
}

// Prefer to use this method over the one above
//
::tt::tt_metal::MemoryConfig
createMemoryConfig(const ::tt::target::MemoryConfigDesc *memcfg,
                   const ::tt::target::TensorRef *tensorRef) {

  ::ttnn::TensorMemoryLayout tensorMemoryLayout =
      ::tt::runtime::ttnn::utils::toTTNNTensorMemoryLayout(
          memcfg->tensor_memory_layout());

  ::ttnn::BufferType bufferType =
      ::tt::runtime::ttnn::utils::toTTNNBufferType(memcfg->buffer_type());

  const ::tt::target::LayoutDesc *layout = tensorRef->desc()->layout();
  const ::flatbuffers::Vector<const tt::target::Dim2dRange *>
      *targetCoreRangeSet = layout->core_range_set();
  CoreRangeSet ttnnCoreRangeSet = toCoreRangeSet(targetCoreRangeSet);
  const ::flatbuffers::Vector<int64_t> *shardShape =
      memcfg->shard_spec()->shard_shape();
  const ::tt::target::Dim2d *tileShape = layout->memory_desc()->tile_shape();

  LOG_ASSERT(targetCoreRangeSet->size() == 1,
             "Currently only single core range/grid is supported");

  LOG_ASSERT(shardShape->size() == 2,
             "Only 2D shard shape is supported in TTNN backend");

  LOG_ASSERT(::tt::runtime::ttnn::utils::isValidTileShape(tileShape),
             "Invalid tile shape");

  std::array<uint32_t, 2> ttnnShardShape = {
      static_cast<uint32_t>(shardShape->Get(0)),
      static_cast<uint32_t>(shardShape->Get(1))};

  ttnnShardShape[0] *= tileShape->y();
  ttnnShardShape[1] *= tileShape->x();

  ::tt::tt_metal::ShardSpec shardSpec(
      ttnnCoreRangeSet, ttnnShardShape,
      ::tt::tt_metal::ShardOrientation::ROW_MAJOR, false);

  ::ttnn::MemoryConfig memoryConfig = {tensorMemoryLayout, bufferType,
                                       shardSpec};
  return memoryConfig;
}

::tt::tt_metal::DistributedTensorConfig distributedTensorConfigFromFlatbuffer(
    const ::tt::target::DistributionStrategy *strategy) {
  switch (strategy->strategy_type()) {
  case ::tt::target::DistributedTensorConfig::ReplicateTensor: {
    return ::tt::tt_metal::ReplicateTensor(
        strategy->strategy_as_ReplicateTensor()->replication_factor());
  }
  case ::tt::target::DistributedTensorConfig::ShardTensor: {
    return ::tt::tt_metal::ShardTensor(
        strategy->strategy_as_ShardTensor()->shard_dim());
  }
  case ::tt::target::DistributedTensorConfig::ShardTensor2D: {
    uint32_t y = strategy->strategy_as_ShardTensor2D()->shard_mesh()->y();
    uint32_t x = strategy->strategy_as_ShardTensor2D()->shard_mesh()->x();
    ::tt::tt_metal::ShardMesh mesh(y, x);
    return ::tt::tt_metal::ShardTensor2D(mesh);
  }
  case ::tt::target::DistributedTensorConfig::AllGatherTensor: {
    return ::tt::tt_metal::AllGatherTensor();
  }
  case ::tt::target::DistributedTensorConfig::NONE: {
    throw std::invalid_argument("Unsupported distributed tensor config");
  }
  }
}
} // namespace tt::runtime::ttnn::operations::utils
