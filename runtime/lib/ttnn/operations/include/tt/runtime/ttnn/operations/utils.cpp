// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "utils.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::utils {

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
  if (tensorPool.isUserOutput(outputGlobalId)) {
    tensorPool.copyTensorToUserOutput(outputGlobalId, tensor);
  } else {
    tensorPool.insert_or_assign(outputGlobalId, tensor);
  }
}

::ttnn::DataType getDataType(const ::tt::target::TensorRef *tensorRef) {
  return ::tt::runtime::ttnn::utils::toTTNNDataType(
      tensorRef->desc()->layout()->memory_desc()->data_type());
}

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
  CoreRangeSet ttnnCoreRangeSet =
      ::tt::runtime::ttnn::utils::toCoreRangeSet(targetCoreRangeSet);
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
    LOG_FATAL("Unsupported distributed tensor config");
  }
  }
}
} // namespace tt::runtime::ttnn::operations::utils
