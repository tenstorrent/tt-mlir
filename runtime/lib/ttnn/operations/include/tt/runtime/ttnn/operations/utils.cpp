// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::utils {

bool isTilized(const ::tt::target::ttnn::TensorRef *tensorRef) {
  const ::tt::target::Dim2d *tileShape =
      tensorRef->desc()->layout()->memory_desc()->tile_shape();
  return tileShape->x() == 32 && tileShape->y() == 32;
}

::ttnn::DataType getDataType(const ::tt::target::ttnn::TensorRef *tensorRef) {
  return ::tt::runtime::ttnn::utils::toTTNNDataType(
      tensorRef->desc()->layout()->memory_desc()->data_type());
}

::tt::tt_metal::DistributedTensorConfig distributedTensorConfigFromFlatbuffer(
    const ::tt::target::ttnn::DistributionStrategy *strategy) {
  switch (strategy->strategy_type()) {
  case ::tt::target::ttnn::DistributedTensorConfig::ReplicateTensor: {
    return ::tt::tt_metal::ReplicateTensor(
        strategy->strategy_as_ReplicateTensor()->replication_factor());
  }
  case ::tt::target::ttnn::DistributedTensorConfig::ShardTensor: {
    return ::tt::tt_metal::ShardTensor(
        strategy->strategy_as_ShardTensor()->shard_dim());
  }
  case ::tt::target::ttnn::DistributedTensorConfig::ShardTensor2D: {
    uint32_t y = strategy->strategy_as_ShardTensor2D()->shard_mesh()->y();
    uint32_t x = strategy->strategy_as_ShardTensor2D()->shard_mesh()->x();
    ::tt::tt_metal::ShardMesh mesh(y, x);
    return ::tt::tt_metal::ShardTensor2D(mesh);
  }
  case ::tt::target::ttnn::DistributedTensorConfig::AllGatherTensor: {
    return ::tt::tt_metal::AllGatherTensor();
  }
  case ::tt::target::ttnn::DistributedTensorConfig::NONE: {
    LOG_FATAL("Unsupported distributed tensor config");
  }
  }
}
} // namespace tt::runtime::ttnn::operations::utils
