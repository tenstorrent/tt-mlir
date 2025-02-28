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

::ttnn::operations::conv::conv2d::Conv2dConfig
createConv2dConfig(const ::tt::target::ttnn::Conv2dConfig *memcfg) {
  std::optional<TensorMemoryLayout> shardLayout = std::nullopt;
  if (memcfg->shard_layout()) {
    shardLayout = ::tt::runtime::ttnn::utils::toTTNNTensorMemoryLayout(
        memcfg->shard_layout().value());
  }

  ::ttnn::operations::conv::conv2d::Conv2dConfig conv2dConfig = {
      .dtype = ::tt::runtime::ttnn::utils::toTTNNDataType(memcfg->dtype()),
      .weights_dtype =
          ::tt::runtime::ttnn::utils::toTTNNDataType(memcfg->weights_dtype()),
      .activation = memcfg->activation()->str(),
      .input_channels_alignment = memcfg->input_channels_alignment(),
      .deallocate_activation = memcfg->deallocate_activation(),
      .reallocate_halo_output = memcfg->reallocate_halo_output(),
      .act_block_h_override = memcfg->act_block_h_override(),
      .act_block_w_div = memcfg->act_block_w_div(),
      .reshard_if_not_optimal = memcfg->reshard_if_not_optimal(),
      .override_sharding_config = memcfg->override_sharding_config(),
      .shard_layout = shardLayout,
      .core_grid = std::nullopt,
      .transpose_shards = memcfg->transpose_shards(),
      .output_layout =
          ::tt::runtime::ttnn::utils::toTTNNLayout(memcfg->output_layout()),
  };

  return conv2dConfig;
}
} // namespace tt::runtime::ttnn::operations::utils
