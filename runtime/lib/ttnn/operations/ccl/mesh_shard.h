// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_CCL_MESH_SHARD_H
#define RUNTIME_LIB_TTNN_OPERATIONS_CCL_MESH_SHARD_H

#include "tt/runtime/ttnn/types.h"
#include "tt/runtime/ttnn/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"

#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/tensor/xtensor/partition.hpp"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::MeshShardOp *op, ProgramContext &context);

namespace mesh_shard {
inline ::ttnn::Tensor
FullToShardShape(const ::ttnn::Tensor &input, ::ttnn::MeshDevice &meshDevice,
                 const ::tt::target::ttnn::MeshShardType &shardType,
                 const std::vector<int64_t> &shardShape,
                 const std::vector<int64_t> &shardDims) {
  if (shardType == ::tt::target::ttnn::MeshShardType::Replicate) {
    return ::ttnn::distributed::distribute_tensor(
        input,
        *::ttnn::distributed::replicate_tensor_to_mesh_mapper(meshDevice));
  } else {
    DEBUG_ASSERT(input.get_logical_shape().rank() > 1,
                 "Sharding requires higher than one dimensional tensor.");
    ::ttnn::distributed::Shard2dConfig shard2dConfig{std::nullopt,
                                                     std::nullopt};
    if (shardDims[0] >= 0) {
      shard2dConfig.row_dim = shardDims[0];
    }
    if (shardDims[1] >= 0) {
      shard2dConfig.col_dim = shardDims[1];
    }
    return ::ttnn::distributed::distribute_tensor(
        input, *::ttnn::distributed::shard_tensor_to_2d_mesh_mapper(
                   meshDevice, meshDevice.shape(), shard2dConfig));
  }
}

inline ::ttnn::Tensor
ShardToFullShape(const ::ttnn::Tensor &input, ::ttnn::MeshDevice &meshDevice,
                 const ::tt::target::ttnn::MeshShardType &shardType,
                 const std::vector<int64_t> &shardShape,
                 const std::vector<int64_t> &shardDims) {
  std::vector<::ttnn::Tensor> input_tensors =
      ::ttnn::distributed::get_tensors_from_multi_device_storage(input);
  if (shardType == ::tt::target::ttnn::MeshShardType::Replicate) {
    return input_tensors[0];
  } else {
    bool bFullConcat = std::all_of(shardDims.begin(), shardDims.end(),
                                   [](int n) { return n >= 0; });
    if (bFullConcat) {
      // Full multi-device storage concatenation.
      ::ttnn::distributed::Concat2dConfig concat2dConfig{
          static_cast<int>(shardDims[0]), static_cast<int>(shardDims[1])};
      return ::ttnn::distributed::aggregate_tensor(
          input, *::ttnn::distributed::concat_2d_mesh_to_tensor_composer(
                     meshDevice, concat2dConfig));
    } else {
      // Partial multi-device storage concatenation.
      // Current ttnn api does not support partial multi-device storage
      // concatenation. Thus, xtensor APIs are being called directly from here.
      size_t stride = 0;
      int targetDim = 0;
      size_t iteration = 0;
      if (shardDims[0] >= 0) {
        targetDim = shardDims[0];
        iteration = shardShape[targetDim];
        stride = meshDevice.num_cols();
      } else {
        targetDim = shardDims[1];
        iteration = shardShape[targetDim];
        stride = 1;
      }
      std::vector<::ttnn::Tensor> target_tensors;
      for (size_t i = 0; i < iteration; ++i) {
        target_tensors.push_back(input_tensors[i * stride]);
      }
      return ::ttnn::experimental::xtensor::concat(target_tensors, targetDim);
    }
  }
}

inline ::ttnn::Tensor
mesh_shard(const ::ttnn::Tensor &input, ::ttnn::MeshDevice &meshDevice,
           const ::tt::target::ttnn::MeshShardDirection &shardDirection,
           const ::tt::target::ttnn::MeshShardType &shardType,
           const std::vector<int64_t> &shardShape,
           const std::vector<int64_t> &shardDims) {
  DEBUG_ASSERT(shardType != ::tt::target::ttnn::MeshShardType::Maximal,
               "Unsupported shard type: Maximal.");

  if (shardType == ::tt::target::ttnn::MeshShardType::Identity) {
    // Forward tensor in runtime for identity shard type assuming that the input
    // tensor is pre-sharded by frontend and output tensor is expected to be
    // pre-sharded by frontend. Thus, no sharding is required, but need to makes
    // sure if the tensor is multi-device or multi-device host tensor.
    LOG_ASSERT(input.storage_type() == ::ttnn::StorageType::MULTI_DEVICE ||
                   input.storage_type() ==
                       ::ttnn::StorageType::MULTI_DEVICE_HOST,
               "Input of mesh_shard with identity shard_type must be MULTI "
               "DEVICE or MULTI DEVICE HOST Storage.");
    return input;
  }

  DEBUG_ASSERT(::tt::runtime::ttnn::utils::isOnHost(input.storage_type()),
               "Input of ttnn::mesh_shard should be host tensor for replicate "
               "and devices operations.");
  if (shardDirection ==
      ::tt::target::ttnn::MeshShardDirection::FullToShardShape) {
    return FullToShardShape(input, meshDevice, shardType, shardShape,
                            shardDims);
  } else {
    return ShardToFullShape(input, meshDevice, shardType, shardShape,
                            shardDims);
  }
}
} // namespace mesh_shard

} // namespace tt::runtime::ttnn::operations::ccl

#endif
