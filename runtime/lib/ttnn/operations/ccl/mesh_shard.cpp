// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/mesh_shard.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/tensor/xtensor/partition.hpp"

namespace tt::runtime::ttnn::operations::ccl {
void FullToShardShape(const ::ttnn::Tensor &input, ::ttnn::Tensor &out,
                      ::ttnn::MeshDevice &meshDevice,
                      const ::tt::target::ttnn::MeshShardType &shardType,
                      const std::vector<int64_t> &shardShape,
                      const std::vector<int64_t> &shardDims) {
  if (shardType == ::tt::target::ttnn::MeshShardType::Replicate) {
    out = ::ttnn::distributed::distribute_tensor(
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
    out = ::ttnn::distributed::distribute_tensor(
        input, *::ttnn::distributed::shard_tensor_to_2d_mesh_mapper(
                   meshDevice, meshDevice.shape(), shard2dConfig));
  }
}

void ShardToFullShape(const ::ttnn::Tensor &input, ::ttnn::Tensor &out,
                      ::ttnn::MeshDevice &meshDevice,
                      const ::tt::target::ttnn::MeshShardType &shardType,
                      const std::vector<int64_t> &shardShape,
                      const std::vector<int64_t> &shardDims) {
  std::vector<::ttnn::Tensor> input_tensors =
      ::ttnn::distributed::get_tensors_from_multi_device_storage(input);
  if (shardType == ::tt::target::ttnn::MeshShardType::Replicate) {
    out = input_tensors[0];
  } else {
    bool bFullConcat = std::all_of(shardDims.begin(), shardDims.end(),
                                   [](int n) { return n >= 0; });
    if (bFullConcat) {
      // Full multi-device storage concatenation.
      ::ttnn::distributed::Concat2dConfig concat2dConfig{
          static_cast<int>(shardDims[0]), static_cast<int>(shardDims[1])};
      out = ::ttnn::distributed::aggregate_tensor(
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
      out = ::ttnn::experimental::xtensor::concat(target_tensors, targetDim);
    }
  }
}

void run(const ::tt::target::ttnn::MeshShardOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.at(op->in()->global_id());
  const ::tt::target::ttnn::MeshShardDirection shardDirection =
      op->shard_direction();
  const ::tt::target::ttnn::MeshShardType shardType = op->shard_type();
  const auto *fbShardShape = op->shard_shape();
  const auto *fbShardDims = op->shard_dims();
  std::vector<int64_t> shardShape(fbShardShape->begin(), fbShardShape->end());
  std::vector<int64_t> shardDims(fbShardDims->begin(), fbShardDims->end());
  DEBUG_ASSERT(::tt::runtime::ttnn::utils::isOnHost(input.storage_type()),
               "Input of ttnn::mesh_shard should be host tensor");

  // Regards manual sharding as no op assuming that the input tensor is
  // pre-sharded by frontend. Thus, no sharding is required, but need to makes
  // sure if the tensor is multi-device host tensor.
  if (shardType == ::tt::target::ttnn::MeshShardType::Manual) {
    LOG_ASSERT(input.storage_type() ==
                   ::tt::tt_metal::StorageType::MULTI_DEVICE_HOST,
               "Input of mesh_shard with manual sharding must be MULTI DEVICE "
               "HOST Storage. id:",
               op->in()->global_id());
    tensorPool.insert_or_assign(op->out()->global_id(), input);
    return;
  }

  if (shardDirection !=
          ::tt::target::ttnn::MeshShardDirection::FullToShardShape &&
      shardDirection !=
          ::tt::target::ttnn::MeshShardDirection::ShardToFullShape) {
    throw std::runtime_error("Unsupported shard direction");
  }

  if (shardType != ::tt::target::ttnn::MeshShardType::Replicate &&
      shardType != ::tt::target::ttnn::MeshShardType::Devices) {
    throw std::runtime_error("Unsupported shard type");
  }

  ::ttnn::MeshDevice &meshDevice =
      context.getSubMesh(op->device()->global_id());

  ::ttnn::Tensor out;
  if (shardDirection ==
      ::tt::target::ttnn::MeshShardDirection::FullToShardShape) {
    FullToShardShape(input, out, meshDevice, shardType, shardShape, shardDims);
  } else {
    ShardToFullShape(input, out, meshDevice, shardType, shardShape, shardDims);
  }
  tensorPool.insert_or_assign(op->out()->global_id(), out);

  DEBUG_ASSERT(::tt::runtime::ttnn::utils::isOnHost(out.storage_type()),
               "Output of ttnn::mesh_shard should be host tensor");
}
} // namespace tt::runtime::ttnn::operations::ccl
