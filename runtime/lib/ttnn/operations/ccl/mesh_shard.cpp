// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_shard.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/tensor/xtensor/partition.hpp"

namespace tt::runtime::ttnn::operations::ccl {

void FullToShardShape(const ::ttnn::Tensor &input, ::ttnn::Tensor &out,
                      ::ttnn::MeshDevice &meshDevice,
                      const ::tt::target::MeshShardType &shardType,
                      const std::vector<int64_t> &shardShape) {
  if (shardType == ::tt::target::MeshShardType::Replicate) {
    out = ::ttnn::distributed::distribute_tensor(
        input, meshDevice,
        *::ttnn::distributed::replicate_tensor_to_mesh_mapper(meshDevice));
  } else {
    LOG_ASSERT(
        input.get_shape().rank() > 1,
        "Sharding requires higher than 2 dimensional tensor. Tensor rank=",
        input.get_shape().rank());
    auto rowMesh = static_cast<size_t>(shardShape[0]);
    auto colMesh = static_cast<size_t>(shardShape[1]);
    int lastDim = input.get_shape().rank() - 1;
    LOG_ASSERT((rowMesh * colMesh) > 1,
               "Sharding requires higher than 1 mesh. shardShape ", rowMesh,
               colMesh);

    ::ttnn::distributed::Shard2dConfig shard2dConfig;
    // last tile replicate
    if (colMesh == 1) {
      if (rowMesh == meshDevice.num_rows()) {
        shard2dConfig = ::ttnn::distributed::Shard2dConfig{
            .row_dim = (lastDim - 1), .col_dim = std::nullopt};
      } else {
        // transpose
        shard2dConfig = ::ttnn::distributed::Shard2dConfig{
            .row_dim = std::nullopt, .col_dim = (lastDim - 1)};
      }
    } else {
      shard2dConfig = ::ttnn::distributed::Shard2dConfig{
          .row_dim = (lastDim - 1), .col_dim = lastDim};
    }

    out = ::ttnn::distributed::distribute_tensor(
        input, meshDevice,
        *::ttnn::distributed::shard_tensor_to_2d_mesh_mapper(
            meshDevice, meshDevice.shape(), shard2dConfig));
  }
}

void ShardToFullShape(const ::ttnn::Tensor &input, ::ttnn::Tensor &out,
                      ::ttnn::MeshDevice &meshDevice,
                      const ::tt::target::MeshShardType &shardType,
                      const std::vector<int64_t> &shardShape) {
  std::vector<::ttnn::Tensor> input_tensors =
      ::ttnn::distributed::get_tensors_from_multi_device_storage(input);
  if (shardType == ::tt::target::MeshShardType::Replicate) {
    out = input_tensors[0];
  } else {
    auto rowMesh = static_cast<size_t>(shardShape[0]);
    auto colMesh = static_cast<size_t>(shardShape[1]);
    int lastDim = input.get_shape().rank() - 1;
    if ((rowMesh * colMesh) ==
        (meshDevice.num_rows() * meshDevice.num_cols())) {
      // Full multi-device storage concatenation
      if (shardShape[0] == 1 || shardShape[1] == 1) {
        out = ::ttnn::distributed::aggregate_tensor(
            input, *::ttnn::distributed::concat_mesh_to_tensor_composer(
                       (shardShape[1] == 1 ? (lastDim - 1) : lastDim)));
      } else {
        out = ::ttnn::distributed::aggregate_tensor(
            input, *::ttnn::distributed::concat_2d_mesh_to_tensor_composer(
                       meshDevice, ::ttnn::distributed::Concat2dConfig{
                                       .row_dim = static_cast<int>(lastDim - 1),
                                       .col_dim = static_cast<int>(lastDim)}));
      }
    } else {
      // Partial multi-device storage concatenation
      // Current ttnn api does not support partial multi-device storage
      // concatenation. Thus, xtensor APIs are being called directly from here.
      std::vector<::ttnn::Tensor> target_tensors;
      bool transpose = (rowMesh != meshDevice.num_rows());
      size_t iteration = (transpose) ? colMesh : rowMesh;
      size_t stride =
          (transpose) ? meshDevice.num_rows() : meshDevice.num_cols();
      for (size_t i = 0; i < iteration; ++i) {
        target_tensors.push_back(input_tensors[i * stride]);
      }
      out = ::ttnn::experimental::xtensor::concat(target_tensors, lastDim - 1);
    }
  }
}

void run(const ::tt::target::ttnn::MeshShardOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.at(op->in()->global_id());
  const ::tt::target::MeshShardDirection shardDirection = op->shard_direction();
  const ::tt::target::MeshShardType shardType = op->shard_type();
  const auto *fbShardShape = op->shard_shape();
  std::vector<int64_t> shardShape(fbShardShape->begin(), fbShardShape->end());

  if (shardDirection != ::tt::target::MeshShardDirection::FullToShardShape &&
      shardDirection != ::tt::target::MeshShardDirection::ShardToFullShape) {
    throw std::runtime_error("Unsupported shard direction");
  }

  if (shardType != ::tt::target::MeshShardType::Replicate &&
      shardType != ::tt::target::MeshShardType::Devices) {
    throw std::runtime_error("Unsupported shard type");
  }

  ::ttnn::MeshDevice &meshDevice =
      context.getSubMesh(op->device()->global_id());

  ::ttnn::Tensor out;
  if (shardDirection == ::tt::target::MeshShardDirection::FullToShardShape) {
    FullToShardShape(input, out, meshDevice, shardType, shardShape);
  } else {
    ShardToFullShape(input, out, meshDevice, shardType, shardShape);
  }
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}

} // namespace tt::runtime::ttnn::operations::ccl
