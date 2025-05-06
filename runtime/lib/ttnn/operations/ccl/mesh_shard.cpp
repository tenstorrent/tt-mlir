// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/operations/utils.h"

#include "operations/ccl/mesh_shard.h"
#include "operations/ccl/mesh_shard_impl.h"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::MeshShardOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());
  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();
  const mesh_shard::MeshShardDirection shardDirection =
      static_cast<mesh_shard::MeshShardDirection>(op->shard_direction());
  const mesh_shard::MeshShardType shardType =
      static_cast<mesh_shard::MeshShardType>(op->shard_type());
  const auto *fbShardShape = op->shard_shape();
  const auto *fbShardDims = op->shard_dims();
  std::vector<int64_t> shardShape(fbShardShape->begin(), fbShardShape->end());
  std::vector<int64_t> shardDims(fbShardDims->begin(), fbShardDims->end());

  if (shardType == mesh_shard::MeshShardType::Identity) {
    // Forward tensor in runtime for identity shard type assuming that the input
    // tensor is pre-sharded by frontend and output tensor is expected to be
    // pre-sharded by frontend. Thus, no sharding is required, but need to makes
    // sure if the tensor is multi-device or multi-device host tensor.
    DEBUG_ASSERT(input.storage_type() == ::ttnn::StorageType::DEVICE ||
                     input.storage_type() ==
                         ::ttnn::StorageType::MULTI_DEVICE_HOST,
                 "Input of mesh_shard with identity shard_type must be Device "
                 " or MULTI DEVICE HOST Storage.");
  } else {
    DEBUG_ASSERT(::tt::runtime::ttnn::utils::isOnHost(input.storage_type()),
                 "Input of ttnn::mesh_shard should be host tensor for "
                 "replicate and devices operations.");
  }

  ::ttnn::Tensor out =
      ::tt::runtime::ttnn::operations::ccl::mesh_shard::mesh_shard(
          input, meshDevice, shardDirection, shardType, shardShape, shardDims);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
