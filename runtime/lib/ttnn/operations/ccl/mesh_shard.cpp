// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/mesh_shard.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"

#include "tt/runtime/ttnn/operations/utils.h"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::MeshShardOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());
  ::ttnn::MeshDevice &meshDevice =
      context.getSubMesh(op->device()->global_id());
  const ::tt::target::ttnn::MeshShardDirection shardDirection =
      op->shard_direction();
  const ::tt::target::ttnn::MeshShardType shardType = op->shard_type();
  const auto *fbShardShape = op->shard_shape();
  const auto *fbShardDims = op->shard_dims();
  std::vector<int64_t> shardShape(fbShardShape->begin(), fbShardShape->end());
  std::vector<int64_t> shardDims(fbShardDims->begin(), fbShardDims->end());

  ::ttnn::Tensor out =
      ::tt::runtime::ttnn::operations::ccl::mesh_shard::mesh_shard(
          input, meshDevice, shardDirection, shardType, shardShape, shardDims);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
