// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/operations/utils.h"

#include "operations/ccl/mesh_shard.h"

namespace tt::runtime::ttnn::operations::ccl {

void run(const ::tt::target::ttnn::MeshShardOp *op, ProgramContext &context) {

  DEBUG_ASSERT(op->shard_type() == ::tt::target::MeshShardType::Identity,
               "We only support identity shard_type for mesh_shard op.");

  // Forward tensor in runtime for identity shard type assuming that the input
  // tensor is pre-sharded by frontend and output tensor is expected to be
  // pre-sharded by frontend.
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());
  tensorPool.insertTTNNTensorAndValidate(op->out(), input);
  return;
}
} // namespace tt::runtime::ttnn::operations::ccl
