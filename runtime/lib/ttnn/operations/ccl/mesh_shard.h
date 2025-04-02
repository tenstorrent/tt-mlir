// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTNN_OPERATIONS_CCL_MESH_SHARD_H
#define RUNTIME_LIB_TTNN_OPERATIONS_CCL_MESH_SHARD_H

#include "tt/runtime/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::MeshShardOp *op, ProgramContext &context);
namespace mesh_shard {
// Forward tensor in runtime for identity shard type assuming that the input
// tensor is pre-sharded by frontend and output tensor is expected to be
// pre-sharded by frontend. Thus, no sharding is required, but need to makes
// sure if the tensor is multi-device or multi-device host tensor.
inline void bypass(const ::tt::target::ttnn::MeshShardOp *op,
                   ProgramContext &context) {
  if (op->shard_type() != ::tt::target::ttnn::MeshShardType::Identity) {
    throw std::runtime_error("Forwording tensor requries identity shard_type.");
  }

  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.getAndValidate(op->in());
  LOG_ASSERT(input.storage_type() == ::ttnn::StorageType::MULTI_DEVICE ||
                 input.storage_type() == ::ttnn::StorageType::MULTI_DEVICE_HOST,
             "Input of mesh_shard with identity shard_type must be MULTI "
             "DEVICE or MULTI DEVICE HOST Storage. id:",
             op->in()->global_id());

  tensorPool.insertAndValidate(op->out(), input);
}
} // namespace mesh_shard
} // namespace tt::runtime::ttnn::operations::ccl

#endif
